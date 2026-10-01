"""Ground Truth per Mehrheitsentscheid dreier Prüfer-Modelle.

    .venv/bin/python -m jev_bench.label [--sample 300] [--max-usd 6]

Ablauf: zufällige, feste Stichprobe aus data/anonymized/emails.jsonl ziehen, jede
E-Mail von jedem Prüfer klassifizieren lassen (Ergebnisse werden sofort angehängt,
ein Abbruch kann also fortgesetzt werden), dann je Feld per Mehrheit (2 von 3)
entscheiden. Felder ohne Mehrheit landen als "offen" zur manuellen Entscheidung.
Gibt nur Zählwerte aus, keine Inhalte.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from jev_bench.config import load_env, require
from jev_bench.llm import MODELS, Classifier
from jev_bench.prepare import read_records
from jev_bench.schema import FIELDS, render_email

JUDGES = ("gpt-6-sol", "opus-5.5", "gemini-3.8-flash")
WORKERS_PER_MODEL = 2  # hält das Langdock-Limit von 150.000 Tokens/Minute sicher ein


def draw_sample(records: Sequence[Mapping[str, object]], n: int, seed: int) -> list[Mapping[str, object]]:
    eligible = [r for r in records if r.get("text") or r.get("betreff")]
    chosen = random.Random(seed).sample(eligible, min(n, len(eligible)))
    return sorted(chosen, key=lambda r: str(r["id"]))


def load_results(path: Path) -> list[dict[str, object]]:
    return read_records(path) if path.exists() else []


class Budget:
    """Summiert Kosten threadsicher; ist das Limit erreicht, werden keine neuen Aufrufe gestartet."""

    def __init__(self, limit_usd: float, spent_usd: float = 0.0) -> None:
        self.limit = limit_usd
        self.spent = spent_usd
        self._lock = threading.Lock()

    def add(self, usd: float) -> None:
        with self._lock:
            self.spent += usd

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self.spent >= self.limit


def run_judges(sample: Sequence[Mapping[str, object]], classifiers: Sequence[Classifier],
               out: Path, budget: Budget, workers: int = WORKERS_PER_MODEL) -> int:
    """Klassifiziert alle noch fehlenden (Modell, E-Mail)-Paare; Rückgabe: Zahl neuer Ergebnisse."""
    done = {(str(r["model"]), str(r["id"])) for r in load_results(out) if r.get("answer") is not None}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    written = 0

    def work(clf: Classifier, rec: Mapping[str, object]) -> None:
        nonlocal written
        if budget.exhausted:
            return
        result = clf.classify(str(rec["id"]), render_email(rec))
        budget.add(result.cost_usd)
        with write_lock, out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_json(), ensure_ascii=False) + "\n")
            written += 1

    pools = [ThreadPoolExecutor(max_workers=workers) for _ in classifiers]
    futures: list[Future[None]] = []
    try:
        for pool, clf in zip(pools, classifiers, strict=True):
            futures += [pool.submit(work, clf, rec) for rec in sample
                        if (clf.spec.key, str(rec["id"])) not in done]
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    for fut in futures:
        fut.result()  # unerwartete Fehler nicht im Thread verschlucken
    return written


def latest_answers(results: Sequence[Mapping[str, object]]) -> dict[str, dict[str, Mapping[str, object]]]:
    """id -> Modell -> letzte gültige Antwort."""
    answers: dict[str, dict[str, Mapping[str, object]]] = {}
    for r in results:
        answer = r.get("answer")
        if isinstance(answer, Mapping):
            answers.setdefault(str(r["id"]), {})[str(r["model"])] = answer
    return answers


def consensus(votes: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Mehrheit (mind. 2 Stimmen) je Feld; ohne Mehrheit: None und Feld in 'offen'."""
    record: dict[str, object] = {}
    open_fields: list[str] = []
    agreement: dict[str, int] = {}
    for f in FIELDS:
        counts = Counter(json.dumps(v[f]) for v in votes.values() if f in v)
        top, n = counts.most_common(1)[0] if counts else ("null", 0)
        agreement[f] = n
        if n >= 2:
            record[f] = json.loads(top)
        else:
            record[f] = None
            open_fields.append(f)
    record["stimmen"] = len(votes)
    record["einigkeit"] = agreement
    record["offen"] = open_fields
    return record


def summarize(results: Sequence[Mapping[str, object]], truth: Mapping[str, Mapping[str, object]],
              judges: Sequence[str]) -> dict[str, object]:
    per_model: dict[str, object] = {}
    for key in judges:
        rows = [r for r in results if r["model"] == key]
        ok = [r for r in rows if r.get("answer") is not None]
        lat = [float(str(r["latency_ms"])) for r in ok]
        per_model[key] = {
            "ok": len(ok), "fehler": len(rows) - len(ok),
            "fehlerarten": dict(Counter(str(r["error"]).split(":", maxsplit=1)[0]
                                        for r in rows if r.get("error"))),
            "kosten_usd": round(sum(float(str(r["cost_usd"])) for r in rows), 4),
            "latenz_p50_ms": round(statistics.median(lat)) if lat else None,
        }
    answers = latest_answers(results)
    unanimous = {f: sum(1 for t in truth.values() if int(str(_get(t, "einigkeit", f))) == len(judges))
                 for f in FIELDS}
    with_majority = {f: sum(1 for t in truth.values() if t.get(f) is not None) for f in FIELDS}
    vs_majority = {key: {f: _agree_rate(answers, truth, key, f) for f in FIELDS} for key in judges}
    return {
        "emails": len(truth),
        "prueferstatistik": per_model,
        "alle_einig": unanimous,
        "mit_mehrheit": with_majority,
        "offene_felder": sum(len(_list(t.get("offen"))) for t in truth.values()),
        "uebereinstimmung_mit_mehrheit": vs_majority,
        "kosten_gesamt_usd": round(sum(float(str(r["cost_usd"])) for r in results), 4),
    }


def _get(record: Mapping[str, object], outer: str, inner: str) -> object:
    value = record.get(outer)
    return value.get(inner, 0) if isinstance(value, Mapping) else 0


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _agree_rate(answers: Mapping[str, Mapping[str, Mapping[str, object]]],
                truth: Mapping[str, Mapping[str, object]], model: str, f: str) -> float | None:
    pairs = [(answers[i][model][f], t[f]) for i, t in truth.items()
             if t.get(f) is not None and model in answers.get(i, {})]
    return round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ground Truth per Mehrheit dreier Prüfer erzeugen")
    parser.add_argument("--anon", type=Path, default=Path("data/anonymized/emails.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-usd", type=float, default=6.0, help="Kostenbremse (Listenpreise)")
    parser.add_argument("--judges", default=",".join(JUDGES))
    args = parser.parse_args(argv)

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    unknown = [j for j in judges if j not in MODELS]
    if unknown:
        print(f"Unbekannte Modelle: {unknown}", file=sys.stderr)
        return 2
    load_env()
    key = require("LANGDOCK_API_KEY")

    sample = draw_sample(read_records(args.anon), args.sample, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sample_ids.txt").write_text("\n".join(str(r["id"]) for r in sample) + "\n",
                                                 encoding="utf-8")
    raw_path = args.out_dir / "judges.jsonl"
    previous = sum(float(str(r["cost_usd"])) for r in load_results(raw_path))
    budget = Budget(args.max_usd, previous)
    print(f"{len(sample)} E-Mails, Prüfer {judges}, bisher {previous:.2f} $, Limit {args.max_usd:.2f} $",
          file=sys.stderr)

    classifiers = [Classifier(MODELS[j], key) for j in judges]
    run_judges(sample, classifiers, raw_path, budget)

    results = [r for r in load_results(raw_path) if str(r["model"]) in judges]
    sample_ids = {str(r["id"]) for r in sample}
    answers = latest_answers(results)
    truth = {i: consensus(answers.get(i, {})) for i in sorted(sample_ids)}
    with (args.out_dir / "ground_truth.jsonl").open("w", encoding="utf-8") as fh:
        for i, t in truth.items():
            fh.write(json.dumps({"id": i, **t}, ensure_ascii=False) + "\n")

    summary = summarize(results, truth, judges)
    summary["budget_erreicht"] = budget.exhausted
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
