"""Auswertung des Benchmarks -> results/metrics.json und results/summary.md.

    .venv/bin/python -m jev_bench.evaluate

Enthält nur Kennzahlen, keine E-Mail-Inhalte – die Ergebnisse dürfen ins Repo.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from jev_bench.prepare import read_records
from jev_bench.schema import DRINGLICHKEIT, FIELDS, KATEGORIE

Row = Mapping[str, object]

CLASSES: dict[str, list[object]] = {
    "dringlichkeit": list(DRINGLICHKEIT),
    "kategorie": list(KATEGORIE),
    "antwort_noetig": [True, False],
    "phishing_verdacht": [True, False],
}
HYBRID_THRESHOLDS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01)
GENERATOR_FAMILY = {"gpt-6-sol": "openai", "opus-5.5": "anthropic", "gemini-3.8-flash": "google"}
CANDIDATE_FAMILY = {"haiku-4.5": "anthropic", "gpt-6-luna": "openai", "jev": "typesafe", "jev-de": "typesafe"}


def label_key(value: object) -> str:
    """Einheitlicher Schlüssel für Klassen in Ausgaben ("sofort", "später", true, false)."""
    return json.dumps(value, ensure_ascii=False)


def _rate(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def field_metrics(pairs: Sequence[tuple[object, object]], classes: Sequence[object]) -> dict[str, object]:
    """pairs: (Vorhersage, Wahrheit)."""
    n = len(pairs)
    correct = sum(p == t for p, t in pairs)
    recall: dict[str, float | None] = {}
    precision: dict[str, float | None] = {}
    f1s: list[float] = []
    for c in classes:
        tp = sum(p == c and t == c for p, t in pairs)
        actual = sum(t == c for _, t in pairs)
        predicted = sum(p == c for p, _ in pairs)
        r, pr = _rate(tp, actual), _rate(tp, predicted)
        recall[label_key(c)] = r
        precision[label_key(c)] = pr
        if actual:  # Macro-F1 nur über Klassen, die in der Wahrheit vorkommen
            f1s.append(2 * (r or 0) * (pr or 0) / ((r or 0) + (pr or 0)) if (r or 0) + (pr or 0) else 0.0)
    confusion: dict[str, dict[str, int]] = {}
    for p, t in pairs:
        row = confusion.setdefault(label_key(t), {})
        key = label_key(p)
        row[key] = row.get(key, 0) + 1
    macro_f1 = round(statistics.mean(f1s), 4) if f1s else None
    return {"n": n, "accuracy": _rate(correct, n), "macro_f1": macro_f1,
            "recall": recall, "precision": precision, "confusion": confusion}


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 1)


def _answers(rows: Iterable[Row]) -> dict[tuple[str, int], Mapping[str, object]]:
    out: dict[tuple[str, int], Mapping[str, object]] = {}
    for r in rows:
        answer = r.get("answer")
        if isinstance(answer, Mapping):
            out[(str(r["id"]), int(str(r["run"])))] = answer
    return out


def model_metrics(rows: Sequence[Row], truth: Mapping[str, Row]) -> dict[str, object]:
    valid = [r for r in rows if isinstance(r.get("answer"), Mapping)]
    answers = _answers(rows)
    fields: dict[str, object] = {}
    for f in FIELDS:
        pairs = [(a[f], truth[i][f]) for (i, _), a in answers.items()
                 if i in truth and truth[i].get(f) is not None]
        fields[f] = field_metrics(pairs, CLASSES[f])
    complete = [(a, truth[i]) for (i, _), a in answers.items()
                if i in truth and all(truth[i].get(f) is not None for f in FIELDS)]
    runs = sorted({run for _, run in answers})
    consistency: dict[str, float | None] = {}
    if len(runs) >= 2:
        ids = {i for i, run in answers if run == runs[0]} & {i for i, run in answers if run == runs[1]}
        first, second = runs[0], runs[1]

        def same(i: str, f: str) -> bool:
            return answers[(i, first)][f] == answers[(i, second)][f]

        for f in FIELDS:
            consistency[f] = _rate(sum(same(i, f) for i in ids), len(ids))
        consistency["alle_felder"] = _rate(sum(all(same(i, f) for f in FIELDS) for i in ids), len(ids))
    latencies = [float(str(r["latency_ms"])) for r in valid]
    costs = [float(str(r["cost_usd"])) for r in rows]
    return {
        "aufrufe": len(rows),
        "ungueltig": len(rows) - len(valid),
        "fehlerarten": dict(Counter(str(r.get("error")).split(":", maxsplit=1)[0]
                                    for r in rows if r.get("answer") is None)),
        "felder": fields,
        "alle_vier_richtig": _rate(sum(all(a[f] == t[f] for f in FIELDS) for a, t in complete),
                                   len(complete)),
        "konstanz": consistency,
        "latenz_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95),
                      "mittel": round(statistics.mean(latencies), 1) if latencies else None},
        "kosten_pro_1000_usd": round(statistics.mean(costs) * 1000, 4) if costs else None,
        "tokens_in_mittel": (round(statistics.mean(float(str(r["tokens_in"])) for r in valid), 1)
                             if valid else None),
    }


def majority_baseline(truth: Mapping[str, Row]) -> dict[str, float | None]:
    """Genauigkeit, wenn man immer die häufigste Klasse rät – zeigt die Schieflage der Daten."""
    out: dict[str, float | None] = {}
    for f in FIELDS:
        values = [t[f] for t in truth.values() if t.get(f) is not None]
        top = Counter(label_key(v) for v in values).most_common(1)[0][1] if values else 0
        out[f] = _rate(top, len(values))
    return out


def jev_certainty(meta: Mapping[str, object]) -> float:
    """Kleinste Sicherheit über alle vier Felder: Choice-Konfidenz bzw. Abstand der
    Noul-Wahrscheinlichkeit von 0,5 (auf 0..1 skaliert)."""
    conf = meta.get("confidence")
    noul = meta.get("noul")
    values: list[float] = []
    if isinstance(conf, Mapping):
        values += [float(str(v)) for v in conf.values() if v is not None]
    if isinstance(noul, Mapping):
        values += [abs(float(str(v)) - 0.5) * 2 for v in noul.values()]
    return min(values) if values else 0.0


def _first_run(rows: Sequence[Row]) -> dict[str, Row]:
    return {str(r["id"]): r for r in rows if int(str(r["run"])) == 1 and isinstance(r.get("answer"), Mapping)}


def _hybrid_point(t: float, ids: Sequence[str], jev: Mapping[str, Row], fb: Mapping[str, Row],
                  truth: Mapping[str, Row]) -> dict[str, object]:
    escalated = 0
    correct = Counter[str]()
    cost = latency = 0.0
    for i in ids:
        meta = jev[i].get("meta")
        use_fb = jev_certainty(meta if isinstance(meta, Mapping) else {}) < t
        answer = (fb[i] if use_fb else jev[i])["answer"]
        assert isinstance(answer, Mapping)
        escalated += use_fb
        for row in (jev[i], fb[i]) if use_fb else (jev[i],):
            cost += float(str(row["cost_usd"]))
            latency += float(str(row["latency_ms"]))
        for f in FIELDS:
            correct[f] += answer[f] == truth[i][f]
        correct["alle_vier"] += all(answer[f] == truth[i][f] for f in FIELDS)
    n = len(ids)
    return {"schwelle": t, "n": n, "eskaliert": _rate(escalated, n),
            "genauigkeit": {k: _rate(v, n) for k, v in correct.items()},
            "kosten_pro_1000_usd": round(cost / n * 1000, 4) if n else None,
            "latenz_mittel_ms": round(latency / n, 1) if n else None}


def hybrid(jev_rows: Sequence[Row], fallback_rows: Sequence[Row], truth: Mapping[str, Row],
           thresholds: Sequence[float] = HYBRID_THRESHOLDS) -> list[dict[str, object]]:
    """JEV entscheidet; liegt seine Sicherheit unter der Schwelle, entscheidet das Fallback-Modell.
    Simuliert aus Lauf 1 beider Modelle; Latenz eskalierter Fälle = JEV + Fallback."""
    jev, fb = _first_run(jev_rows), _first_run(fallback_rows)
    ids = [i for i in jev if i in fb and i in truth and all(truth[i].get(f) is not None for f in FIELDS)]
    return [_hybrid_point(t, ids, jev, fb, truth) for t in thresholds]


def generator_bias(rows: Sequence[Row], emails: Mapping[str, Row],
                   truth: Mapping[str, Row]) -> dict[str, object]:
    """Anteil 'alle vier richtig' je Generator – profitiert ein Kandidat von der eigenen Modellfamilie?"""
    by_gen: dict[str, list[bool]] = {}
    for (i, _), a in _answers(rows).items():
        t = truth.get(i)
        if t is None or any(t.get(f) is None for f in FIELDS):
            continue
        gen = str(emails[i].get("generator"))
        by_gen.setdefault(gen, []).append(all(a[f] == t[f] for f in FIELDS))
    return {g: {"n": len(v), "alle_vier_richtig": _rate(sum(v), len(v)), "familie": GENERATOR_FAMILY.get(g)}
            for g, v in sorted(by_gen.items())}


def intent_agreement(emails: Mapping[str, Row], truth: Mapping[str, Row]) -> dict[str, float | None]:
    """Wie oft bestätigen die Prüfer die beim Erzeugen beabsichtigten Labels?"""
    out: dict[str, float | None] = {}
    for f in FIELDS:
        pairs = [(e["absicht"][f], truth[i][f]) for i, e in emails.items()  # type: ignore[index]
                 if i in truth and truth[i].get(f) is not None and isinstance(e.get("absicht"), Mapping)]
        out[f] = _rate(sum(a == b for a, b in pairs), len(pairs))
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95-%-Konfidenzintervall für einen Anteil k/n (Wilson)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(centre - half, 4), round(centre + half, 4)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exakter zweiseitiger McNemar-Test (Binomialtest auf den diskordanten Paaren)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def _all_four_run1(rows: Sequence[Row], truth: Mapping[str, Row]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for (i, run), a in _answers(rows).items():
        t = truth.get(i)
        if run == 1 and t is not None and all(t.get(f) is not None for f in FIELDS):
            out[i] = all(a[f] == t[f] for f in FIELDS)
    return out


def significance(rows: Sequence[Row], truth: Mapping[str, Row], models: Sequence[str]) -> dict[str, object]:
    """'Alle vier richtig' in Lauf 1: Konfidenzintervalle je Modell und paarweise McNemar-Tests.
    Nur Lauf 1, weil die zwei Läufe derselben E-Mail nicht unabhängig sind."""
    correct = {m: _all_four_run1([r for r in rows if r["model"] == m], truth) for m in models}
    intervals = {m: {"k": sum(c.values()), "n": len(c), "anteil": _rate(sum(c.values()), len(c)),
                     "ki95": wilson_interval(sum(c.values()), len(c))} for m, c in correct.items()}
    pairs: dict[str, object] = {}
    for x, a in enumerate(models):
        for b in models[x + 1:]:
            common = correct[a].keys() & correct[b].keys()
            only_a = sum(correct[a][i] and not correct[b][i] for i in common)
            only_b = sum(correct[b][i] and not correct[a][i] for i in common)
            pairs[f"{a} vs {b}"] = {"nur_a": only_a, "nur_b": only_b, "p": mcnemar_exact(only_a, only_b)}
    return {"alle_vier_lauf1": intervals, "mcnemar": pairs}


def evaluate(results: Sequence[Row], truths: Mapping[str, Mapping[str, Row]],
             synth_emails: Mapping[str, Row]) -> dict[str, object]:
    models = sorted({str(r["model"]) for r in results})
    report: dict[str, object] = {"datensaetze": {}}
    for ds, truth in truths.items():
        rows = [r for r in results if r["dataset"] == ds]
        per_model = {m: model_metrics([r for r in rows if r["model"] == m], truth) for m in models}
        entry: dict[str, object] = {
            "emails": len(truth),
            "verteilung": {f: dict(Counter(label_key(t[f]) for t in truth.values()))
                           for f in FIELDS},
            "immer_haeufigste_klasse": majority_baseline(truth),
            "modelle": per_model,
            "hybrid": {fb: hybrid([r for r in rows if r["model"] == "jev"],
                                  [r for r in rows if r["model"] == fb], truth)
                       for fb in models if fb not in ("jev", "jev-de")} if "jev" in models else {},
        }
        if ds == "synthetisch":
            entry["generator_bias"] = {
                m: generator_bias([r for r in rows if r["model"] == m], synth_emails, truth) for m in models}
            entry["pruefer_vs_absicht"] = intent_agreement(synth_emails, truth)
        entry["signifikanz"] = significance(rows, truth, models)
        report["datensaetze"][ds] = entry  # type: ignore[index]
    return report


def _pct(v: object) -> str:
    return "–" if v is None else f"{float(str(v)) * 100:.1f} %"


def summary_markdown(report: Mapping[str, object]) -> str:
    lines = ["# Ergebnisse", ""]
    datasets = report["datensaetze"]
    assert isinstance(datasets, Mapping)
    for ds, entry in datasets.items():
        assert isinstance(entry, Mapping)
        lines += [f"## Datensatz: {ds} ({entry['emails']} E-Mails)", "",
                  "| Modell | alle 4 richtig | Dringlichkeit | Recall „sofort“ | Kategorie | Antwort nötig "
                  "| Phishing (Recall) | Konstanz | Latenz p50 | p95 | $/1000 | ungültig |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        models = entry["modelle"]
        assert isinstance(models, Mapping)
        for m, mm in models.items():
            f = mm["felder"]
            lat = mm["latenz_ms"]
            lines.append(
                f"| {m} | {_pct(mm['alle_vier_richtig'])} | {_pct(f['dringlichkeit']['accuracy'])} "
                f"| {_pct(f['dringlichkeit']['recall'].get(label_key('sofort')))} "
                f"| {_pct(f['kategorie']['accuracy'])} | {_pct(f['antwort_noetig']['accuracy'])} "
                f"| {_pct(f['phishing_verdacht']['recall'].get(label_key(True)))} "
                f"| {_pct(mm['konstanz'].get('alle_felder'))} | {lat['p50']} ms | {lat['p95']} ms "
                f"| {mm['kosten_pro_1000_usd']} | {mm['ungueltig']}/{mm['aufrufe']} |")
        base = entry["immer_haeufigste_klasse"]
        assert isinstance(base, Mapping)
        lines += ["", "Immer die häufigste Klasse raten: " +
                  ", ".join(f"{k} {_pct(v)}" for k, v in base.items()), ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark auswerten")
    parser.add_argument("--results", type=Path, default=Path("data/bench/results.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    truths = {
        "echt": {str(t["id"]): t for t in read_records(Path("data/labels/ground_truth.jsonl"))},
        "synthetisch": {str(t["id"]): t for t in read_records(Path("data/labels_synth/ground_truth.jsonl"))},
    }
    synth = {str(e["id"]): e for e in read_records(Path("data/synthetic/emails.jsonl"))}
    report = evaluate(read_records(args.results), truths, synth)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    (args.out_dir / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print(summary_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
