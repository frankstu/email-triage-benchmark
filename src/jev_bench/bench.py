"""Benchmark der Kandidaten auf echten und synthetischen E-Mails.

    .venv/bin/python -m jev_bench.bench [--runs 2] [--max-usd 4]

Jeder Kandidat bearbeitet seine Anfragen nacheinander (ein Aufruf zur Zeit), damit
die Latenz nicht durch eigene Parallelität verfälscht wird; die Kandidaten laufen
untereinander parallel. Vor der Messung gibt es pro Kandidat Aufwärmaufrufe, die nicht
zählen. Ergebnisse werden sofort angehängt; ein Abbruch kann fortgesetzt werden.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jev_bench.config import load_env, require
from jev_bench.jev import JEV_MODELS, JevClassifier
from jev_bench.label import Budget
from jev_bench.llm import MODELS, Classifier, ModelSpec, Result
from jev_bench.prepare import read_records
from jev_bench.schema import render_email

CANDIDATES = ("jev", "jev-de", "haiku-4.5", "gpt-6-luna")
WARMUP_CALLS = 3
_WARMUP_TEXT = render_email({"absender": "info@beispiel.de", "betreff": "Aufwärmen",
                             "text": "Dies ist eine Testnachricht ohne Inhalt.", "an_mich_direkt": True,
                             "empfaenger_anzahl": 1, "verteiler": False, "anhaenge": 0})


class Candidate(Protocol):
    @property
    def spec(self) -> ModelSpec: ...

    def classify(self, email_id: str, text: str) -> Result: ...


@dataclass(frozen=True)
class Dataset:
    name: str
    records: Sequence[Mapping[str, object]]


def load_dataset(name: str, emails: Path, truth: Path) -> Dataset:
    """Nur E-Mails, für die es eine Ground Truth gibt."""
    ids = {str(t["id"]) for t in read_records(truth)}
    records = [r for r in read_records(emails) if str(r["id"]) in ids]
    missing = ids - {str(r["id"]) for r in records}
    if missing:
        raise SystemExit(f"{name}: {len(missing)} E-Mails der Ground Truth fehlen in {emails}")
    return Dataset(name, records)


def run_benchmark(candidates: Sequence[Candidate], datasets: Sequence[Dataset], runs: int, out: Path,
                  budget: Budget, *, warmup: int = WARMUP_CALLS) -> int:
    """Rückgabe: Zahl neuer Messungen. Bereits gültige Messungen werden übersprungen."""
    done = {(str(r["model"]), str(r["dataset"]), str(r["id"]), int(str(r["run"])))
            for r in (read_records(out) if out.exists() else []) if r.get("answer") is not None}
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    written = 0

    def work(cand: Candidate) -> None:
        nonlocal written
        todo = [(run, ds, rec) for run in range(1, runs + 1) for ds in datasets for rec in ds.records
                if (cand.spec.key, ds.name, str(rec["id"]), run) not in done]
        if not todo:
            return
        for _ in range(warmup):
            budget.add(cand.classify("warmup", _WARMUP_TEXT).cost_usd)
        for run, ds, rec in todo:
            if budget.exhausted:
                return
            result = cand.classify(str(rec["id"]), render_email(rec))
            budget.add(result.cost_usd)
            row = {**result.to_json(), "dataset": ds.name, "run": run}
            with lock, out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
        futures = [pool.submit(work, c) for c in candidates]
    for fut in futures:
        fut.result()
    return written


def make_candidates(keys: Sequence[str]) -> list[Candidate]:
    load_env()
    candidates: list[Candidate] = []
    for key in keys:
        if key in JEV_MODELS:
            candidates.append(JevClassifier(JEV_MODELS[key], require("JEV_API_KEY")))
        elif key in MODELS:
            candidates.append(Classifier(MODELS[key], require("LANGDOCK_API_KEY")))
        else:
            raise SystemExit(f"Unbekannter Kandidat: {key}")
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kandidaten auf beiden Datensätzen messen")
    parser.add_argument("--out", type=Path, default=Path("data/bench/results.jsonl"))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-usd", type=float, default=4.0)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    args = parser.parse_args(argv)

    datasets = [
        load_dataset("echt", Path("data/anonymized/emails.jsonl"), Path("data/labels/ground_truth.jsonl")),
        load_dataset("synthetisch", Path("data/synthetic/emails.jsonl"),
                     Path("data/labels_synth/ground_truth.jsonl")),
    ]
    candidates = make_candidates([c.strip() for c in args.candidates.split(",") if c.strip()])
    previous = sum(float(str(r["cost_usd"])) for r in read_records(args.out)) if args.out.exists() else 0.0
    budget = Budget(args.max_usd, previous)
    print(f"{sum(len(d.records) for d in datasets)} E-Mails × {args.runs} Läufe × {len(candidates)} "
          f"Kandidaten, bisher {previous:.2f} $, Limit {args.max_usd:.2f} $", file=sys.stderr)
    written = run_benchmark(candidates, datasets, args.runs, args.out, budget)
    print(json.dumps({"neu": written, "kosten_usd": round(budget.spent, 4),
                      "budget_erreicht": budget.exhausted}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
