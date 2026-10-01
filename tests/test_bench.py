"""Benchmark-Runner und Auswertung mit Fake-Kandidaten – keine API-Aufrufe."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from jev_bench.bench import Dataset, run_benchmark
from jev_bench.evaluate import (evaluate, field_metrics, generator_bias, hybrid, jev_certainty, label_key,
                                majority_baseline, mcnemar_exact, model_metrics, significance,
                                summary_markdown, wilson_interval)
from jev_bench.label import Budget
from jev_bench.llm import ModelSpec, Result
from jev_bench.prepare import read_records

A = {"dringlichkeit": "sofort", "kategorie": "persönlich", "antwort_noetig": True, "phishing_verdacht": False}
B = {"dringlichkeit": "später", "kategorie": "newsletter_werbung", "antwort_noetig": False,
     "phishing_verdacht": False}


class FakeCandidate:
    def __init__(self, key: str, answer: Mapping[str, object] | None = None, usd: float = 0.0,
                 refuse: bool = False) -> None:
        self.spec = ModelSpec(key, key, "fake", key, 0.0, 0.0)
        self._answer = None if refuse else (answer if answer is not None else A)
        self._usd = usd
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def classify(self, email_id: str, text: str) -> Result:  # pylint: disable=unused-argument
        with self._lock:
            self.calls.append(email_id)
        answer = dict(self._answer) if self._answer is not None else None
        return Result(self.spec.key, email_id, answer, None if answer else "refusal", 10.0, 100, 5, self._usd)


def ds(name: str, n: int) -> Dataset:
    return Dataset(name, [{"id": f"{name}{i}", "betreff": "b", "text": "t"} for i in range(n)])


# --- Runner ------------------------------------------------------------------------

def test_run_benchmark_all_runs_warmup_and_resume(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    c1, c2 = FakeCandidate("m1"), FakeCandidate("m2", B)
    assert run_benchmark([c1, c2], [ds("echt", 3), ds("synth", 2)], 2, out, Budget(10), warmup=2) == 20
    rows = read_records(out)
    assert {(r["model"], r["dataset"], r["run"]) for r in rows} >= {("m1", "echt", 1), ("m2", "synth", 2)}
    assert c1.calls.count("warmup") == 2 and all(r["id"] != "warmup" for r in rows)
    c3 = FakeCandidate("m1")
    assert run_benchmark([c3], [ds("echt", 3), ds("synth", 2)], 3, out, Budget(10), warmup=2) == 5
    assert c3.calls.count("warmup") == 2  # nur der neue dritte Lauf


def test_run_benchmark_skips_warmup_when_nothing_to_do(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    run_benchmark([FakeCandidate("m1")], [ds("echt", 1)], 1, out, Budget(10), warmup=1)
    again = FakeCandidate("m1")
    assert run_benchmark([again], [ds("echt", 1)], 1, out, Budget(10), warmup=1) == 0
    assert not again.calls


def test_run_benchmark_budget(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"
    written = run_benchmark([FakeCandidate("m1", usd=1.0)], [ds("echt", 10)], 1, out, Budget(4.0), warmup=0)
    assert written == 4


# --- Kennzahlen --------------------------------------------------------------------

def test_field_metrics() -> None:
    pairs = [("sofort", "sofort"), ("später", "sofort"), ("später", "später"), ("bald", "später")]
    m = field_metrics(pairs, ["sofort", "bald", "später"])
    assert m["accuracy"] == 0.5
    recall = m["recall"]
    assert isinstance(recall, dict)
    assert recall[label_key("sofort")] == 0.5 and recall[label_key("bald")] is None
    # Macro-F1 über sofort (P=1, R=0,5) und später (P=0,5, R=0,5); "bald" kommt nicht vor
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2, abs=1e-4)
    assert m["confusion"] == {label_key("sofort"): {label_key("sofort"): 1, label_key("später"): 1},
                              label_key("später"): {label_key("später"): 1, label_key("bald"): 1}}


def rows_for(model: str, answers: Mapping[tuple[str, int], Mapping[str, object] | None],
             latency: float = 10.0, cost: float = 0.001, meta: Mapping[str, object] | None = None
             ) -> list[dict[str, object]]:
    return [{"model": model, "dataset": "echt", "id": i, "run": run, "answer": a,
             "error": None if a else "refusal", "latency_ms": latency, "tokens_in": 100, "cost_usd": cost,
             **({"meta": meta} if meta else {})} for (i, run), a in answers.items()]


def test_model_metrics_quality_consistency_latency_cost() -> None:
    truth = {"1": {"id": "1", **A}, "2": {"id": "2", **B}, "3": {"id": "3", **B, "kategorie": None}}
    rows = rows_for("m", {("1", 1): A, ("1", 2): A, ("2", 1): B, ("2", 2): A, ("3", 1): None})
    m = model_metrics(rows, truth)
    assert m["aufrufe"] == 5 and m["ungueltig"] == 1 and m["fehlerarten"] == {"refusal": 1}
    assert m["alle_vier_richtig"] == 0.75  # 3 von 4 vollständigen Antworten
    konstanz = m["konstanz"]
    assert isinstance(konstanz, dict) and konstanz["alle_felder"] == 0.5
    assert m["latenz_ms"] == {"p50": 10.0, "p95": 10.0, "mittel": 10.0}
    assert m["kosten_pro_1000_usd"] == 1.0
    assert m["tokens_in_mittel"] == 100.0 and m["tokens_out_mittel"] == 0.0


def test_majority_baseline() -> None:
    truth = {str(i): {**B} for i in range(9)} | {"x": {**A}}
    assert majority_baseline(truth)["dringlichkeit"] == 0.9


def test_jev_certainty() -> None:
    meta = {"confidence": {"a": 0.9, "b": 0.7}, "noul": {"c": 0.95, "d": 0.6}}
    assert jev_certainty(meta) == pytest.approx(0.2)  # |0,6 − 0,5| · 2
    assert jev_certainty({}) == 0.0


def test_hybrid_escalates_uncertain_cases() -> None:
    truth = {"1": {**A}, "2": {**B}}
    sure = {"confidence": {"d": 1.0}, "noul": {"x": 1.0}}
    unsure = {"confidence": {"d": 0.3}, "noul": {"x": 1.0}}
    jev_rows = rows_for("jev", {("1", 1): A}, latency=5, cost=0.0, meta=sure) + \
        rows_for("jev", {("2", 1): A}, latency=5, cost=0.0, meta=unsure)
    fb_rows = rows_for("fb", {("1", 1): B, ("2", 1): B}, latency=100, cost=0.01)
    by_t = {h["schwelle"]: h for h in hybrid(jev_rows, fb_rows, truth, thresholds=(0.0, 0.5, 1.01))}
    assert by_t[0.0]["eskaliert"] == 0.0 and by_t[0.0]["genauigkeit"]["alle_vier"] == 0.5  # type: ignore[index]
    assert by_t[0.5]["eskaliert"] == 0.5 and by_t[0.5]["genauigkeit"]["alle_vier"] == 1.0  # type: ignore[index]
    assert by_t[0.5]["latenz_mittel_ms"] == 55.0 and by_t[0.5]["kosten_pro_1000_usd"] == 5.0
    assert by_t[1.01]["eskaliert"] == 1.0 and by_t[1.01]["genauigkeit"]["alle_vier"] == 0.5  # type: ignore[index]


def test_generator_bias_and_full_report() -> None:
    truth = {"S1": {**A}, "S2": {**B}}
    emails = {"S1": {"id": "S1", "generator": "gpt-6-sol", "absicht": A},
              "S2": {"id": "S2", "generator": "opus-5.5", "absicht": A}}
    rows = [dict(r, dataset="synthetisch") for r in rows_for("m", {("S1", 1): A, ("S2", 1): A})]
    bias = generator_bias(rows, emails, truth)
    assert bias["gpt-6-sol"] == {"n": 1, "alle_vier_richtig": 1.0, "familie": "openai"}
    assert bias["opus-5.5"]["alle_vier_richtig"] == 0.0  # type: ignore[index]
    report = evaluate(rows, {"synthetisch": truth}, emails)
    entry = report["datensaetze"]["synthetisch"]  # type: ignore[index]
    assert entry["pruefer_vs_absicht"]["dringlichkeit"] == 0.5  # type: ignore[index]
    md = summary_markdown(report)
    assert "## Datensatz: synthetisch (2 E-Mails)" in md and "| m |" in md


def test_wilson_and_mcnemar() -> None:
    lo, hi = wilson_interval(97, 150) or (0.0, 0.0)
    assert lo == pytest.approx(0.567, abs=0.001) and hi == pytest.approx(0.719, abs=0.001)
    assert wilson_interval(0, 0) is None
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(21, 20) == 1.0
    assert mcnemar_exact(39, 14) == pytest.approx(0.0008, abs=0.0002)
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024, abs=1e-4)


def test_significance_uses_run1_only() -> None:
    truth = {"1": {**A}, "2": {**A}}
    rows = rows_for("x", {("1", 1): A, ("2", 1): B, ("2", 2): A}) + rows_for("y", {("1", 1): B, ("2", 1): B})
    sig = significance(rows, truth, ["x", "y"])
    intervals = sig["alle_vier_lauf1"]
    assert isinstance(intervals, dict) and intervals["x"]["k"] == 1 and intervals["x"]["n"] == 2
    assert sig["mcnemar"] == {"x vs y": {"nur_a": 1, "nur_b": 0, "p": 1.0}}
