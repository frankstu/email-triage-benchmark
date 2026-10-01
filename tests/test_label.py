"""Labeling mit Fake-Modellen – keine API-Aufrufe."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from jev_bench.label import (Budget, apply_decisions, consensus, draw_sample, latest_answers, run_judges,
                             summarize)
from jev_bench.llm import MODELS, ClassifyError, Classifier, ModelSpec, gemini_schema
from jev_bench.prepare import read_records
from jev_bench.schema import FIELDS, JSON_SCHEMA

A = {"dringlichkeit": "sofort", "kategorie": "persönlich", "antwort_noetig": True, "phishing_verdacht": False}
B = {"dringlichkeit": "bald", "kategorie": "termin", "antwort_noetig": False, "phishing_verdacht": False}
C = {"dringlichkeit": "später", "kategorie": "sonstiges", "antwort_noetig": True, "phishing_verdacht": True}


def spec(key: str, usd_in: float = 1.0) -> ModelSpec:
    return ModelSpec(key, key, "openai", key, usd_in, 0.0)


def fake(key: str, answer: Mapping[str, object] | None = None, *, usd_in: float = 1.0,
         fail: Exception | None = None, calls: list[str] | None = None) -> Classifier:
    lock = threading.Lock()

    def raw(text: str) -> tuple[Mapping[str, object], int, int]:
        with lock:
            if calls is not None:
                calls.append(text)
        if fail is not None:
            raise fail
        return dict(answer or A), 1000, 10

    return Classifier(spec(key, usd_in), "unused", raw_call=raw)


def records(n: int) -> list[dict[str, object]]:
    return [{"id": f"{i:04d}", "betreff": f"B{i}", "text": f"T{i}"} for i in range(1, n + 1)]


# --- Mehrheitsentscheid ------------------------------------------------------------

def test_consensus_majority_and_open_fields() -> None:
    t = consensus({"x": A, "y": A, "z": B})
    assert {f: t[f] for f in FIELDS} == A
    assert t["offen"] == [] and t["einigkeit"] == {f: 2 if A[f] != B[f] else 3 for f in FIELDS}

    t = consensus({"x": A, "y": B, "z": C})
    assert t["dringlichkeit"] is None and t["kategorie"] is None
    assert t["antwort_noetig"] is True  # A und C stimmen überein
    assert t["phishing_verdacht"] is False
    assert t["offen"] == ["dringlichkeit", "kategorie"]


def test_consensus_with_missing_judge() -> None:
    t = consensus({"x": A, "y": B})  # dritter Prüfer ausgefallen
    assert t["stimmen"] == 2
    assert t["phishing_verdacht"] is False  # beide gleich
    assert t["dringlichkeit"] is None  # 1:1 -> offen
    assert consensus({})["offen"] == list(FIELDS)


def test_consensus_does_not_confuse_true_and_1() -> None:
    # bool und Zahl dürfen nicht als gleich gezählt werden
    t = consensus({"x": {**A, "antwort_noetig": True}, "y": {**A, "antwort_noetig": 1}, "z": B})
    assert t["antwort_noetig"] is None


# --- Stichprobe --------------------------------------------------------------------

def test_draw_sample_is_reproducible_and_skips_empty() -> None:
    recs = records(50) + [{"id": "9999", "betreff": "", "text": ""}]
    s1, s2 = draw_sample(recs, 10, seed=7), draw_sample(recs, 10, seed=7)
    assert [r["id"] for r in s1] == [r["id"] for r in s2]
    assert len(s1) == 10 and all(r["id"] != "9999" for r in s1)
    assert [r["id"] for r in s1] == sorted(str(r["id"]) for r in s1)
    assert len(draw_sample(recs, 500, seed=1)) == 50


# --- Ausführung --------------------------------------------------------------------

def test_run_judges_writes_all_results_and_resumes(tmp_path: Path) -> None:
    out = tmp_path / "judges.jsonl"
    calls: list[str] = []
    clfs = [fake("j1", calls=calls), fake("j2", B, calls=calls)]
    assert run_judges(records(5), clfs, out, Budget(100)) == 10
    rows = read_records(out)
    assert len(rows) == 10 and {(r["model"], r["id"]) for r in rows} == {
        (m, f"{i:04d}") for m in ("j1", "j2") for i in range(1, 6)}
    calls.clear()
    assert run_judges(records(6), clfs, out, Budget(100)) == 2  # nur die neue E-Mail
    assert len(calls) == 2


def test_failed_results_are_retried_on_resume(tmp_path: Path) -> None:
    out = tmp_path / "judges.jsonl"
    run_judges(records(2), [fake("j1", fail=ClassifyError("refusal"))], out, Budget(100))
    rows = read_records(out)
    assert all(r["answer"] is None and r["error"] == "refusal" for r in rows)
    assert run_judges(records(2), [fake("j1")], out, Budget(100)) == 2


def test_budget_stops_new_calls(tmp_path: Path) -> None:
    out = tmp_path / "judges.jsonl"
    # 1000 Tokens * 1000 $/Mio = 1 $ pro Aufruf, Limit 3 $
    written = run_judges(records(20), [fake("j1", usd_in=1000.0)], out, Budget(3.0), workers=1)
    assert written == 3


def test_schema_violation_becomes_error(tmp_path: Path) -> None:
    out = tmp_path / "judges.jsonl"
    run_judges(records(1), [fake("j1", {**A, "dringlichkeit": "gestern"})], out, Budget(10))
    row = read_records(out)[0]
    assert row["answer"] is None and "dringlichkeit" in str(row["error"])


def test_unexpected_exception_is_not_swallowed(tmp_path: Path) -> None:
    with pytest.raises(ZeroDivisionError):
        run_judges(records(1), [fake("j1", fail=ZeroDivisionError())], tmp_path / "x.jsonl", Budget(10))


def test_summarize() -> None:
    results: list[dict[str, object]] = []
    for i, (a, b, c) in enumerate([(A, A, A), (A, A, B), (A, B, C)]):
        for model, ans in zip(("j1", "j2", "j3"), (a, b, c), strict=True):
            results.append({"model": model, "id": str(i), "answer": ans, "error": None,
                            "latency_ms": 100.0, "cost_usd": 0.01})
    results.append({"model": "j3", "id": "9", "answer": None, "error": "refusal",
                    "latency_ms": 5.0, "cost_usd": 0.0})
    answers = latest_answers(results)
    truth = {i: consensus(answers[i]) for i in ("0", "1", "2")}
    s = summarize(results, truth, ["j1", "j2", "j3"])
    stats = s["prueferstatistik"]
    assert isinstance(stats, dict) and stats["j3"]["fehler"] == 1 and stats["j3"]["ok"] == 3
    assert s["alle_einig"] == {"dringlichkeit": 1, "kategorie": 1,
                               "antwort_noetig": 1, "phishing_verdacht": 2}
    assert s["offene_felder"] == 2
    agree = s["uebereinstimmung_mit_mehrheit"]
    assert isinstance(agree, dict) and agree["j1"]["dringlichkeit"] == 1.0
    assert s["kosten_gesamt_usd"] == 0.09


# --- Modellkonfiguration -----------------------------------------------------------

def test_model_cost_and_registry() -> None:
    assert MODELS["opus-5.5"].cost(1_000_000, 100_000) == pytest.approx(6.0)
    assert {m.provider for m in MODELS.values()} == {"anthropic", "openai", "google"}


def test_gemini_schema_conversion() -> None:
    g = gemini_schema(JSON_SCHEMA)
    assert g["type"] == "OBJECT" and g["required"] == list(FIELDS)
    props = g["properties"]
    assert isinstance(props, dict)
    assert props["antwort_noetig"] == {"type": "BOOLEAN"}
    assert props["dringlichkeit"]["enum"] == ["sofort", "bald", "später"]
    assert "additionalProperties" not in g


def test_apply_decisions() -> None:
    truth = {"1": consensus({"x": A, "y": B, "z": C})}
    apply_decisions(truth, {"1": {"dringlichkeit": "sofort"}})
    assert truth["1"]["dringlichkeit"] == "sofort"
    assert truth["1"]["offen"] == ["kategorie"] and truth["1"]["manuell"] == ["dringlichkeit"]
    with pytest.raises(SystemExit):
        apply_decisions(truth, {"9": {"dringlichkeit": "bald"}})
    with pytest.raises(SystemExit):
        apply_decisions(truth, {"1": {"farbe": "rot"}})
