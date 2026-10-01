"""Generator mit Fake-Modellen – keine API-Aufrufe."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

from jev_bench.label import Budget
from jev_bench.llm import ClassifyError, RawCall
from jev_bench.prepare import read_records
from jev_bench.schema import FIELDS, KATEGORIE, DRINGLICHKEIT, render_email
from jev_bench.synth import (GENERATORS, PHISHING_GENERATORS, SCENARIOS, build_specs, generate,
                             prompt_for, to_record)

MAIL = {"absender": "[PERSON_1] <[EMAIL_1]@firma-beispiel.de>", "betreff": "Freigabe",
        "text": "Hallo [ICH], bitte bis 15 Uhr freigeben.", "anhaenge": 0}


def fake(answer: Mapping[str, object] | None = None, fail: bool = False) -> RawCall:
    def call(_prompt: str) -> tuple[Mapping[str, object], int, int]:
        if fail:
            raise ClassifyError("refusal")
        return answer if answer is not None else MAIL, 100, 1000

    return call


def costs(usd: float = 0.01) -> dict[str, Callable[[int, int], float]]:
    def cost(_tokens_in: int, _tokens_out: int) -> float:
        return usd

    return {g: cost for g in GENERATORS}


def test_scenarios_use_valid_labels_and_cover_all_classes() -> None:
    assert {s.dringlichkeit for s in SCENARIOS} == set(DRINGLICHKEIT)
    assert {s.kategorie for s in SCENARIOS} == set(KATEGORIE)
    assert any(s.phishing for s in SCENARIOS) and any(s.antwort_noetig for s in SCENARIOS)
    assert len({s.key for s in SCENARIOS}) == len(SCENARIOS)


def test_build_specs_reproducible_round_robin_and_balanced() -> None:
    specs = build_specs(300, seed=1)
    assert specs == build_specs(300, seed=1)
    phishing = {sc.key for sc in SCENARIOS if sc.phishing}
    normal = [s.generator for s in specs if s.scenario not in phishing]
    assert normal[:6] == list(GENERATORS) * 2  # reihum
    assert all(s.generator in PHISHING_GENERATORS for s in specs if s.scenario in phishing)
    assert all(n >= 80 for n in Counter(s.generator for s in specs).values())  # grob gleich verteilt
    intended = Counter(next(sc.dringlichkeit for sc in SCENARIOS if sc.key == s.scenario) for s in specs)
    assert all(intended[d] >= 50 for d in DRINGLICHKEIT)  # jede Stufe ausreichend vertreten


def test_cc_and_mass_mail_flags() -> None:
    for s in build_specs(400, seed=3):
        if s.scenario in ("info_cc", "cc_entscheidung"):
            assert not s.an_mich_direkt and not s.verteiler and s.empfaenger_anzahl >= 3
        elif s.scenario in ("newsletter", "werbung_druck", "rundmail"):
            assert s.verteiler and not s.an_mich_direkt
        else:
            assert s.an_mich_direkt and s.empfaenger_anzahl == 1


def test_prompt_contains_scenario_but_record_rendering_hides_intent() -> None:
    spec = build_specs(1, seed=5)[0]
    assert "Szenario:" in prompt_for(spec)
    rec = to_record(spec, MAIL, 0.02)
    assert rec["synthetisch"] is True and rec["generator"] == spec.generator
    absicht = rec["absicht"]
    assert isinstance(absicht, dict) and set(absicht) == set(FIELDS)
    rendered = render_email(rec)
    # Prüfer und Kandidaten dürfen Szenario und Absicht nicht sehen
    assert spec.scenario not in rendered and "absicht" not in rendered and "generator" not in rendered


def test_generate_resumes_and_counts_failures(tmp_path: Path) -> None:
    out = tmp_path / "synth.jsonl"
    specs = build_specs(9, seed=2)
    calls = {"gpt-6-sol": fake(), "opus-5.5": fake(fail=True), "gemini-3.8-flash": fake({**MAIL, "text": ""})}
    by_sol = sum(s.generator == "gpt-6-sol" for s in specs)
    assert 0 < by_sol < len(specs)
    ok, failed = generate(specs, calls, out, Budget(10), costs())
    assert (ok, failed) == (by_sol, len(specs) - by_sol)
    assert {r["generator"] for r in read_records(out)} == {"gpt-6-sol"}
    calls = {g: fake() for g in GENERATORS}
    ok, failed = generate(specs, calls, out, Budget(10), costs())
    assert (ok, failed) == (len(specs) - by_sol, 0)
    assert sorted(str(r["id"]) for r in read_records(out)) == [s.id for s in specs]


def test_generate_respects_budget(tmp_path: Path) -> None:
    out = tmp_path / "synth.jsonl"
    calls = {g: fake() for g in GENERATORS}
    ok, _ = generate(build_specs(30, seed=4), calls, out, Budget(0.05), costs(0.01), workers=1)
    assert 5 <= ok <= 7  # je Generator ein Thread; höchstens ein laufender Aufruf pro Thread überzieht


def test_rejects_bool_as_attachment_count(tmp_path: Path) -> None:
    out = tmp_path / "synth.jsonl"
    calls = {g: fake({**MAIL, "anhaenge": True}) for g in GENERATORS}
    ok, failed = generate(build_specs(3, seed=1), calls, out, Budget(10), costs())
    assert (ok, failed) == (0, 3)
