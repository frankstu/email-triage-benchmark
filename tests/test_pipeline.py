"""End-to-end über die Kommandozeilen-Einstiegspunkte, inkl. echter spaCy-Modelle."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from conftest import MakeEml
from jev_bench import prepare, review
from jev_bench.ner import SpacyNer, detect_language


def test_detect_language() -> None:
    assert detect_language("Hallo, bitte bis Freitag die Unterlagen schicken.") == "de"
    assert detect_language("Hi, please send the documents by Friday.") == "en"


@pytest.fixture(name="ner", scope="module")
def fixture_ner() -> SpacyNer:
    return SpacyNer()


def test_spacy_finds_names_in_both_languages(ner: SpacyNer) -> None:
    texts = ["Morgen kommt Katharina Vogel ins Büro, bitte Bescheid geben.",
             "Please call Jonathan Miller tomorrow, thanks.", ""]
    found = [[texts[i][s:e] for s, e in spans] for i, spans in enumerate(ner(texts))]
    assert "Katharina Vogel" in found[0]
    assert "Jonathan Miller" in found[1]
    assert found[2] == []


def _corpus(make_eml: MakeEml, folder: Path, n: int) -> None:
    for i in range(n):
        make_eml(folder=folder, sender=f"Person{i} Nachname <p{i}@gmail.com>",
                 subject=f"Rückfrage {i}",
                 plain=(f"Hallo Frank,\nbitte ruf mich unter 0171 {1000000 + i} an.\u2028"
                        f"Katharina Vogel kommt auch.\n\nViele Grüße\nPerson{i}\n\n"
                        f"Am 29.09.2026 schrieb Frank Beispiel <frank.beispiel@gmail.com>:\n> alt"))


def test_prepare_and_review(make_eml: MakeEml, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw, out = tmp_path / "raw", tmp_path / "anon" / "emails.jsonl"
    _corpus(make_eml, raw, 5)
    assert prepare.main(["--raw", str(raw), "--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ausgegeben"] == 5 and summary["nicht_lesbar"] == 0
    assert summary["rest_muster"] == {"CARD": 0, "EMAIL": 0, "IBAN": 0, "PHONE": 0}

    records = prepare.read_records(out)
    text = str(records[0]["text"])
    assert "Katharina" not in text and "Vogel" not in text  # nur von spaCy erkannt
    assert "0171" not in text and "Person0" not in text and "Frank" not in text
    assert "alt" not in text  # Verlauf abgeschnitten

    page = tmp_path / "review.html"
    assert review.main(["--raw", str(raw), "--anon", str(out), "--out", str(page), "--n", "3"]) == 0
    html = page.read_text(encoding="utf-8")
    assert html.count("class='mail'") == 3 and "<mark>[TELEFON]</mark>" in html


def test_prepare_without_files(tmp_path: Path) -> None:
    assert prepare.main(["--raw", str(tmp_path), "--out", str(tmp_path / "x.jsonl")]) == 1


def test_load_1000_mails(make_eml: MakeEml, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw, out = tmp_path / "raw", tmp_path / "emails.jsonl"
    _corpus(make_eml, raw, 1000)
    started = time.perf_counter()
    assert prepare.main(["--raw", str(raw), "--out", str(out)]) == 0
    elapsed = time.perf_counter() - started
    summary = json.loads(capsys.readouterr().out)
    assert summary["ausgegeben"] == 1000
    assert sum(summary["rest_muster"].values()) == 0
    assert elapsed < 120, f"zu langsam: {elapsed:.1f}s"
