"""Prüft, ob die Modelle über Langdock erreichbar sind und das Schema einhalten.

    .venv/bin/python scripts/probe_langdock.py [modell ...]

Schickt nur eine erfundene Test-E-Mail, keine echten Daten.
"""

from __future__ import annotations

import sys

from jev_bench.config import load_env, require
from jev_bench.llm import MODELS, Classifier
from jev_bench.schema import render_email

TEST_RECORD = {
    "datum": "2026-10-01T09:12:00+02:00",
    "absender": "[PERSON_1] <[EMAIL_1]@gmail.com>",
    "an_mich_direkt": True,
    "empfaenger_anzahl": 1,
    "verteiler": False,
    "anhaenge": 0,
    "betreff": "Unterlagen für morgen",
    "text": "Hallo [ICH],\nkannst du mir bitte bis heute 18 Uhr die Unterlagen für den Termin "
            "morgen schicken? Sonst kann ich nichts vorbereiten.\n\nDanke!\n[PERSON_1]",
}
EXPECTED = {"dringlichkeit": "sofort", "kategorie": "persönlich",
            "antwort_noetig": True, "phishing_verdacht": False}


def main(argv: list[str]) -> int:
    load_env()
    key = require("LANGDOCK_API_KEY")
    text = render_email(TEST_RECORD)
    for name in argv or list(MODELS):
        r = Classifier(MODELS[name], key).classify("probe", text)
        status = "ok" if r.error is None else f"FEHLER {r.error}"
        match = "" if r.answer is None else ("erwartet" if r.answer == EXPECTED else f"abweichend {r.answer}")
        print(f"{name:18} {r.latency_ms:6.0f} ms  in {r.tokens_in:5}  out {r.tokens_out:4}  "
              f"{r.cost_usd * 100:.3f} ct  {status}  {match}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
