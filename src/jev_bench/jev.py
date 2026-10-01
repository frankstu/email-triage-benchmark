"""JEV (TypeSafe AI) als Kandidat – gleiche Schnittstelle wie llm.Classifier.

Alle vier Entscheidungen gehen in einem Aufruf an POST /v1/systemone: Dringlichkeit
und Kategorie als "choice", Antwort und Phishing als "noul" (Wahrscheinlichkeit für
"ja", Schwelle 0,5). Wahrscheinlichkeiten und Konfidenzen landen in Result.meta.

Laut TypeSafe ist Englisch die Hauptsprache; deshalb gibt es die Fragen auf Englisch
("jev") und zum Vergleich auf Deutsch ("jev-de"). Die E-Mails selbst bleiben unverändert.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import httpx

from jev_bench.llm import ModelSpec, Result
from jev_bench.schema import ANTWORT_NOETIG, DRINGLICHKEIT, KATEGORIE, PHISHING_VERDACHT, validate

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-1.13.0"  # fest, damit sich Ergebnisse nicht unbemerkt ändern
NOUL_THRESHOLD = 0.5
_RETRY_STATUS = frozenset({429, 529, 500, 502, 503, 504})
_RETRIES = 3

# Preis laut TypeSafe: 0,042 $ pro 1 Mio. Input-Tokens, Output kostenlos
JEV_MODELS: dict[str, ModelSpec] = {m.key: m for m in (
    ModelSpec("jev", "JEV 1.13 (Fragen englisch)", "typesafe", JEV_MODEL, 0.042, 0.0, {"lang": "en"}),
    ModelSpec("jev-de", "JEV 1.13 (Fragen deutsch)", "typesafe", JEV_MODEL, 0.042, 0.0, {"lang": "de"}),
)}

_CONTEXT = {
    "en": "The state is an email from my inbox. Names, addresses and numbers are pseudonymized "
          "([PERSON_3], [TELEFON] …); [ICH] is me, the recipient. For links only the domain is kept. ",
    "de": "Der Zustand ist eine E-Mail aus meinem Postfach. Namen, Adressen und Nummern sind "
          "pseudonymisiert ([PERSON_3], [TELEFON] …); [ICH] bin ich, der Empfänger. "
          "Bei Links ist nur die Domain erhalten. ",
}
_URGENCY_EN = {
    "sofort": "Act today: deadline today or tomorrow, a direct question that blocks someone, a meeting "
              "very soon, a security warning about my own account.",
    "bald": "Handle within the next few days: request without an acute deadline, invoice with a payment "
            "term, scheduling for later.",
    "später": "No action or eventually: newsletter, advertising, information, automatic notification "
              "without need for action.",
}
_CATEGORY_EN = {
    "persönlich": "A personal message from a person to me.",
    "termin": "Appointment, invitation, booking, reservation.",
    "rechnung_finanzen": "Invoice, payment, bank, insurance, tax, order with costs.",
    "benachrichtigung": "Automatic message from a service (shipping, login, status, account).",
    "newsletter_werbung": "Newsletter, advertising, offers, marketing.",
    "sonstiges": "Fits no other category.",
}
_TEXTS = {
    "en": {
        "dringlichkeit": ("How quickly do I have to react to this email?", _URGENCY_EN),
        "kategorie": ("Which category does this email belong to?", _CATEGORY_EN),
        "antwort_noetig": "Does the email expect a reply from me (not just a click or acknowledgement)?",
        "phishing_verdacht": "Are there signs of phishing or fraud (spoofed sender, pressure, request to "
                             "log in or pay via suspicious links)?",
    },
    "de": {
        "dringlichkeit": ("Wie schnell muss ich auf diese E-Mail reagieren?", DRINGLICHKEIT),
        "kategorie": ("Zu welcher Kategorie gehört diese E-Mail?", KATEGORIE),
        "antwort_noetig": ANTWORT_NOETIG,
        "phishing_verdacht": PHISHING_VERDACHT,
    },
}


def build_questions(lang: str) -> dict[str, object]:
    texts = _TEXTS[lang]
    ctx = _CONTEXT[lang]
    questions: dict[str, object] = {}
    for key in ("dringlichkeit", "kategorie"):
        instructions, criteria = texts[key]
        assert isinstance(criteria, Mapping)
        questions[key] = {"type": "choice", "instructions": ctx + str(instructions),
                          "criteria": dict(criteria)}
    for key in ("antwort_noetig", "phishing_verdacht"):
        questions[key] = {"type": "noul", "instructions": ctx + str(texts[key])}
    return questions


def parse_response(data: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """JEV-Antwort -> (Antwort im gemeinsamen Schema, Zusatzinfos)."""
    answers = data["answers"]
    assert isinstance(answers, Mapping)
    answer: dict[str, object] = {}
    meta: dict[str, object] = {"model": data.get("model"), "confidence": {}, "probabilities": {}, "noul": {}}
    for key in ("dringlichkeit", "kategorie"):
        a = answers[key]
        assert isinstance(a, Mapping)
        answer[key] = a["choice"]
        meta["confidence"][key] = a.get("confidence")  # type: ignore[index]
        meta["probabilities"][key] = a.get("probabilities")  # type: ignore[index]
    for key in ("antwort_noetig", "phishing_verdacht"):
        a = answers[key]
        assert isinstance(a, Mapping)
        p = float(str(a["noul"]))
        answer[key] = p >= NOUL_THRESHOLD
        meta["noul"][key] = p  # type: ignore[index]
    return answer, meta


Post = Callable[[dict[str, object]], httpx.Response]


class JevClassifier:
    def __init__(self, spec: ModelSpec, key: str, post: Post | None = None) -> None:
        self.spec = spec
        self._questions = build_questions(str(spec.options.get("lang", "en")))
        if post is None:
            client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=30)

            def _post(body: dict[str, object]) -> httpx.Response:
                return client.post(JEV_URL, json=body)

            post = _post
        self._post = post

    def classify(self, email_id: str, text: str) -> Result:
        body: dict[str, object] = {"model": self.spec.model_id, "state": text, "questions": self._questions}
        started = time.perf_counter()
        answer: dict[str, object] | None = None
        meta: dict[str, object] | None = None
        error: str | None = None
        tokens_in = tokens_out = 0
        attempts = 0
        try:
            r = self._post(body)
            attempts = 1
            while r.status_code in _RETRY_STATUS and attempts < _RETRIES:
                time.sleep(2 ** attempts * 0.5)
                r = self._post(body)
                attempts += 1
            r.raise_for_status()
            data = r.json()
            usage = data.get("usage", {})
            tokens_in, tokens_out = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
            parsed, meta = parse_response(data)
            meta["attempts"] = attempts
            problems = validate(parsed)
            if problems:
                error = "Schema: " + "; ".join(problems)
            else:
                answer = parsed
        except httpx.HTTPStatusError as e:
            error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        except httpx.HTTPError as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"
        except (KeyError, AssertionError, ValueError, TypeError) as e:
            error = f"Antwort unlesbar: {type(e).__name__}: {e}"
        latency = (time.perf_counter() - started) * 1000
        return Result(self.spec.key, email_id, answer, error, latency, tokens_in, tokens_out,
                      self.spec.cost(tokens_in, tokens_out), meta)
