"""JEV-Anbindung gegen einen Fake-Endpunkt – keine API-Aufrufe."""

from __future__ import annotations

import httpx

from jev_bench.jev import JEV_MODELS, JEV_URL, NOUL_THRESHOLD, JevClassifier, build_questions, parse_response
from jev_bench.schema import DRINGLICHKEIT, FIELDS, KATEGORIE

OK_BODY = {
    "model": "jev-1.13.0",
    "answers": {
        "dringlichkeit": {"type": "choice", "choice": "bald", "confidence": 0.8,
                          "probabilities": {"sofort": 0.1, "bald": 0.85, "später": 0.05}},
        "kategorie": {"type": "choice", "choice": "termin", "confidence": 0.4,
                      "probabilities": {"termin": 0.5, "persönlich": 0.5}},
        "antwort_noetig": {"type": "noul", "noul": 0.5},
        "phishing_verdacht": {"type": "noul", "noul": 0.49},
    },
    "usage": {"input_tokens": 1000, "output_tokens": 60},
}


def response(status: int, body: object) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", JEV_URL))


def scripted(*responses: httpx.Response) -> tuple[list[dict[str, object]], object]:
    sent: list[dict[str, object]] = []
    queue = list(responses)

    def post(body: dict[str, object]) -> httpx.Response:
        sent.append(body)
        return queue.pop(0)

    return sent, post


def test_questions_cover_schema_in_both_languages() -> None:
    for lang in ("en", "de"):
        q = build_questions(lang)
        assert set(q) == set(FIELDS)
        d, k = q["dringlichkeit"], q["kategorie"]
        assert isinstance(d, dict) and isinstance(k, dict)
        assert set(d["criteria"]) == set(DRINGLICHKEIT) and set(k["criteria"]) == set(KATEGORIE)
        assert q["antwort_noetig"] == {"type": "noul", "instructions": q["antwort_noetig"]["instructions"]}  # type: ignore[index]
        assert "[ICH]" in str(d["instructions"])  # Kontext steht in jeder Frage
    assert "How quickly" in str(build_questions("en")["dringlichkeit"]["instructions"])  # type: ignore[index]
    assert "Wie schnell" in str(build_questions("de")["dringlichkeit"]["instructions"])  # type: ignore[index]


def test_parse_response_threshold_and_meta() -> None:
    answer, meta = parse_response(OK_BODY)
    assert answer == {"dringlichkeit": "bald", "kategorie": "termin",
                      "antwort_noetig": True, "phishing_verdacht": False}  # 0,5 -> ja, 0,49 -> nein
    assert NOUL_THRESHOLD == 0.5
    assert meta["confidence"] == {"dringlichkeit": 0.8, "kategorie": 0.4}
    assert meta["noul"] == {"antwort_noetig": 0.5, "phishing_verdacht": 0.49}


def test_classify_success_sends_pinned_model_and_state() -> None:
    sent, post = scripted(response(200, OK_BODY))
    r = JevClassifier(JEV_MODELS["jev"], "k", post=post).classify("0001", "E-Mail-Text")  # type: ignore[arg-type]
    assert r.error is None and r.answer is not None and r.answer["dringlichkeit"] == "bald"
    assert sent[0]["model"] == "jev-1.13.0" and sent[0]["state"] == "E-Mail-Text"
    assert r.tokens_in == 1000 and r.cost_usd == 1000 * 0.042 / 1_000_000
    assert r.meta is not None and r.meta["attempts"] == 1
    assert "meta" in r.to_json()


def test_classify_retries_on_overload_then_succeeds() -> None:
    _, post = scripted(response(529, {}), response(429, {}), response(200, OK_BODY))
    r = JevClassifier(JEV_MODELS["jev"], "k", post=post).classify("1", "x")  # type: ignore[arg-type]
    assert r.error is None and r.meta is not None and r.meta["attempts"] == 3


def test_classify_reports_validation_error_without_retry() -> None:
    sent, post = scripted(response(422, {"error": "bad question"}))
    r = JevClassifier(JEV_MODELS["jev-de"], "k", post=post).classify("1", "x")  # type: ignore[arg-type]
    assert r.answer is None and r.error is not None and r.error.startswith("HTTP 422")
    assert len(sent) == 1


def test_classify_gives_up_after_retries() -> None:
    _, post = scripted(response(529, {}), response(529, {}), response(529, {}))
    r = JevClassifier(JEV_MODELS["jev"], "k", post=post).classify("1", "x")  # type: ignore[arg-type]
    assert r.answer is None and r.error is not None and "529" in r.error


def test_classify_handles_malformed_and_off_schema_answers() -> None:
    _, post = scripted(response(200, {"answers": {}}))
    r = JevClassifier(JEV_MODELS["jev"], "k", post=post).classify("1", "x")  # type: ignore[arg-type]
    assert r.answer is None and r.error is not None and "unlesbar" in r.error
    bad = {**OK_BODY, "answers": {**OK_BODY["answers"],  # type: ignore[dict-item]
                                  "kategorie": {"type": "choice", "choice": "spam", "confidence": 1.0}}}
    _, post = scripted(response(200, bad))
    r = JevClassifier(JEV_MODELS["jev"], "k", post=post).classify("1", "x")  # type: ignore[arg-type]
    assert r.answer is None and r.error is not None and r.error.startswith("Schema")
