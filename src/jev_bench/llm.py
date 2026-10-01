"""Einheitlicher Aufruf der LLMs über Langdock (OpenAI-, Anthropic- und Google-Format).

Jeder Aufruf liefert die vier Entscheidungen aus schema.py oder einen Fehler,
plus Latenz, Tokens und Kosten zu Listenpreisen der Anbieter.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import anthropic
import httpx
import openai

from jev_bench.config import LANGDOCK_BASE, LANGDOCK_REGION
from jev_bench.schema import FIELDS, INSTRUCTIONS, JSON_SCHEMA, validate

TOOL_NAME = "klassifikation"
_TOOL_HINT = f"\n\nGib das Ergebnis ausschließlich über das Tool {TOOL_NAME} zurück."
_GOOGLE_RETRIES = 3
_GOOGLE_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    provider: str  # "anthropic" | "openai" | "google"
    model_id: str
    usd_in: float  # Listenpreis pro 1 Mio. Input-Tokens
    usd_out: float  # Listenpreis pro 1 Mio. Output-Tokens
    options: Mapping[str, object] = field(default_factory=dict)

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_in + tokens_out * self.usd_out) / 1_000_000


# Preise: Anthropic laut API-Preisliste, OpenAI/Google laut Anbieterangaben (Stand 10/2026)
MODELS: dict[str, ModelSpec] = {m.key: m for m in (
    ModelSpec("haiku-4.5", "Claude Haiku 4.5", "anthropic", "claude-haiku-4-5-20251001", 1.00, 5.00,
              {"forced_tool": True}),
    ModelSpec("opus-5.5", "Claude Opus 5.5", "anthropic", "claude-opus-5-5", 4.00, 20.00,
              {"forced_tool": False}),  # Opus 5.5 lehnt erzwungenen tool_choice ab
    ModelSpec("gpt-6-luna", "GPT-6 Luna", "openai", "gpt-6-luna", 0.10, 0.50,
              {"reasoning_effort": "none"}),
    ModelSpec("gpt-6-sol", "GPT-6 Sol", "openai", "gpt-6-sol", 2.00, 10.00,
              {"reasoning_effort": "medium"}),
    ModelSpec("gemini-3.8-flash", "Gemini 3.8 Flash", "google", "gemini-3.8-flash", 0.75, 3.75),
)}


class ClassifyError(Exception):
    """Antwort fehlt, wurde verweigert oder passt nicht zum Schema."""


@dataclass(frozen=True)
class Result:
    model: str
    email_id: str
    answer: dict[str, object] | None
    error: str | None
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float

    def to_json(self) -> dict[str, object]:
        return {"model": self.model, "id": self.email_id, "answer": self.answer, "error": self.error,
                "latency_ms": round(self.latency_ms, 1), "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "cost_usd": self.cost_usd}


# Rohaufruf: E-Mail-Text -> (Antwort, Input-Tokens, Output-Tokens)
RawCall = Callable[[str], tuple[Mapping[str, object], int, int]]


def _anthropic_call(spec: ModelSpec, key: str) -> RawCall:
    client = anthropic.Anthropic(base_url=f"{LANGDOCK_BASE}/anthropic/{LANGDOCK_REGION}/", api_key=key)
    forced = bool(spec.options.get("forced_tool"))
    tool: anthropic.types.ToolParam = {"name": TOOL_NAME, "description": "Klassifizierung der E-Mail",
                                       "input_schema": JSON_SCHEMA}  # type: ignore[typeddict-item]

    def call(text: str) -> tuple[Mapping[str, object], int, int]:
        # Langdock reicht output_config.format nicht durch -> Tool-Aufruf mit Schema
        response = client.messages.create(
            model=spec.model_id,
            max_tokens=4000,  # Opus 5.5 denkt immer mit; das zählt in max_tokens
            system=INSTRUCTIONS + _TOOL_HINT,
            messages=[{"role": "user", "content": text}],
            tools=[tool],
            tool_choice={"type": "tool", "name": TOOL_NAME} if forced else {"type": "auto"},
        )
        if response.stop_reason == "refusal":
            raise ClassifyError("refusal")
        answer = next((b.input for b in response.content if b.type == "tool_use"), None)
        if not isinstance(answer, dict):
            raise ClassifyError(f"kein Tool-Aufruf (stop_reason={response.stop_reason})")
        return answer, response.usage.input_tokens, response.usage.output_tokens

    return call


def _openai_call(spec: ModelSpec, key: str) -> RawCall:
    client = openai.OpenAI(base_url=f"{LANGDOCK_BASE}/openai/{LANGDOCK_REGION}/v1", api_key=key)
    effort = str(spec.options.get("reasoning_effort", "low"))

    def call(text: str) -> tuple[Mapping[str, object], int, int]:
        response = client.chat.completions.create(
            model=spec.model_id,
            messages=[{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": text}],
            response_format={"type": "json_schema", "json_schema": {
                "name": TOOL_NAME, "strict": True, "schema": JSON_SCHEMA}},
            reasoning_effort=effort,  # type: ignore[arg-type]  # erlaubte Werte je nach Modell
        )
        content = response.choices[0].message.content
        if not content:
            raise ClassifyError(f"leere Antwort (finish_reason={response.choices[0].finish_reason})")
        usage = response.usage
        tokens_in, tokens_out = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
        return _parse_json(content), tokens_in, tokens_out

    return call


def gemini_schema(schema: Mapping[str, object]) -> dict[str, object]:
    """JSON-Schema -> Gemini-responseSchema (OpenAPI-Teilmenge ohne additionalProperties)."""
    props = schema["properties"]
    assert isinstance(props, dict)
    return {
        "type": "OBJECT",
        "properties": {k: {"type": str(v["type"]).upper(), **({"enum": v["enum"]} if "enum" in v else {})}
                       for k, v in props.items()},
        "required": list(FIELDS),
        "propertyOrdering": list(FIELDS),
    }


def _google_call(spec: ModelSpec, key: str) -> RawCall:
    url = f"{LANGDOCK_BASE}/google/{LANGDOCK_REGION}/v1beta/models/{spec.model_id}:generateContent"
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=120)
    config = {"responseMimeType": "application/json", "responseSchema": gemini_schema(JSON_SCHEMA)}

    def call(text: str) -> tuple[Mapping[str, object], int, int]:
        body = {"systemInstruction": {"parts": [{"text": INSTRUCTIONS}]},
                "contents": [{"role": "user", "parts": [{"text": text}]}],
                "generationConfig": config}
        r = client.post(url, json=body)
        for attempt in range(1, _GOOGLE_RETRIES):
            if r.status_code not in _GOOGLE_RETRY_STATUS:
                break
            time.sleep(2 ** attempt)
            r = client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
        try:
            content = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as e:
            raise ClassifyError(f"keine Antwort ({data.get('promptFeedback') or e})") from e
        meta = data.get("usageMetadata", {})
        # Denk-Tokens werden bei Gemini separat ausgewiesen und als Output berechnet
        out = int(meta.get("candidatesTokenCount", 0)) + int(meta.get("thoughtsTokenCount", 0))
        return _parse_json(content), int(meta.get("promptTokenCount", 0)), out

    return call


def _parse_json(content: str) -> Mapping[str, object]:
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as e:
        raise ClassifyError("kein gültiges JSON") from e
    if not isinstance(answer, dict):
        raise ClassifyError("JSON ist kein Objekt")
    return answer


_FACTORIES: dict[str, Callable[[ModelSpec, str], RawCall]] = {
    "anthropic": _anthropic_call, "openai": _openai_call, "google": _google_call,
}

API_ERRORS = (anthropic.APIError, openai.APIError, httpx.HTTPError)


class Classifier:
    def __init__(self, spec: ModelSpec, key: str, raw_call: RawCall | None = None) -> None:
        self.spec = spec
        self._call = raw_call or _FACTORIES[spec.provider](spec, key)

    def classify(self, email_id: str, text: str) -> Result:
        started = time.perf_counter()
        answer: dict[str, object] | None = None
        error: str | None = None
        tokens_in = tokens_out = 0
        try:
            raw, tokens_in, tokens_out = self._call(text)
            problems = validate(raw)
            if problems:
                error = "Schema: " + "; ".join(problems)
            else:
                answer = dict(raw)
        except ClassifyError as e:
            error = str(e)
        except API_ERRORS as e:
            error = f"{type(e).__name__}: {str(e)[:200]}"
        latency = (time.perf_counter() - started) * 1000
        return Result(self.spec.key, email_id, answer, error, latency, tokens_in, tokens_out,
                      self.spec.cost(tokens_in, tokens_out))
