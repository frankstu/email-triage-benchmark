"""Liest Zugangsdaten aus .env (ohne zusätzliche Abhängigkeit)."""

from __future__ import annotations

import os
from pathlib import Path

LANGDOCK_REGION = "eu"
LANGDOCK_BASE = "https://api.langdock.com"


def load_env(path: Path = Path(".env")) -> None:
    """Setzt KEY=VALUE-Zeilen als Umgebungsvariablen; bereits gesetzte gewinnen."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SystemExit(f"{name} fehlt – bitte in .env eintragen")
    return value
