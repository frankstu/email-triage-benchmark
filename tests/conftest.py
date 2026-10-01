"""Erzeugt synthetische .eml-Dateien – die Tests verwenden keine echten E-Mails."""

from __future__ import annotations

from collections.abc import Callable
from email.message import EmailMessage
from pathlib import Path

import pytest

MakeEml = Callable[..., Path]


@pytest.fixture
def make_eml(tmp_path: Path) -> MakeEml:
    counter = iter(range(1, 100_000))

    def _make(*, sender: str = "Maria Schulz <maria.schulz@gmail.com>",  # pylint: disable=too-many-arguments
              to: str = "Frank Beispiel <frank.beispiel@gmail.com>", cc: str | None = None,
              subject: str = "Test", plain: str | None = "Hallo", html: str | None = None,
              date: str = "Tue, 30 Sep 2026 10:15:00 +0200", headers: dict[str, str] | None = None,
              attachment: bytes | None = None, raw: bytes | None = None,
              folder: Path | None = None) -> Path:
        n = next(counter)
        target = (folder or tmp_path) / f"{n:04d}_2026-09-30.eml"
        target.parent.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            target.write_bytes(raw)
            return target
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to
        if cc:
            msg["Cc"] = cc
        msg["Subject"] = subject
        msg["Date"] = date
        msg["Message-ID"] = f"<test-{n}@example.org>"
        for key, value in (headers or {}).items():
            msg[key] = value
        if plain is not None:
            msg.set_content(plain)
        if html is not None:
            if plain is None:
                msg.set_content(html, subtype="html")
            else:
                msg.add_alternative(html, subtype="html")
        if attachment is not None:
            msg.add_attachment(attachment, maintype="application", subtype="pdf",
                               filename="rechnung.pdf")
        target.write_bytes(msg.as_bytes())
        return target

    return _make
