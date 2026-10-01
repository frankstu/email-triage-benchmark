"""Liest .eml-Dateien und extrahiert, was für die Klassifizierung gebraucht wird.

Zitierte Verläufe und Signaturen werden abgeschnitten: Sie enthalten die meisten
personenbezogenen Daten und tragen zur Dringlichkeit der neuen Nachricht kaum bei.
Weitergeleitete Nachrichten bleiben erhalten, weil dort der eigentliche Inhalt steht.
"""

from __future__ import annotations

import email
import email.policy
import email.utils
import re
from dataclasses import dataclass
from datetime import datetime
from email.headerregistry import Address
from email.message import EmailMessage, Message
from pathlib import Path

from bs4 import BeautifulSoup

MAX_BODY_CHARS = 2000


@dataclass(frozen=True)
class Mailbox:
    name: str
    addr: str  # klein geschrieben


@dataclass(frozen=True)
class ParsedEmail:
    id: str
    date: datetime | None
    sender: Mailbox | None
    to: tuple[Mailbox, ...]
    cc: tuple[Mailbox, ...]
    subject: str
    body: str
    attachments: int
    is_list: bool  # Verteiler/Newsletter laut List-Id/List-Unsubscribe


# Zeilen, ab denen der zitierte Verlauf beginnt.
_ATTRIBUTION = re.compile(r"^\s*(Am|On)\b.{5,250}\b(schrieb|wrote)\b.*:\s*$", re.IGNORECASE)
_ORIGINAL_MARKER = re.compile(
    r"^\s*-{2,}\s*(Original Message|Ursprüngliche Nachricht|Originalnachricht)\s*-{2,}\s*$",
    re.IGNORECASE,
)
_FORWARD_MARKER = re.compile(
    r"^\s*-*\s*(Forwarded message|Weitergeleitete Nachricht|Begin forwarded message"
    r"|Anfang der weitergeleiteten Nachricht)\s*:?\s*-*\s*$",
    re.IGNORECASE,
)
_FORWARD_HEADER_LINES = 8  # so viele Zeilen nach dem Marker gehören zum Kopf der Weiterleitung
_OUTLOOK_FROM = re.compile(r"^\s*\*?(Von|From)\s*:\*?\s+\S")
_OUTLOOK_FOLLOWUP = re.compile(r"^\s*\*?(Gesendet|Sent|Datum|Date|An|To)\s*:\*?\s")
_SIGNATURE = re.compile(r"^-- ?$")
_QUOTED_LINE = re.compile(r"^\s*>")
_INVISIBLE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad\u034f]")
_HORIZONTAL_SPACE = re.compile(r"[ \t   ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def parse_file(path: Path) -> ParsedEmail:
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    assert isinstance(msg, EmailMessage)  # policy.default liefert EmailMessage
    senders = _mailboxes(msg, "From")
    return ParsedEmail(
        id=path.stem.split("_", 1)[0],
        date=_date(msg),
        sender=senders[0] if senders else None,
        to=_mailboxes(msg, "To"),
        cc=_mailboxes(msg, "Cc"),
        subject=_clean_inline(_header_str(msg, "Subject")),
        body=truncate(clean_body(_body_text(msg))),
        attachments=sum(1 for _ in msg.iter_attachments()),
        is_list=msg["List-Id"] is not None or msg["List-Unsubscribe"] is not None,
    )


def _header_str(msg: EmailMessage, name: str) -> str:
    try:
        value = msg[name]
    except (ValueError, IndexError):  # kaputte Kodierung im Header
        return ""
    return "" if value is None else str(value)


def _mailboxes(msg: EmailMessage, name: str) -> tuple[Mailbox, ...]:
    try:
        header = msg[name]
        addresses: tuple[Address, ...] = header.addresses if header is not None else ()
    except (ValueError, IndexError, AttributeError):
        # Fallback für Header, die der strenge Parser ablehnt
        pairs = email.utils.getaddresses([_header_str(msg, name)])
        return tuple(Mailbox(n.strip(), a.strip().lower()) for n, a in pairs if a)
    return tuple(
        Mailbox(a.display_name.strip(), a.addr_spec.strip().lower())
        for a in addresses
        if a.addr_spec and a.addr_spec != "<>"
    )


def _date(msg: EmailMessage) -> datetime | None:
    raw = _header_str(msg, "Date")
    if not raw:
        return None
    try:
        return email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None


def _body_text(msg: EmailMessage) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    text = _part_content(part)
    if part.get_content_subtype() == "html":
        text = html_to_text(text)
    return text


def _part_content(part: Message) -> str:
    if isinstance(part, EmailMessage):
        try:
            content = part.get_content()
            if isinstance(content, str):
                return content
        except (LookupError, ValueError, AssertionError):
            pass  # unbekannter Zeichensatz o. Ä. – unten tolerant dekodieren
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    # Zitate in HTML-Antworten (Gmail, Apple Mail) entfernen
    for tag in soup.select("blockquote, div.gmail_quote, div.gmail_extra"):
        tag.decompose()
    return soup.get_text("\n")


def _clean_inline(text: str) -> str:
    return _HORIZONTAL_SPACE.sub(" ", _INVISIBLE.sub("", text)).strip()


def clean_body(text: str) -> str:
    text = _INVISIBLE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    lines = lines[: _history_start(lines)]
    lines = [line for line in lines if not _QUOTED_LINE.match(line)]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def _history_start(lines: list[str]) -> int:
    """Index der ersten Zeile von Signatur oder zitiertem Verlauf, sonst len(lines)."""
    forward_header_until = -1
    for i, line in enumerate(lines):
        if _FORWARD_MARKER.match(line):
            forward_header_until = i + _FORWARD_HEADER_LINES
            continue
        if i <= forward_header_until:
            continue  # Von:/Datum:-Block einer Weiterleitung ist kein Verlauf
        if _SIGNATURE.match(line) or _ORIGINAL_MARKER.match(line):
            return i
        # Attributionszeilen werden in Plain-Text oft umbrochen
        joined = f"{line} {lines[i + 1]}" if i + 1 < len(lines) else line
        if _ATTRIBUTION.match(line) or (not line.endswith(":") and _ATTRIBUTION.match(joined)):
            return i
        if _OUTLOOK_FROM.match(line) and any(
            _OUTLOOK_FOLLOWUP.match(nxt) for nxt in lines[i + 1 : i + 5]
        ):
            return i
    return len(lines)


def truncate(text: str, limit: int = MAX_BODY_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > limit // 2 else limit].rstrip() + " …"
