"""Lokale Pseudonymisierung von E-Mails – ohne Netzwerkzugriff.

Vorgehen pro Text: Alle Fundstellen (E-Mail-Adressen, URLs, Telefonnummern, IBANs,
Kartennummern, Adressen, lange Nummern, Personennamen) werden als Spannen gesammelt,
Überlappungen nach Priorität aufgelöst und dann von hinten ersetzt. So stören sich
die Erkennungen nicht gegenseitig, und die Namenserkennung sieht den Originaltext.

Personen und Adressen erhalten über das ganze Korpus feste Platzhalter
([PERSON_3], [EMAIL_2]@gmail.com), damit erkennbar bleibt, wer wem schreibt.
Organisationsnamen und Absender-Domains bleiben erhalten, weil sie für die
Klassifizierung (Newsletter, Rechnung, Phishing) wichtig sind – außer eine Domain
enthält einen Personennamen.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from jev_bench.parse import Mailbox, ParsedEmail

# Liefert pro Text die Zeichenbereiche erkannter Personennamen.
NerFn = Callable[[Sequence[str]], list[list[tuple[int, int]]]]

FREEMAIL_DOMAINS = frozenset({
    "gmail.com", "googlemail.com", "gmx.de", "gmx.net", "gmx.at", "gmx.ch", "web.de",
    "t-online.de", "freenet.de", "posteo.de", "mailbox.org", "outlook.com", "outlook.de",
    "hotmail.com", "hotmail.de", "live.com", "live.de", "msn.com", "icloud.com", "me.com",
    "mac.com", "yahoo.com", "yahoo.de", "aol.com", "aol.de", "arcor.de", "online.de",
    "proton.me", "protonmail.com", "kabelmail.de", "vodafone.de", "email.de", "mail.de",
})

_ROLE_LOCALPART = re.compile(
    r"^(no-?reply|do-?not-?reply|donotreply|noreply|info|newsletters?|news|service|services"
    r"|support|kontakt|contact|hello|hallo|team|mail|mailer|notifications?|notify"
    r"|benachrichtigung(en)?|rechnung(en)?|billing|invoices?|orders?|bestellung(en)?|versand"
    r"|shipping|shipment|account|accounts|security|sicherheit|alerts?|updates?|marketing|presse"
    r"|press|jobs|karriere|careers|feedback|help|hilfe|admin|webmaster|postmaster"
    r"|mailer-daemon|bounces?|reply|antwort|kundenservice|customerservice|crm|events?"
    r"|community|digest|calendar-notification|auto-?confirm|confirm|verify|receipts?"
    r"|payments?|zahlung(en)?|buchung(en)?|booking|reservations?|tickets?|status|system)"
    r"([.\-_+].*)?$",
    re.IGNORECASE,
)

_ORG_HINT = re.compile(
    r"(?i)(\b(gmbh|ag|kg|ohg|e\.\s?v\.?|ev|ug|se|inc|ltd|llc|corp|co|team|service|services"
    r"|support|newsletter|news|info|shop|store|bank|sparkasse|volksbank|versicherung|verlag"
    r"|online|club|verein|kundenservice|noreply|no-reply|portal|app|community|stiftung|amt"
    r"|stadt|gemeinde|praxis|kanzlei|hotel|airlines?|official|offiziell)\b|[.@&|/:])"
)

_NAME_PARTICLES = frozenset({"von", "van", "der", "den", "de", "del", "di", "da", "du", "la",
                             "le", "zu", "zum", "zur", "ten", "ter", "vom"})
_TITLES = re.compile(r"(?i)^(herrn?|frau|dr\.?|prof\.?|mr\.?|mrs\.?|ms\.?|dipl\.-?\w*\.?)\s+")
_NAME_TOKEN = re.compile(r"[^\W\d_][^\W\d_'’\-]*(?:[-'’][^\W\d_]+)*")
_MIN_TOKEN_LEN = 3

# --- Muster für strukturierte Daten -------------------------------------------------

EMAIL_RE = re.compile(r"(?<![\w.%+\-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'\]\)]+")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
CARD_RE = re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])")
PHONE_RE = re.compile(
    r"(?<![\w+])(?:(?:\+|00)[1-9]\d{0,2}|\(?0[1-9]\d{1,4}\)?)"  # Ländervorwahl oder 0-Vorwahl
    r"(?:[ \-/]{0,3}\(?\d+\)?){1,6}"
    r"(?![\w.,]?\d)"
)
ADDRESS_RE = re.compile(
    r"\b[A-ZÄÖÜ][\wäöüß\-]*(?:straße|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer"
    r"|chaussee|steig|pfad|markt)\s+\d{1,4}\s?[a-zA-Z]?\b"
)
PLZ_ORT_RE = re.compile(r"(?<![\w\-])\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?\b")
LONG_NUMBER_RE = re.compile(r"(?<![\w.,])\d{6,}(?![\w]|[.,]\d)")

_GREETING_RE = re.compile(
    r"(?:\b(?:Hallo|Hi|Hey|Moin|Servus|Liebe[rs]?|Lieber|Dear|Hello|Guten\s+(?:Tag|Morgen|Abend))"
    r"|\b(?:Sehr\s+geehrte[rs]?\s+)?(?:Herr|Frau|Herrn|Mr\.?|Mrs\.?|Ms\.?|Dr\.|Prof\.))"
    r"[ \t]+((?:Dr\.\s+|Prof\.\s+)?[A-ZÄÖÜ][\wäöüß'’\-]+(?:[ \t]+[A-ZÄÖÜ][\wäöüß'’\-]+)?)"
)
_CLOSING_RE = re.compile(
    r"(?im)^[ \t]*(?:[\wäöüß]+[ \t]+){0,2}"
    r"(?:grüße|gruß|grüßen|grüsse|gruss|regards|wishes|best|cheers|thanks|danke|lg|vg|mfg|bg)"
    r"(?:[ \t]+[\wäöüß]+){0,3}[ \t]*[,!.]?[ \t]*\n[ \t]*"
    r"([A-ZÄÖÜ][\wäöüß'’\-]+(?:[ \t]+[A-ZÄÖÜ][\wäöüß'’\-]+)?)[ \t]*$"
)
# Wörter nach Anrede oder Grußformel, die keine Namen sind
_GREETING_WORDS = frozenset({
    "hallo", "hi", "hey", "liebe", "lieber", "liebes", "dear", "hello", "team", "zusammen",
    "alle", "all", "everyone", "leute", "ihr", "ihre", "sie", "du", "dein", "deine", "euer",
    "eure", "your", "the", "guten", "morgen", "tag", "abend", "kolleginnen", "kollegen",
    "freunde", "freundinnen", "mitglieder", "kunde", "kundin", "kunden", "nutzer", "nutzerin",
    "leser", "leserin", "abonnent", "abonnentin", "customer", "member", "user", "sir", "madam",
    "damen", "herren", "nachbarn", "eltern", "familie",
})
_NOT_A_CITY = frozenset({"euro", "eur", "dollar", "franken", "pfund", "punkte", "stück",
                         "meter", "kilometer", "mal", "mitglieder", "menschen", "teilnehmer"})

# Höhere Priorität gewinnt bei Überlappung.
_PRIORITY = {"EMAIL": 6, "URL": 6, "IBAN": 5, "CARD": 5, "PHONE": 4, "ADDRESS": 4,
             "PLZ_ORT": 3, "NUMBER": 2, "PERSON": 1}
_FIXED_TAGS = {"IBAN": "[IBAN]", "CARD": "[KARTE]", "PHONE": "[TELEFON]",
               "ADDRESS": "[ADRESSE]", "PLZ_ORT": "[PLZ_ORT]", "NUMBER": "[NUMMER]"}
_TRAILING_URL_PUNCT = ".,;:!?…"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    kind: str


@dataclass
class Stats:
    replaced: Counter[str] = field(default_factory=Counter)
    emails_total: int = 0
    emails_without_text: int = 0
    known_name_tokens: int = 0


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _is_iban(value: str) -> bool:
    # Mindestens 10 Ziffern, damit Wörter in Großbuchstaben nicht als IBAN gelten
    return sum(ch.isdigit() for ch in value) >= 10


def name_tokens(name: str) -> list[str]:
    """Namensbestandteile ab Mindestlänge, ohne Adelspartikel und Titel."""
    return [t for t in _NAME_TOKEN.findall(_strip_title(name))
            if len(t) >= _MIN_TOKEN_LEN and t.lower() not in _NAME_PARTICLES]


def _strip_title(name: str) -> str:
    name = name.strip().strip("\"'")
    while True:
        stripped = _TITLES.sub("", name)
        if stripped == name:
            return name
        name = stripped


def _normalize_display_name(name: str) -> str:
    """'Schulz, Maria' -> 'Maria Schulz'."""
    name = name.strip().strip("\"'")
    if name.count(",") == 1:
        last, first = (p.strip() for p in name.split(","))
        if last and first:
            return f"{first} {last}"
    return name


def looks_like_person(display_name: str, addr: str) -> bool:
    name = _normalize_display_name(display_name)
    if not name or any(ch.isdigit() for ch in name) or _ORG_HINT.search(name):
        return False
    tokens = _NAME_TOKEN.findall(_strip_title(name))
    if not 1 <= len(tokens) <= 4 or not all(t[0].isupper() for t in tokens
                                            if t.lower() not in _NAME_PARTICLES):
        return False
    local, _, domain = addr.partition("@")
    if _ROLE_LOCALPART.match(local) and domain not in FREEMAIL_DOMAINS:
        return False
    if domain in FREEMAIL_DOMAINS:
        return True
    local_l = local.lower()
    if any(len(t) >= _MIN_TOKEN_LEN and t.lower() in local_l for t in tokens):
        return True
    return len(tokens) >= 2


def infer_identity(emails: Iterable[ParsedEmail]) -> tuple[str | None, str | None]:
    """Eigene Adresse = häufigster Empfänger; eigener Name = häufigster Anzeigename dazu."""
    addr_counts: Counter[str] = Counter()
    names: dict[str, Counter[str]] = {}
    for mail in emails:
        for mb in (*mail.to, *mail.cc):
            addr_counts[mb.addr] += 1
            if mb.name:
                names.setdefault(mb.addr, Counter())[_normalize_display_name(mb.name)] += 1
    if not addr_counts:
        return None, None
    own = addr_counts.most_common(1)[0][0]
    own_names = names.get(own)
    return own, (own_names.most_common(1)[0][0] if own_names else None)


class Pseudonymizer:
    """Vergibt korpusweit stabile Platzhalter für Personen und E-Mail-Adressen."""

    def __init__(self, own_addr: str | None, own_name: str | None) -> None:
        self._own_addr = own_addr
        self._persons: dict[str, int] = {}
        self._emails: dict[str, int] = {}
        self._own_tokens = {t.lower() for t in name_tokens(own_name or "")}
        self._all_person_tokens: set[str] = set(self._own_tokens)

    @property
    def person_tokens(self) -> set[str]:
        return self._all_person_tokens

    def register_person(self, name: str) -> None:
        self.person(name)

    def person(self, name: str) -> str:
        tokens = [t.lower() for t in name_tokens(_normalize_display_name(name))]
        if not tokens:
            return "[PERSON]"
        if self._own_tokens and set(tokens) <= self._own_tokens:
            return "[ICH]"
        key = " ".join(tokens)
        pid = self._persons.get(key)
        if pid is None and len(tokens) == 1:
            # Einzelner Teilname ("Maria" zu "Maria Schulz") übernimmt die bekannte Nummer.
            # Mehrteilige Namen nicht: "Maria Müller" ist eine andere Person.
            pid = self._persons.get(tokens[0])
        if pid is None:
            pid = len(set(self._persons.values())) + 1
        self._persons.setdefault(key, pid)
        for t in tokens:
            self._persons.setdefault(t, pid)
            self._all_person_tokens.add(t)
        return f"[PERSON_{pid}]"

    def email(self, addr: str) -> str:
        addr = addr.lower()
        if self._own_addr and addr == self._own_addr:
            return "[ICH]"
        local, _, domain = addr.partition("@")
        if domain not in FREEMAIL_DOMAINS and _ROLE_LOCALPART.match(local) \
                and not self._domain_has_name(domain):
            return addr  # Funktionsadresse eines Unternehmens, nicht personenbezogen
        eid = self._emails.setdefault(addr, len(self._emails) + 1)
        return f"[EMAIL_{eid}]@{self.domain(domain)}"

    def domain(self, domain: str) -> str:
        if domain in FREEMAIL_DOMAINS or not self._domain_has_name(domain):
            return domain
        return "[DOMAIN]"

    def _domain_has_name(self, domain: str) -> bool:
        labels = domain.lower().split(".")[:-1]  # ohne TLD
        return any(tok in label for label in labels for tok in self._all_person_tokens)


def _find_structured(text: str) -> list[Span]:
    spans: list[Span] = []
    for kind, pattern in (("EMAIL", EMAIL_RE), ("ADDRESS", ADDRESS_RE), ("NUMBER", LONG_NUMBER_RE)):
        spans += [Span(m.start(), m.end(), kind) for m in pattern.finditer(text)]
    spans += [Span(m.start(), m.end(), "IBAN") for m in IBAN_RE.finditer(text) if _is_iban(m.group())]
    spans += [Span(m.start(), m.end(), "PLZ_ORT") for m in PLZ_ORT_RE.finditer(text)
              if m.group().split()[-1].lower() not in _NOT_A_CITY]
    for m in URL_RE.finditer(text):
        end = m.end()
        while end > m.start() and text[end - 1] in _TRAILING_URL_PUNCT:
            end -= 1
        spans.append(Span(m.start(), end, "URL"))
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            spans.append(Span(m.start(), m.end(), "CARD"))
    for m in PHONE_RE.finditer(text):
        if sum(ch.isdigit() for ch in m.group()) >= 7:
            spans.append(Span(m.start(), m.end(), "PHONE"))
    return spans


def _find_greeting_names(text: str) -> list[Span]:
    spans: list[Span] = []
    for pattern in (_GREETING_RE, _CLOSING_RE):
        for m in pattern.finditer(text):
            if m.group(1).split()[0].lower() not in _GREETING_WORDS:
                spans.append(Span(m.start(1), m.end(1), "PERSON"))
    return spans


def _known_name_pattern(tokens: Iterable[str]) -> re.Pattern[str] | None:
    alternatives = sorted({t for t in tokens if len(t) >= _MIN_TOKEN_LEN}, key=len, reverse=True)
    if not alternatives:
        return None
    # Ganze Wörter, unabhängig von der Schreibweise ("McDonald"); ob das Vorkommen
    # großgeschrieben ist, prüft der Aufrufer
    body = "|".join(re.escape(t) for t in alternatives)
    return re.compile(rf"(?<![\w\-])(?:{body})(?![\w\-])", re.IGNORECASE)


def _resolve(spans: Iterable[Span]) -> list[Span]:
    chosen: list[Span] = []
    for span in sorted(spans, key=lambda s: (-_PRIORITY[s.kind], -(s.end - s.start), s.start)):
        if span.end > span.start and all(span.end <= c.start or span.start >= c.end for c in chosen):
            chosen.append(span)
    return sorted(chosen, key=lambda s: s.start)


def _merge_adjacent_persons(text: str, spans: list[Span]) -> list[Span]:
    """'[Maria] [Schulz]' als eine Person behandeln, wenn nur Leerzeichen dazwischen."""
    merged: list[Span] = []
    for span in spans:
        prev = merged[-1] if merged else None
        if prev and prev.kind == span.kind == "PERSON" and text[prev.end:span.start].strip(" \t") == "":
            merged[-1] = Span(prev.start, span.end, "PERSON")
        else:
            merged.append(span)
    return merged


class TextAnonymizer:
    def __init__(self, pseudonymizer: Pseudonymizer, stats: Stats) -> None:
        self._p = pseudonymizer
        self._stats = stats
        self._known = _known_name_pattern(pseudonymizer.person_tokens)

    def anonymize(self, text: str, ner_spans: Sequence[tuple[int, int]] = ()) -> str:
        spans = _find_structured(text) + _find_greeting_names(text)
        spans += [Span(s, e, "PERSON") for s, e in ner_spans if _plausible_ner_name(text[s:e])]
        if self._known is not None:
            spans += [Span(m.start(), m.end(), "PERSON") for m in self._known.finditer(text)
                      if m.group()[0].isupper()]
        resolved = _merge_adjacent_persons(text, _resolve(spans))
        out: list[str] = []
        pos = 0
        for span in resolved:
            out.append(text[pos:span.start])
            out.append(self._replacement(span.kind, text[span.start:span.end]))
            self._stats.replaced[span.kind] += 1
            pos = span.end
        out.append(text[pos:])
        return "".join(out)

    def _replacement(self, kind: str, value: str) -> str:
        if kind == "PERSON":
            return self._p.person(value)
        if kind == "EMAIL":
            return self._p.email(value)
        if kind == "URL":
            host = urlsplit(value if "://" in value else f"http://{value}").hostname or ""
            host = host.removeprefix("www.")
            return f"[URL:{self._p.domain(host)}]" if host else "[URL]"
        return _FIXED_TAGS[kind]


def _plausible_ner_name(value: str) -> bool:
    tokens = _NAME_TOKEN.findall(value)
    return bool(tokens) and any(t[0].isupper() for t in tokens) \
        and tokens[0].lower() not in _GREETING_WORDS and not _ORG_HINT.search(value)


def _format_mailbox(mb: Mailbox, p: Pseudonymizer, text_anon: TextAnonymizer) -> str:
    addr = p.email(mb.addr)
    if not mb.name:
        return addr
    name = p.person(mb.name) if looks_like_person(mb.name, mb.addr) else text_anon.anonymize(mb.name)
    return f"{name} <{addr}>"


def anonymize_corpus(emails: Sequence[ParsedEmail], ner: NerFn | None,
                     own_addr: str | None = None) -> tuple[list[dict[str, object]], Stats]:
    """Pseudonymisiert alle E-Mails. Zwei Durchgänge: erst bekannte Personen aus den
    Kopfzeilen des ganzen Korpus sammeln, dann alle Texte ersetzen."""
    inferred_addr, own_name = infer_identity(emails)
    own_addr = (own_addr or inferred_addr or "").lower() or None
    if own_addr != inferred_addr:
        own_name = None
    stats = Stats(emails_total=len(emails))
    p = Pseudonymizer(own_addr, own_name)
    for mail in emails:
        for mb in (mail.sender, *mail.to, *mail.cc):
            if mb is not None and mb.addr != own_addr and looks_like_person(mb.name, mb.addr):
                p.register_person(mb.name)
    stats.known_name_tokens = len(p.person_tokens)
    text_anon = TextAnonymizer(p, stats)

    texts = [t for mail in emails for t in (mail.subject, mail.body)]
    ner_results = ner(texts) if ner is not None else [[] for _ in texts]

    records: list[dict[str, object]] = []
    for i, mail in enumerate(emails):
        subj_spans, body_spans = ner_results[2 * i], ner_results[2 * i + 1]
        if not mail.body:
            stats.emails_without_text += 1
        records.append({
            "id": mail.id,
            "datum": mail.date.isoformat() if mail.date else None,
            "absender": _format_mailbox(mail.sender, p, text_anon) if mail.sender else None,
            "an_mich_direkt": own_addr is not None and any(mb.addr == own_addr for mb in mail.to),
            "empfaenger_anzahl": len(mail.to) + len(mail.cc),
            "betreff": text_anon.anonymize(mail.subject, subj_spans),
            "text": text_anon.anonymize(mail.body, body_spans),
            "anhaenge": mail.attachments,
            "verteiler": mail.is_list,
        })
    return records, stats


def residual_findings(records: Iterable[Mapping[str, object]]) -> Counter[str]:
    """Zählt Muster, die nach der Anonymisierung noch vorkommen (Selbstkontrolle).
    Funktionsadressen von Unternehmen werden bewusst behalten und nicht gezählt."""
    found: Counter[str] = Counter({"CARD": 0, "EMAIL": 0, "IBAN": 0, "PHONE": 0})
    for rec in records:
        text = " ".join(str(rec.get(k) or "") for k in ("absender", "betreff", "text"))
        for m in EMAIL_RE.finditer(text):
            local, _, domain = m.group().lower().partition("@")
            if domain in FREEMAIL_DOMAINS or not _ROLE_LOCALPART.match(local):
                found["EMAIL"] += 1
        found["IBAN"] += sum(1 for m in IBAN_RE.finditer(text) if _is_iban(m.group()))
        found["PHONE"] += sum(1 for m in PHONE_RE.finditer(text)
                              if sum(ch.isdigit() for ch in m.group()) >= 7)
        found["CARD"] += sum(1 for m in CARD_RE.finditer(text)
                             if _luhn_ok(re.sub(r"\D", "", m.group()))
                             and 13 <= len(re.sub(r"\D", "", m.group())) <= 19)
    return found
