"""Die vier Entscheidungen pro E-Mail – eine Definition für alle Modelle."""

from __future__ import annotations

import json
from collections.abc import Mapping

DRINGLICHKEIT = {
    "sofort": "Heute reagieren: Frist heute/morgen, direkte Rückfrage, die jemanden blockiert, "
              "Termin in Kürze, Sicherheitswarnung zum eigenen Konto.",
    "bald": "In den nächsten Tagen erledigen: Anfrage ohne akute Frist, Rechnung mit Zahlungsziel, "
            "Terminabstimmung für später.",
    "später": "Keine Handlung oder irgendwann: Newsletter, Werbung, Info, automatische "
              "Benachrichtigung ohne Handlungsbedarf.",
}
KATEGORIE = {
    "persönlich": "Persönliche Nachricht einer Person an mich.",
    "termin": "Termin, Einladung, Buchung, Reservierung.",
    "rechnung_finanzen": "Rechnung, Zahlung, Bank, Versicherung, Steuer, Bestellung mit Kosten.",
    "benachrichtigung": "Automatische Nachricht eines Dienstes (Versand, Login, Status, Konto).",
    "newsletter_werbung": "Newsletter, Werbung, Angebote, Marketing.",
    "sonstiges": "Passt in keine andere Kategorie.",
}
ANTWORT_NOETIG = "Erwartet die E-Mail eine Antwort von mir (nicht nur einen Klick oder Kenntnisnahme)?"
PHISHING_VERDACHT = ("Gibt es Anzeichen für Phishing oder Betrug (gefälschter Absender, Druck, "
                     "Aufforderung zu Login/Zahlung über verdächtige Links)?")

FIELDS = ("dringlichkeit", "kategorie", "antwort_noetig", "phishing_verdacht")

JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "dringlichkeit": {"type": "string", "enum": list(DRINGLICHKEIT)},
        "kategorie": {"type": "string", "enum": list(KATEGORIE)},
        "antwort_noetig": {"type": "boolean"},
        "phishing_verdacht": {"type": "boolean"},
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}


def _bullets(options: Mapping[str, str]) -> str:
    return "\n".join(f"- {key}: {desc}" for key, desc in options.items())


INSTRUCTIONS = f"""Du klassifizierst eine E-Mail aus meinem privaten Postfach.
Personennamen, Adressen und Nummern sind pseudonymisiert ([PERSON_3], [TELEFON] usw.);
[ICH] bin ich, der Empfänger. Bei Links ist nur die Domain erhalten.

dringlichkeit – wie schnell muss ich reagieren?
{_bullets(DRINGLICHKEIT)}

kategorie:
{_bullets(KATEGORIE)}

antwort_noetig: {ANTWORT_NOETIG}
phishing_verdacht: {PHISHING_VERDACHT}

Antworte ausschließlich im vorgegebenen JSON-Format."""


def render_email(record: Mapping[str, object]) -> str:
    """Einheitliche Darstellung einer pseudonymisierten E-Mail für alle Modelle."""
    lines = [
        f"Datum: {record.get('datum') or 'unbekannt'}",
        f"Von: {record.get('absender') or 'unbekannt'}",
        f"An mich direkt: {'ja' if record.get('an_mich_direkt') else 'nein (Kopie/Verteiler)'}",
        f"Empfänger insgesamt: {record.get('empfaenger_anzahl')}",
        f"Verteiler/Newsletter-Kopfzeile: {'ja' if record.get('verteiler') else 'nein'}",
        f"Anhänge: {record.get('anhaenge')}",
        f"Betreff: {record.get('betreff') or ''}",
        "",
        str(record.get("text") or "(kein Text)"),
    ]
    return "\n".join(lines)


def validate(answer: Mapping[str, object]) -> list[str]:
    """Leere Liste = Antwort passt zum Schema."""
    errors: list[str] = []
    if set(answer) != set(FIELDS):
        errors.append(f"Felder {sorted(answer)} statt {sorted(FIELDS)}")
    if answer.get("dringlichkeit") not in DRINGLICHKEIT:
        errors.append(f"dringlichkeit={answer.get('dringlichkeit')!r}")
    if answer.get("kategorie") not in KATEGORIE:
        errors.append(f"kategorie={answer.get('kategorie')!r}")
    for key in ("antwort_noetig", "phishing_verdacht"):
        if not isinstance(answer.get(key), bool):
            errors.append(f"{key}={answer.get(key)!r}")
    return errors


def dumps(answer: Mapping[str, object]) -> str:
    return json.dumps(answer, ensure_ascii=False, sort_keys=True)
