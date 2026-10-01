"""Synthetische Geschäfts-E-Mails mit gesteuerter Klassenverteilung.

    .venv/bin/python -m jev_bench.synth [--n 200] [--max-usd 3]

Jede E-Mail entsteht aus einem Szenario (gewünschte Dringlichkeit, Kategorie, Antwort,
Phishing und eine Schwierigkeit) plus zufälligen Rahmendaten (Branche, Absenderrolle,
Länge, Ton, Sprache). Die drei Generatoren wechseln sich ab, damit kein Kandidat von
"seinem" Schreibstil profitiert; das Feld "generator" hält fest, wer geschrieben hat.

Das Szenario ist nur die Absicht. Die Ground Truth bestimmen danach die drei Prüfer
(jev_bench.label) – genau wie bei den echten E-Mails.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jev_bench.config import load_env, require
from jev_bench.label import Budget
from jev_bench.llm import API_ERRORS, MODELS, ClassifyError, RawCall, Task, make_raw_call
from jev_bench.prepare import read_records

GENERATORS = ("gpt-6-sol", "opus-5.5", "gemini-3.8-flash")
# Opus 5.5 verweigert das Schreiben von Phishing-Beispielen; die übernehmen die anderen beiden
PHISHING_GENERATORS = ("gpt-6-sol", "gemini-3.8-flash")


@dataclass(frozen=True)
class Scenario:
    key: str
    beschreibung: str
    dringlichkeit: str
    kategorie: str
    antwort_noetig: bool
    phishing: bool
    gewicht: int  # relative Häufigkeit


# Bewusst mit schwierigen Fällen: versteckte Fristen, falsche Dringlichkeit, Kopie statt direkt
SCENARIOS = (
    # --- sofort -------------------------------------------------------------------
    Scenario("frist_heute", "Kolleg:in braucht bis heute Nachmittag eine Freigabe oder Zahl, "
             "sonst verzögert sich ein Projekt.", "sofort", "persönlich", True, False, 6),
    Scenario("kunde_eskalation", "Verärgerter Kunde eskaliert ein Problem und erwartet heute eine "
             "Rückmeldung.", "sofort", "persönlich", True, False, 4),
    Scenario("termin_gleich", "Termin in einer Stunde wurde verlegt oder der Raum hat sich geändert.",
             "sofort", "termin", False, False, 3),
    Scenario("frist_versteckt", "Lange, freundliche Statusmail; erst im vorletzten Absatz steht, "
             "dass bis morgen früh eine Entscheidung nötig ist.", "sofort", "persönlich", True, False, 4),
    Scenario("ausfall", "Produktivsystem ist ausgefallen; Bitte um sofortige Einschätzung, wer "
             "informiert werden muss.", "sofort", "persönlich", True, False, 3),
    Scenario("zahlung_mahnung", "Letzte Mahnung eines echten Lieferanten mit Frist morgen und "
             "Hinweis auf Lieferstopp.", "sofort", "rechnung_finanzen", False, False, 3),
    Scenario("konto_sicherheit", "Echte Sicherheitswarnung des eigenen IT-Teams: ungewöhnlicher "
             "Login, bitte sofort Passwort über das interne Portal ändern.", "sofort",
             "benachrichtigung", False, False, 2),
    # --- bald ---------------------------------------------------------------------
    Scenario("rueckfrage_ohne_frist", "Fachliche Rückfrage einer Kollegin ohne konkrete Frist.",
             "bald", "persönlich", True, False, 6),
    Scenario("terminabstimmung", "Bitte, für ein Meeting nächste Woche Zeitfenster vorzuschlagen.",
             "bald", "termin", True, False, 5),
    Scenario("rechnung_ziel", "Normale Rechnung mit Zahlungsziel in 14 Tagen.", "bald",
             "rechnung_finanzen", False, False, 4),
    Scenario("review_bitte", "Bitte, ein Dokument bis Ende der Woche zu kommentieren.", "bald",
             "persönlich", True, False, 4),
    Scenario("einladung_event", "Einladung zu einer internen Veranstaltung in drei Wochen mit "
             "Bitte um Zu- oder Absage.", "bald", "termin", True, False, 3),
    Scenario("cc_entscheidung", "Ich stehe nur in CC, aber am Ende werde ich direkt nach meiner "
             "Meinung gefragt.", "bald", "persönlich", True, False, 3),
    Scenario("vertrag_verlaengerung", "Hinweis eines Dienstleisters, dass ein Vertrag in vier "
             "Wochen ausläuft und gekündigt oder verlängert werden muss.", "bald",
             "rechnung_finanzen", True, False, 2),
    # --- später -------------------------------------------------------------------
    Scenario("newsletter", "Fach-Newsletter mit mehreren Artikeln und Links.", "später",
             "newsletter_werbung", False, False, 8),
    Scenario("werbung_druck", "Werbung mit künstlichem Zeitdruck (\"Nur heute 50 %!\").", "später",
             "newsletter_werbung", False, False, 5),
    Scenario("info_cc", "Ausführliche Projektinfo, in der ich nur in CC stehe und nichts tun muss.",
             "später", "sonstiges", False, False, 5),
    Scenario("system_benachrichtigung", "Automatische Benachrichtigung (Ticket geschlossen, "
             "Build erfolgreich, Paket zugestellt).", "später", "benachrichtigung", False, False, 6),
    Scenario("rundmail", "Rundmail an alle Mitarbeitenden (Kantine, Parkplätze, Betriebsfeier).",
             "später", "sonstiges", False, False, 4),
    Scenario("danke", "Kurzes Dankeschön nach erledigter Aufgabe, keine Frage.", "später",
             "persönlich", False, False, 3),
    Scenario("terminbestaetigung", "Automatische Bestätigung eines bereits zugesagten Termins.",
             "später", "termin", False, False, 3),
    Scenario("zahlungsbeleg", "Zahlungsbestätigung oder Gutschrift ohne Handlungsbedarf.", "später",
             "rechnung_finanzen", False, False, 3),
    # --- Phishing (Dringlichkeit bewusst offen; entscheiden die Prüfer) ------------
    Scenario("ceo_fraud", "Angeblich der Geschäftsführer: dringende, vertrauliche Überweisung, "
             "bitte nicht telefonisch nachfragen.", "sofort", "rechnung_finanzen", True, True, 3),
    Scenario("passwort_phishing", "Gefälschte IT-Mail: Postfach läuft über, Login über einen Link "
             "auf einer fremden Domain.", "sofort", "benachrichtigung", False, True, 3),
    Scenario("rechnung_phishing", "Gefälschte Rechnung eines unbekannten Absenders mit Anhang und "
             "Drohung mit Inkasso.", "sofort", "rechnung_finanzen", False, True, 2),
    Scenario("paket_phishing", "Gefälschte Paketbenachrichtigung mit Zollgebühr über einen Link.",
             "später", "benachrichtigung", False, True, 2),
)

BRANCHEN = ("Maschinenbau", "Software-Dienstleister", "Stadtverwaltung", "Krankenhaus",
            "Steuerberatung", "Logistik", "Energieversorger", "Hochschule", "Handel", "Verband")
LAENGEN = ("sehr kurz (1–2 Sätze)", "kurz (3–5 Sätze)", "mittel (2–3 Absätze)", "lang (4–6 Absätze)")
TOENE = ("förmlich", "kollegial-locker", "knapp und sachlich", "freundlich-ausführlich")

SYSTEM = """Du schreibst realistische, aber vollständig erfundene geschäftliche E-Mails für einen
Benchmark zur E-Mail-Klassifizierung. Halte dich exakt an das vorgegebene Szenario.

Pseudonymisierung (Pflicht, genau so wie in echten Daten nach der Anonymisierung):
- Personen nur als [PERSON_1], [PERSON_2] … – nie echte oder erfundene Klarnamen.
- Den Empfänger (mich) immer als [ICH] ansprechen.
- Telefonnummern als [TELEFON], Adressen als [ADRESSE], lange Nummern als [NUMMER],
  IBAN als [IBAN].
- Links nur als [URL:domain], z. B. [URL:firma-beispiel.de]; bei Phishing eine passend
  verdächtige Domain.
- Absender als "[PERSON_n] <[EMAIL_n]@domain>" für Personen oder als echte Funktionsadresse
  wie "rechnung@lieferant-beispiel.de" für Systeme und Firmen. Nur erfundene Domains und
  erfundene Firmennamen, keine echten Unternehmen, Marken oder Behörden.
Das Feld "text" enthält nur den Nachrichtentext: keine Kopfzeilen wie Von:, An:, Cc: oder
Betreff:. Keine Erklärungen, keine Hinweise auf den Benchmark oder das Szenario."""

OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "absender": {"type": "string"},
        "betreff": {"type": "string"},
        "text": {"type": "string"},
        "anhaenge": {"type": "integer"},
    },
    "required": ["absender", "betreff", "text", "anhaenge"],
    "additionalProperties": False,
}

GENERATE = Task("email", "Die erzeugte E-Mail", SYSTEM, OUTPUT_SCHEMA, max_tokens=6000)


@dataclass(frozen=True)
class Spec:
    id: str
    scenario: str
    generator: str
    branche: str
    laenge: str
    ton: str
    sprache: str
    an_mich_direkt: bool
    empfaenger_anzahl: int
    verteiler: bool
    datum: str


def build_specs(n: int, seed: int, generators: Sequence[str] = GENERATORS) -> list[Spec]:
    rng = random.Random(seed)
    weights = [s.gewicht for s in SCENARIOS]
    start = datetime(2026, 9, 1, 8, 0, tzinfo=timezone(timedelta(hours=2)))
    specs: list[Spec] = []
    turn = {"normal": 0, "phishing": 0}
    for i in range(n):
        sc = rng.choices(SCENARIOS, weights)[0]
        pool = [g for g in generators if g in PHISHING_GENERATORS] if sc.phishing else list(generators)
        kind = "phishing" if sc.phishing else "normal"
        generator = pool[turn[kind] % len(pool)]
        turn[kind] += 1
        cc_only = sc.key in ("info_cc", "cc_entscheidung")
        mass = sc.key in ("newsletter", "werbung_druck", "rundmail")
        specs.append(Spec(
            id=f"S{i + 1:03d}",
            scenario=sc.key,
            generator=generator,
            branche=rng.choice(BRANCHEN),
            laenge="lang (4–6 Absätze)" if sc.key == "frist_versteckt" else rng.choice(LAENGEN),
            ton=rng.choice(TOENE),
            sprache="englisch" if rng.random() < 0.15 else "deutsch",
            an_mich_direkt=not (cc_only or mass),
            empfaenger_anzahl=1 if not (cc_only or mass) else rng.randint(3, 40) if cc_only else 500,
            verteiler=mass,
            datum=(start + timedelta(days=rng.randint(0, 29), minutes=rng.randint(0, 600))).isoformat(),
        ))
    return specs


_BY_KEY = {s.key: s for s in SCENARIOS}


def prompt_for(spec: Spec) -> str:
    sc = _BY_KEY[spec.scenario]
    empfang = ("direkt an mich" if spec.an_mich_direkt
               else "an einen Verteiler" if spec.verteiler else "an andere, ich stehe nur in CC")
    return (f"Szenario: {sc.beschreibung}\n"
            f"Organisation des Empfängers: {spec.branche}\n"
            f"Gesendet {empfang}; Datum {spec.datum}\n"
            f"Länge: {spec.laenge}; Ton: {spec.ton}; Sprache: {spec.sprache}\n"
            "Schreibe genau diese eine E-Mail.")


def to_record(spec: Spec, mail: Mapping[str, object], cost_usd: float = 0.0) -> dict[str, object]:
    sc = _BY_KEY[spec.scenario]
    return {
        "id": spec.id,
        "datum": spec.datum,
        "absender": str(mail["absender"]),
        "an_mich_direkt": spec.an_mich_direkt,
        "empfaenger_anzahl": spec.empfaenger_anzahl,
        "betreff": str(mail["betreff"]),
        "text": str(mail["text"]),
        "anhaenge": int(str(mail["anhaenge"])),
        "verteiler": spec.verteiler,
        "synthetisch": True,
        "generator": spec.generator,
        "szenario": spec.scenario,
        "absicht": {"dringlichkeit": sc.dringlichkeit, "kategorie": sc.kategorie,
                    "antwort_noetig": sc.antwort_noetig, "phishing_verdacht": sc.phishing},
        "rahmen": {k: v for k, v in asdict(spec).items()
                   if k in ("branche", "laenge", "ton", "sprache")},
        "kosten_usd": cost_usd,
    }


def _valid_mail(mail: Mapping[str, object]) -> bool:
    return all(isinstance(mail.get(k), str) and str(mail.get(k)).strip()
               for k in ("absender", "betreff", "text")) \
        and isinstance(mail.get("anhaenge"), int) and not isinstance(mail.get("anhaenge"), bool)


def generate(specs: Sequence[Spec], calls: Mapping[str, RawCall], out: Path, budget: Budget,
             costs: Mapping[str, Callable[[int, int], float]], *, workers: int = 2) -> tuple[int, int]:
    """Erzeugt fehlende E-Mails (fortsetzbar). Rückgabe: (neu geschrieben, fehlgeschlagen)."""
    done = {str(r["id"]) for r in read_records(out)} if out.exists() else set()
    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}

    def work(spec: Spec) -> None:
        if budget.exhausted:
            return
        try:
            mail, tokens_in, tokens_out = calls[spec.generator](prompt_for(spec))
        except (ClassifyError, *API_ERRORS) as e:
            print(f"{spec.id} {spec.generator}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            with lock:
                counts["fail"] += 1
            return
        cost = costs[spec.generator](tokens_in, tokens_out)
        budget.add(cost)
        with lock:
            if not _valid_mail(mail):
                counts["fail"] += 1
                return
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(to_record(spec, mail, cost), ensure_ascii=False) + "\n")
            counts["ok"] += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    pending = [s for s in specs if s.id not in done]
    pools = {g: ThreadPoolExecutor(max_workers=workers) for g in calls}
    futures: list[Future[None]] = []
    try:
        futures += [pools[s.generator].submit(work, s) for s in pending]
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    for fut in futures:
        fut.result()
    return counts["ok"], counts["fail"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetische Geschäfts-E-Mails erzeugen")
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/emails.jsonl"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-usd", type=float, default=3.0)
    args = parser.parse_args(argv)

    load_env()
    key = require("LANGDOCK_API_KEY")
    specs = build_specs(args.n, args.seed)
    calls = {g: make_raw_call(MODELS[g], key, GENERATE) for g in GENERATORS}
    costs = {g: MODELS[g].cost for g in GENERATORS}
    existing = read_records(args.out) if args.out.exists() else []
    previous = sum(float(str(r.get("kosten_usd", 0))) for r in existing)
    budget = Budget(args.max_usd, previous)
    ok, failed = generate(specs, calls, args.out, budget, costs)
    total = len(read_records(args.out)) if args.out.exists() else 0
    print(json.dumps({"neu": ok, "fehlgeschlagen": failed, "gesamt": total, "soll": args.n,
                      "kosten_usd": round(budget.spent, 4), "budget_erreicht": budget.exhausted},
                     ensure_ascii=False, indent=2))
    return 0 if total == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
