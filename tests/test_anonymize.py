from __future__ import annotations

from collections.abc import Sequence

import pytest

from jev_bench.anonymize import (Pseudonymizer, Stats, TextAnonymizer, anonymize_corpus,
                                 infer_identity, looks_like_person, residual_findings)
from jev_bench.parse import Mailbox, ParsedEmail

ME = Mailbox("Frank Beispiel", "frank.beispiel@gmail.com")


def mail(*, body: str = "", subject: str = "", sender: Mailbox | None = None,  # pylint: disable=too-many-arguments
         to: Sequence[Mailbox] = (ME,), cc: Sequence[Mailbox] = (),
         mid: str = "0001") -> ParsedEmail:
    return ParsedEmail(id=mid, date=None, sender=sender, to=tuple(to), cc=tuple(cc),
                       subject=subject, body=body, attachments=0, is_list=False)


def anon(text: str, known: Sequence[str] = (), own: Mailbox = ME) -> str:
    p = Pseudonymizer(own.addr, own.name)
    for name in known:
        p.register_person(name)
    return TextAnonymizer(p, Stats()).anonymize(text)


# --- strukturierte Daten -----------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Ruf an: 0171 1234567", "Tel. +49 30 12345678", "Tel: 030/1234567", "(0221) 123 45 67",
    "0049 89 1234 5678", "Mobil +49 (0)171 2345678",
])
def test_phone_numbers(text: str) -> None:
    assert "[TELEFON]" in anon(text)


@pytest.mark.parametrize("text", [
    "Termin am 01.10.2026 um 09:30", "Betrag: 1.234,56 €", "Zeitraum 2025-2026",
    "Kosten 0,99 Euro", "PLZ-freier Text mit 12 Teilnehmern", "Version 3.14.7",
])
def test_no_false_positive_phone_or_number(text: str) -> None:
    assert anon(text) == text


def test_iban_card_long_number() -> None:
    out = anon("IBAN DE89 3704 0044 0532 0130 00, Karte 4111 1111 1111 1111, Bestellung 30212345")
    assert "[IBAN]" in out and "[KARTE]" in out and "[NUMMER]" in out
    assert "3704" not in out and "4111" not in out and "30212345" not in out


def test_card_requires_luhn() -> None:
    assert "[KARTE]" not in anon("Nummer 1234 5678 9012 3456")


def test_capitalized_words_are_no_iban() -> None:
    assert anon("DE12 ABCD EFGH IJKL ist kein Konto") == "DE12 ABCD EFGH IJKL ist kein Konto"


def test_address_and_postcode() -> None:
    out = anon("Lieferung an Musterstraße 12a, 10115 Berlin.")
    assert out == "Lieferung an [ADRESSE], [PLZ_ORT]."


def test_postcode_not_amount() -> None:
    assert anon("Spende 10000 Euro") == "Spende 10000 Euro"


def test_urls_keep_only_host() -> None:
    out = anon("Login: https://www.paypal.com/signin?token=abc123. Oder www.bahn.de/info")
    assert out == "Login: [URL:paypal.com]. Oder [URL:bahn.de]"


def test_url_host_with_person_name_is_masked() -> None:
    assert anon("Siehe https://maria-schulz.de/cv", known=["Maria Schulz"]) == "Siehe [URL:[DOMAIN]]"


# --- E-Mail-Adressen ---------------------------------------------------------------

def test_personal_address_is_pseudonymized_consistently() -> None:
    out = anon("Von peter.meier@gmx.de, nochmal peter.meier@gmx.de und anna@web.de")
    assert out == "Von [EMAIL_1]@gmx.de, nochmal [EMAIL_1]@gmx.de und [EMAIL_2]@web.de"


def test_role_address_of_company_is_kept() -> None:
    assert anon("Antwort an service@paypal.de") == "Antwort an service@paypal.de"


def test_role_address_at_freemail_is_masked() -> None:
    assert anon("info@gmail.com") == "[EMAIL_1]@gmail.com"


def test_business_address_domain_with_name_is_masked() -> None:
    assert anon("m.schulz@kanzlei-schulz.de", known=["Maria Schulz"]) == "[EMAIL_1]@[DOMAIN]"


def test_own_address_and_name() -> None:
    assert anon("Kopie an frank.beispiel@gmail.com, Frank") == "Kopie an [ICH], [ICH]"


# --- Personen ----------------------------------------------------------------------

def test_known_names_from_headers_and_parts() -> None:
    out = anon("Maria Schulz hat zugesagt. Maria bringt Kuchen, Schulz die Getränke.",
               known=["Maria Schulz"])
    assert out == "[PERSON_1] hat zugesagt. [PERSON_1] bringt Kuchen, [PERSON_1] die Getränke."


def test_lowercase_occurrence_of_name_is_not_replaced() -> None:
    # "rose" klein = Pflanze, kein Name
    assert anon("Die rose blüht", known=["Rose Weber"]) == "Die rose blüht"


def test_different_people_with_same_first_name_differ() -> None:
    p = Pseudonymizer(None, None)
    assert p.person("Maria Schulz") == "[PERSON_1]"
    assert p.person("Maria Müller") == "[PERSON_2]"
    assert p.person("Maria") == "[PERSON_1]"  # Einzelname: erster bekannter Treffer
    assert p.person("Schulz, Maria") == "[PERSON_1]"
    assert p.person("Dr. Maria Schulz") == "[PERSON_1]"


def test_greetings_and_closings() -> None:
    text = "Hallo Jonas,\nwie geht's?\n\nViele Grüße\nLena Hoffmann"
    assert anon(text) == "Hallo [PERSON_1],\nwie geht's?\n\nViele Grüße\n[PERSON_2]"


@pytest.mark.parametrize("text", [
    "Hallo zusammen,", "Liebe Kolleginnen und Kollegen,", "Sehr geehrte Damen und Herren,",
    "Viele Grüße\nIhr Amazon-Team", "Hallo Team,",
])
def test_greeting_without_name_untouched(text: str) -> None:
    assert anon(text) == text


def test_titles_in_greeting() -> None:
    assert anon("Sehr geehrte Frau Dr. Wagner,") == "Sehr geehrte Frau [PERSON_1],"
    assert anon("Guten Tag Herr Becker") == "Guten Tag [PERSON_1]"


def test_ner_spans_are_used_and_filtered() -> None:
    p = Pseudonymizer(None, None)
    text = "Treffen mit Johanna und der Sparkasse GmbH"
    out = TextAnonymizer(p, Stats()).anonymize(text, [(12, 19), (28, 42)])
    assert out == "Treffen mit [PERSON_1] und der Sparkasse GmbH"


def test_overlap_email_wins_over_name() -> None:
    assert anon("maria.schulz@web.de", known=["Maria Schulz"]) == "[EMAIL_1]@web.de"


# --- Personenerkennung in Kopfzeilen -----------------------------------------------

@pytest.mark.parametrize(("name", "addr", "expected"), [
    ("Maria Schulz", "maria.schulz@gmail.com", True),
    ("Maria Schulz", "ms@firma.de", True),
    ("Schulz, Maria", "ms@firma.de", True),
    ("Jonas", "jonas@web.de", True),
    ("Jonas", "jk@firma.de", False),
    ("Amazon.de", "versand@amazon.de", False),
    ("Deutsche Bahn", "noreply@bahn.de", False),
    ("PayPal Service", "service@paypal.de", False),
    ("Sparkasse KölnBonn", "info@sparkasse.de", False),
    ("Musterfirma GmbH", "m.mueller@musterfirma.de", False),
    ("Team 42", "team@x.de", False),
    ("", "a@b.de", False),
])
def test_looks_like_person(name: str, addr: str, expected: bool) -> None:
    assert looks_like_person(name, addr) is expected


def test_infer_identity() -> None:
    other = Mailbox("Bob", "bob@web.de")
    emails = [mail(to=[ME]), mail(to=[ME, other]), mail(to=[other], cc=[ME])]
    assert infer_identity(emails) == (ME.addr, ME.name)
    assert infer_identity([]) == (None, None)


# --- Korpus ------------------------------------------------------------------------

def test_corpus_end_to_end() -> None:
    lena = Mailbox("Lena Hoffmann", "lena.h@gmail.com")
    shop = Mailbox("Zalando", "service@zalando.de")
    emails = [
        mail(mid="0001", sender=lena, subject="Frage von Lena",
             body="Hi Frank,\nkannst du mich unter 0171 2345678 anrufen?\nLena"),
        mail(mid="0002", sender=shop, subject="Deine Bestellung 12345678",
             body="Hallo Frank, deine Lieferung kommt. Fragen an service@zalando.de"),
        mail(mid="0003", sender=Mailbox("Bob", "bob@web.de"), to=[lena], cc=[ME],
             body="Lena, schau mal: https://x.example/a?b=1"),
        mail(mid="0004", sender=Mailbox("Bob", "bob@web.de"), to=[ME]),
    ]
    records, stats = anonymize_corpus(emails, ner=None)
    assert len(records) == 4
    r1, r2, r3, r4 = records[0], records[1], records[2], records[3]
    assert r1["absender"] == "[PERSON_1] <[EMAIL_1]@gmail.com>"
    assert r1["betreff"] == "Frage von [PERSON_1]"
    assert r1["text"] == "Hi [ICH],\nkannst du mich unter [TELEFON] anrufen?\n[PERSON_1]"
    assert r1["an_mich_direkt"] is True
    assert r2["absender"] == "Zalando <service@zalando.de>"
    assert r2["betreff"] == "Deine Bestellung [NUMMER]"
    assert r2["text"] == "Hallo [ICH], deine Lieferung kommt. Fragen an service@zalando.de"
    assert r3["text"] == "[PERSON_1], schau mal: [URL:x.example]"
    assert r3["an_mich_direkt"] is False and r3["empfaenger_anzahl"] == 2
    assert r4["text"] == "" and stats.emails_without_text == 1
    assert stats.replaced["PERSON"] >= 4 and stats.replaced["PHONE"] == 1
    assert sum(residual_findings(records).values()) == 0


def test_corpus_with_explicit_own_address() -> None:
    # Abweichende eigene Adresse: der vermutete Name darf nicht als [ICH] gelten
    records, _ = anonymize_corpus([mail(body="Hallo Frank")], ner=None, own_addr="anders@web.de")
    assert records[0]["text"] == "Hallo [PERSON_1]"
    assert records[0]["an_mich_direkt"] is False


def test_residual_findings_detects_leaks() -> None:
    leaked = [{"absender": "x", "betreff": "", "text": "a@gmx.de 0171 2345678 info@firma.de"}]
    found = residual_findings(leaked)
    assert found["EMAIL"] == 1 and found["PHONE"] == 1
