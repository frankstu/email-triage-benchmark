from __future__ import annotations

from datetime import datetime, timedelta, timezone

from conftest import MakeEml
from jev_bench.parse import MAX_BODY_CHARS, Mailbox, clean_body, parse_file, truncate


def test_header_fields(make_eml: MakeEml) -> None:
    mail = parse_file(make_eml(cc="Peter <peter@web.de>, info@verein.de", subject="Treffen  morgen"))
    assert mail.id == "0001"
    assert mail.sender == Mailbox("Maria Schulz", "maria.schulz@gmail.com")
    assert mail.to == (Mailbox("Frank Beispiel", "frank.beispiel@gmail.com"),)
    assert mail.cc == (Mailbox("Peter", "peter@web.de"), Mailbox("", "info@verein.de"))
    assert mail.subject == "Treffen morgen"
    assert mail.date == datetime(2026, 9, 30, 10, 15, tzinfo=timezone(timedelta(hours=2)))
    assert not mail.is_list
    assert mail.attachments == 0


def test_list_and_attachment_flags(make_eml: MakeEml) -> None:
    mail = parse_file(make_eml(headers={"List-Unsubscribe": "<mailto:u@x.de>"}, attachment=b"%PDF"))
    assert mail.is_list
    assert mail.attachments == 1


def test_html_only_mail_becomes_text_without_quotes(make_eml: MakeEml) -> None:
    html = ("<html><head><style>p{color:red}</style></head><body><p>Neue&nbsp;Nachricht</p>"
            "<div class='gmail_quote'>Alter Verlauf</div><blockquote>Zitat</blockquote>"
            "<script>alert(1)</script></body></html>")
    body = parse_file(make_eml(plain=None, html=html)).body
    assert "Neue Nachricht" in body
    assert "Alter Verlauf" not in body and "Zitat" not in body
    assert "alert" not in body and "color" not in body


def test_plain_preferred_over_html(make_eml: MakeEml) -> None:
    body = parse_file(make_eml(plain="Klartext", html="<p>HTML-Fassung</p>")).body
    assert body == "Klartext"


def test_missing_body_and_broken_date(make_eml: MakeEml) -> None:
    raw = (b"From: a@b.de\r\nTo: c@d.de\r\nSubject: =?utf-8?q?Gr=C3=BC=C3=9Fe?=\r\n"
           b"Date: kein Datum\r\nContent-Type: application/pdf\r\n\r\n%PDF")
    mail = parse_file(make_eml(raw=raw))
    assert mail.body == ""
    assert mail.date is None
    assert mail.subject == "Grüße"


def test_unknown_charset_is_decoded_tolerantly(make_eml: MakeEml) -> None:
    raw = (b"From: a@b.de\r\nTo: c@d.de\r\nSubject: x\r\n"
           b"Content-Type: text/plain; charset=x-unbekannt\r\n\r\nHallo Welt")
    assert parse_file(make_eml(raw=raw)).body == "Hallo Welt"


def test_latin1_body(make_eml: MakeEml) -> None:
    raw = (b"From: a@b.de\r\nTo: c@d.de\r\nSubject: x\r\n"
           b"Content-Type: text/plain; charset=iso-8859-1\r\n\r\nGr\xfc\xdfe aus K\xf6ln")
    assert parse_file(make_eml(raw=raw)).body == "Grüße aus Köln"


def test_cuts_german_attribution() -> None:
    text = "Passt, danke!\n\nAm 29.09.2026 um 18:03 schrieb Maria Schulz <m@x.de>:\n> Geht Freitag?"
    assert clean_body(text) == "Passt, danke!"


def test_cuts_wrapped_english_attribution() -> None:
    text = "Sounds good.\n\nOn Mon, Sep 29, 2026 at 6:03 PM Maria Schulz\n<m@x.de> wrote:\n> Friday?"
    assert clean_body(text) == "Sounds good."


def test_cuts_outlook_header_block() -> None:
    text = "Anbei.\n\nVon: Maria Schulz <m@x.de>\nGesendet: Montag, 29. September 2026\nAn: Frank\n\nAlt"
    assert clean_body(text) == "Anbei."


def test_cuts_original_message_marker_and_signature() -> None:
    assert clean_body("Ok\n-----Ursprüngliche Nachricht-----\nalt") == "Ok"
    assert clean_body("Ok\n-- \nMaria Schulz\nTel. 0171 1234567") == "Ok"


def test_keeps_forwarded_content() -> None:
    text = ("Schau mal:\n\n---------- Forwarded message ---------\nVon: Bank <info@bank.de>\n"
            "Date: Mo., 29. Sept. 2026\nSubject: Frist\nTo: <frank@x.de>\n\n"
            "Bitte bis Freitag bestätigen.")
    cleaned = clean_body(text)
    assert "Bitte bis Freitag bestätigen." in cleaned
    assert cleaned.startswith("Schau mal:")


def test_removes_inline_quote_lines_and_invisible_chars() -> None:
    assert clean_body("Ja\u200b\n> zitiert\nNein\n\n\n\nEnde") == "Ja\nNein\n\nEnde"


def test_truncate() -> None:
    assert truncate("kurz") == "kurz"
    long_text = "wort " * 1000
    cut = truncate(long_text)
    assert len(cut) <= MAX_BODY_CHARS + 2
    assert cut.endswith(" …")
    assert truncate("x" * 3000).startswith("x" * MAX_BODY_CHARS)  # ohne Leerzeichen hart kürzen
