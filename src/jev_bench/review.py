"""Stichprobe zur Sichtprüfung: Original und pseudonymisierte Fassung nebeneinander.

    .venv/bin/python -m jev_bench.review [--n 25] [--seed 1]

Die HTML-Datei enthält Originaltexte und wird deshalb unter data/raw/ abgelegt
(per .gitignore ausgeschlossen). Nur lokal im Browser öffnen.
"""

from __future__ import annotations

import argparse
import html
import random
import re
import sys
from pathlib import Path

from jev_bench.parse import parse_file
from jev_bench.prepare import read_records

_PLACEHOLDER = re.compile(r"\[(?:PERSON|EMAIL|ICH|TELEFON|IBAN|KARTE|ADRESSE|PLZ_ORT|NUMMER"
                          r"|URL|DOMAIN)[^\]\s]*\]")

_PAGE = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>Anonymisierung prüfen</title>
<style>
body {{ font: 14px/1.45 -apple-system, system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
h1 {{ font-size: 20px; }}
.mail {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin: 0 0 24px; }}
.mail h2 {{ font-size: 14px; margin: 0; padding: 8px 12px; border-bottom: 1px solid #ddd; background: #f0f0f0; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; }}
.cols > div {{ padding: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }}
.cols > div + div {{ border-left: 1px solid #ddd; }}
.label {{ font-weight: 600; color: #666; display: block; margin-bottom: 6px; white-space: normal; }}
mark {{ background: #ffe28a; border-radius: 3px; padding: 0 2px; }}
.hint {{ color: #555; max-width: 70ch; }}
</style></head><body>
<h1>Anonymisierung prüfen – {count} zufällige E-Mails</h1>
<p class="hint">Links das bereinigte Original, rechts das, was an die Modelle geht.
Bitte rechts nach übrig gebliebenen Namen, Adressen, Telefonnummern o. Ä. suchen.
Ersetzte Stellen sind gelb markiert.</p>
{items}
</body></html>
"""


def _mark(text: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _PLACEHOLDER.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        out.append(f"<mark>{html.escape(m.group())}</mark>")
        pos = m.end()
    out.append(html.escape(text[pos:]))
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stichprobe zur Sichtprüfung erzeugen")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--anon", type=Path, default=Path("data/anonymized/emails.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/review.html"))
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    records = read_records(args.anon)
    sample = random.Random(args.seed).sample(records, min(args.n, len(records)))
    raw_by_id = {p.stem.split("_", 1)[0]: p for p in args.raw.glob("*.eml")}

    items: list[str] = []
    for rec in sorted(sample, key=lambda r: str(r["id"])):
        rec_id = str(rec["id"])
        original = parse_file(raw_by_id[rec_id])
        sender = f"{original.sender.name} <{original.sender.addr}>" if original.sender else ""
        left = (f"<span class='label'>Original</span>Von: {html.escape(sender)}\n"
                f"Betreff: {html.escape(original.subject)}\n\n{html.escape(original.body)}")
        right = (f"<span class='label'>Pseudonymisiert</span>Von: {_mark(str(rec['absender'] or ''))}\n"
                 f"Betreff: {_mark(str(rec['betreff']))}\n\n{_mark(str(rec['text']))}")
        items.append(f"<section class='mail'><h2>E-Mail {html.escape(rec_id)}</h2>"
                     f"<div class='cols'><div>{left}</div><div>{right}</div></div></section>")

    args.out.write_text(_PAGE.format(count=len(items), items="\n".join(items)), encoding="utf-8")
    print(f"{len(items)} E-Mails nach {args.out} geschrieben")
    return 0


if __name__ == "__main__":
    sys.exit(main())
