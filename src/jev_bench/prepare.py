"""data/raw/*.eml -> data/anonymized/emails.jsonl (lokal, ohne Netzwerk).

    .venv/bin/python -m jev_bench.prepare [--raw data/raw] [--out data/anonymized/emails.jsonl]

Gibt nur Zählwerte aus, keine Inhalte – die Ausgabe darf also z. B. in einen
Chat kopiert werden.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from email.errors import MessageError
from pathlib import Path

from jev_bench.anonymize import NerFn, anonymize_corpus, residual_findings
from jev_bench.parse import ParsedEmail, parse_file


def read_records(path: Path) -> list[dict[str, object]]:
    # Nur an "\n" trennen: splitlines() trennt auch an U+2028 u. Ä., die json.dumps
    # mit ensure_ascii=False unmaskiert in Strings stehen lässt
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E-Mails parsen und lokal pseudonymisieren")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/anonymized/emails.jsonl"))
    parser.add_argument("--me", help="eigene Adresse (Standard: häufigster Empfänger)")
    parser.add_argument("--no-ner", action="store_true", help="ohne spaCy (nur für Tests)")
    args = parser.parse_args(argv)

    files = sorted(args.raw.glob("*.eml"))
    if not files:
        print(f"Keine .eml-Dateien in {args.raw}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    emails: list[ParsedEmail] = []
    failed = 0
    for path in files:
        try:
            emails.append(parse_file(path))
        except (MessageError, ValueError, UnicodeError):
            failed += 1
    parsed_at = time.perf_counter()

    ner: NerFn | None = None
    if not args.no_ner:
        from jev_bench.ner import SpacyNer  # pylint: disable=import-outside-toplevel
        ner = SpacyNer()
    records, stats = anonymize_corpus(emails, ner, args.me)
    done_at = time.perf_counter()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    residual = residual_findings(records)
    summary = {
        "dateien": len(files),
        "nicht_lesbar": failed,
        "ausgegeben": len(records),
        "ohne_text": stats.emails_without_text,
        "bekannte_namensbestandteile": stats.known_name_tokens,
        "ersetzt": dict(sorted(stats.replaced.items())),
        "rest_muster": dict(sorted(residual.items())),
        "sekunden_parsen": round(parsed_at - started, 1),
        "sekunden_anonymisieren": round(done_at - parsed_at, 1),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
