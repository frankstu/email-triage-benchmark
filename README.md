# JEV Mail Benchmark

Vergleich von JEV (TypeSafe AI) mit LLMs bei der Klassifizierung von E-Mails:
Qualität, Latenz, Kosten. Pro E-Mail vier Entscheidungen in einem Aufruf –
Dringlichkeit, Kategorie, Antwort nötig, Phishing-Verdacht (siehe `src/jev_bench/schema.py`).

## Datenschutz

Echte E-Mails verlassen den Rechner nur pseudonymisiert. Rohdaten, pseudonymisierte
Daten und Labels liegen unter `data/` und sind per `.gitignore` ausgeschlossen.
Die Anonymisierung läuft lokal (Regex + spaCy), ohne Netzwerkzugriff.

## Einrichtung

    python3 -m venv .venv
    .venv/bin/pip install -e '.[dev]'
    .venv/bin/python -m spacy download de_core_news_md
    .venv/bin/python -m spacy download en_core_web_md
    cp .env.example .env   # Keys eintragen

## Pipeline

1. **Export** aus Apple Mail (neueste N empfangene E-Mails eines Kontos als `.eml`):

       osascript -l JavaScript scripts/export_mail.js                 # Konten auflisten
       osascript -l JavaScript scripts/export_mail.js "Google" 1000 data/raw

2. **Parsen und pseudonymisieren** → `data/anonymized/emails.jsonl`:

       .venv/bin/python -m jev_bench.prepare

3. **Sichtprüfung** einer Stichprobe (Original und Ergebnis nebeneinander, nur lokal):

       .venv/bin/python -m jev_bench.review && open data/raw/review.html

4. **Ground Truth** per Mehrheit dreier Prüfer-Modelle (GPT-6 Sol, Claude Opus 5.5,
   Gemini 3.8 Flash) über Langdock, mit Kostenbremse:

       .venv/bin/python -m jev_bench.label --sample 300 --max-usd 6

5. **Benchmark** der Kandidaten (JEV, Claude Haiku 4.5, GPT-6 Luna) – folgt.

Erreichbarkeit der Modelle prüfen (eine erfundene Test-E-Mail):

    .venv/bin/python scripts/probe_langdock.py

## Prüfen

    .venv/bin/python -m pytest -q
    .venv/bin/pyright && .venv/bin/pyflakes src tests scripts && .venv/bin/pylint src tests scripts
