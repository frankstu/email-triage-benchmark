# JEV Mail Benchmark

> **English summary.** An independent benchmark of TypeSafe's decision model JEV 1.13 against
> GPT-6 Luna and Claude Haiku 4.5: four decisions per email (urgency, category, reply needed,
> phishing) in one call, on 150 synthetic business emails (mostly German) and 300 pseudonymized
> private emails, two runs each. On the business emails JEV reached similar accuracy to Luna
> (65% vs 64% all four decisions correct, McNemar p = 1.0), with a 4× lower median latency
> (257 ms vs 1,031 ms), about 47% lower cost and the most consistent answers (97%). Reference
> labels are a majority vote of GPT-6 Sol, Claude Opus 5.5 and Gemini 3.8 Flash, which also wrote
> the synthetic emails. Write-up with all limits:
> [frankstuch.de/en/articles/jev-hands-on](https://frankstuch.de/en/articles/jev-hands-on/).
> Results: [`results/summary.md`](results/summary.md), [`results/metrics.json`](results/metrics.json);
> raw synthetic data and every model answer: [`data/synthetic/`](data/synthetic/).
> The rest of this README is in German.

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

4. **Synthetische Geschäfts-E-Mails** (150, breit gestreute Klassen inkl. Phishing), reihum
   geschrieben von GPT-6 Sol, Claude Opus 5.5 und Gemini 3.8 Flash → `data/synthetic/emails.jsonl`:

       .venv/bin/python -m jev_bench.synth --n 150

5. **Ground Truth** per Mehrheit dreier Prüfer-Modelle (GPT-6 Sol, Claude Opus 5.5,
   Gemini 3.8 Flash) über Langdock, mit Kostenbremse; offene Fälle per Hand
   (`data/synthetic/manual_decisions.json`):

       .venv/bin/python -m jev_bench.label --sample 300 --max-usd 6
       .venv/bin/python -m jev_bench.label --anon data/synthetic/emails.jsonl \
           --out-dir data/labels_synth --sample 150 --decisions data/synthetic/manual_decisions.json

6. **Benchmark** der Kandidaten – JEV 1.13 (Fragen englisch und deutsch), Claude Haiku 4.5,
   GPT-6 Luna – in zwei Läufen, je Kandidat sequenziell für saubere Latenzen:

       .venv/bin/python -m jev_bench.bench --runs 2

7. **Auswertung** → `results/metrics.json`, `results/summary.md` (nur Kennzahlen):

       .venv/bin/python -m jev_bench.evaluate

8. **Diagramme** → `results/charts/` (Überblick, Hybrid-Kurve, Latenzverteilung):

       .venv/bin/pip install -e '.[charts]' && .venv/bin/python scripts/charts.py

Veröffentlicht sind unter `data/synthetic/` die synthetischen E-Mails, die Ground Truth,
die Einzelantworten der Prüfer (`judges.jsonl`, mit Latenz und Kosten) und aller Kandidaten
in beiden Läufen (`candidates.jsonl`, bei JEV mit Wahrscheinlichkeiten). Damit lassen sich
Kennzahlen, Tests und Hybrid-Kurve ohne neue API-Aufrufe nachrechnen. Die echten E-Mails und
alles daraus Abgeleitete bleiben lokal. Kosten sind zu Listenpreisen der Anbieter aus den
gemeldeten Tokens berechnet.

## Bekannte Grenzen

- Die Ground Truth stammt von denselben drei Modellen, die die synthetischen E-Mails
  geschrieben haben. Einen Vorteil für die eigene Modellfamilie zeigt die Auswertung nicht
  (`generator_bias` in `results/metrics.json`), unabhängig ist die Referenz trotzdem nicht.
- Die Anweisung an die LLMs (`INSTRUCTIONS` in `schema.py`) spricht von einem privaten
  Postfach, auch bei den Geschäfts-E-Mails. Die Kategorien „persönlich“, „Termin“ und
  „Rechnung“ können sich überschneiden. Beides ändert sich erst mit einem neuen Messlauf.
- Die Hybrid-Kurve ist eine Simulation aus Lauf 1; die Schwelle ist auf denselben Daten gewählt.
- Prüfer-Reasoning: GPT-6 Sol mit `medium`, Claude Opus 5.5 und Gemini 3.8 Flash mit
  Standardeinstellungen; GPT-6 Luna ohne Reasoning.

Erreichbarkeit der Modelle prüfen (eine erfundene Test-E-Mail):

    .venv/bin/python scripts/probe_langdock.py

## Prüfen

    .venv/bin/python -m pytest -q
    .venv/bin/pyright && .venv/bin/pyflakes src tests scripts && .venv/bin/pylint src tests scripts
