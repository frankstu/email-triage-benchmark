# Ergebnisse

## Datensatz: echt (300 E-Mails)

| Modell | alle 4 richtig | Dringlichkeit | Recall „sofort“ | Kategorie | Antwort nötig | Phishing (Recall) | Konstanz | Latenz p50 | p95 | $/1000 | ungültig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-luna | 81.5 % | 97.5 % | 100.0 % | 84.0 % | 100.0 % | – | 99.0 % | 985.4 ms | 1530.9 ms | 0.1039 | 0/600 |
| haiku-4.5 | 88.5 % | 96.5 % | 75.0 % | 92.8 % | 99.7 % | – | 91.0 % | 1116.2 ms | 1403.3 ms | 2.3881 | 0/600 |
| jev | 79.8 % | 95.0 % | 50.0 % | 87.8 % | 97.5 % | – | 98.7 % | 259.1 ms | 361.6 ms | 0.0551 | 0/600 |
| jev-de | 81.0 % | 96.7 % | 50.0 % | 88.5 % | 96.3 % | – | 98.3 % | 261.2 ms | 363.8 ms | 0.059 | 0/600 |

Immer die häufigste Klasse raten: dringlichkeit 96.0 %, kategorie 73.7 %, antwort_noetig 99.0 %, phishing_verdacht 100.0 %

## Datensatz: synthetisch (150 E-Mails)

| Modell | alle 4 richtig | Dringlichkeit | Recall „sofort“ | Kategorie | Antwort nötig | Phishing (Recall) | Konstanz | Latenz p50 | p95 | $/1000 | ungültig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-6-luna | 64.3 % | 85.3 % | 91.5 % | 84.7 % | 92.7 % | 83.3 % | 92.7 % | 1031.1 ms | 1538.2 ms | 0.0916 | 0/300 |
| haiku-4.5 | 48.3 % | 77.7 % | 96.8 % | 75.7 % | 88.3 % | 79.2 % | 86.7 % | 1091.2 ms | 1413.8 ms | 2.2475 | 0/300 |
| jev | 65.3 % | 86.7 % | 87.2 % | 89.7 % | 88.0 % | 83.3 % | 96.7 % | 256.9 ms | 357.9 ms | 0.0485 | 0/300 |
| jev-de | 61.7 % | 89.0 % | 89.4 % | 88.0 % | 83.0 % | 77.1 % | 94.7 % | 252.7 ms | 350.3 ms | 0.0524 | 0/300 |

Immer die häufigste Klasse raten: dringlichkeit 36.7 %, kategorie 30.0 %, antwort_noetig 59.3 %, phishing_verdacht 84.0 %

