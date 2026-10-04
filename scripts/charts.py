"""Diagramme für den Artikel -> results/charts/*.png

    .venv/bin/python scripts/charts.py

Liest results/metrics.json (Kennzahlen) und für die Latenzverteilung die lokalen
Messungen der synthetischen E-Mails aus data/bench/results.jsonl. Alle Zahlen und
Verhältnisse in den Diagrammen werden aus den Daten berechnet, nicht eingetippt.

Gestaltung: Hervorhebung statt Kategorienfarben – JEV in Blau, die Vergleichsmodelle
in Grau, Zeilen direkt beschriftet. Statische PNGs für LinkedIn, daher nur hell.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pylint: disable=wrong-import-position
from matplotlib.axes import Axes  # noqa: E402  # pylint: disable=wrong-import-position

OUT = Path("results/charts")
METRICS = Path("results/metrics.json")
BENCH = Path("data/bench/results.jsonl")
REPO = "github.com/frankstu/email-triage-benchmark"

# Referenzpalette (dataviz): Akzent + Grau, Tinte und Linien in Text-Tokens.
# "hell" für LinkedIn (Blau, auf Weiß gut lesbar),
# "dunkel" passend zum Blog (Bernstein, Fläche wie dessen Karten).
THEMES: dict[str, dict[str, str]] = {
    "hell": {"ACCENT": "#2a78d6", "DEEMPH": "#a9a79f", "SURFACE": "#fcfcfb", "INK": "#0b0b0b",
             "INK_2": "#52514e", "MUTED": "#898781", "GRID": "#e1e0d9", "BASELINE": "#c3c2b7", "suffix": ""},
    "dunkel": {"ACCENT": "#f2c037", "DEEMPH": "#6e6e76", "SURFACE": "#121214", "INK": "#f4f4f5",
               "INK_2": "#b4b4bb", "MUTED": "#8a8a93", "GRID": "#26262b", "BASELINE": "#3a3a40",
               "suffix": "-dunkel"},
}


class C:  # pylint: disable=too-few-public-methods
    """Aktives Farbschema; apply_theme() setzt die Werte."""
    ACCENT = DEEMPH = SURFACE = INK = INK_2 = MUTED = GRID = BASELINE = suffix = ""


def apply_theme(name: str) -> None:
    for key, value in THEMES[name].items():
        setattr(C, key, value)
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
        "font.size": 12, "axes.edgecolor": C.BASELINE, "axes.labelcolor": C.INK_2,
        "xtick.color": C.MUTED, "ytick.color": C.INK_2, "figure.facecolor": C.SURFACE,
        "axes.facecolor": C.SURFACE, "savefig.facecolor": C.SURFACE,
    })


MODELS = (("jev", "JEV 1.13"), ("gpt-6-luna", "GPT-6 Luna"), ("haiku-4.5", "Claude Haiku 4.5"))
HYBRID_SHOWN = 0.5  # Schwelle, die im Überblick genannt wird
# OpenAI Ultrafast für GPT-6 Astra, $ pro 1 Mio. Tokens (Input, Output), seit 29.09.2026 – nicht gemessen,
# nur als Schätzung mit Lunas gemessenem Tokenverbrauch
ULTRAFAST_ASTRA_USD = (60.0, 300.0)
DPI = 180



def de(value: float, digits: int = 0) -> str:
    """Deutsche Zahl: 1.234,5"""
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def en(value: float, digits: int = 0) -> str:
    """Englische Zahl: 1,234.5"""
    return f"{value:,.{digits}f}"


def color(model: str) -> str:
    return C.ACCENT if model == "jev" else C.DEEMPH


def _style(ax: Axes, xmax: float) -> None:
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(MODELS) - 0.4)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(C.BASELINE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0, labelsize=10, colors=C.MUTED)
    ax.grid(axis="x", color=C.GRID, linewidth=0.8)
    ax.set_axisbelow(True)


@dataclass(frozen=True)
class Panel:
    title: str
    note: str
    values: Sequence[float]
    labels: Sequence[str]
    xmax: float
    ticks: Sequence[float]
    tick_fmt: str
    whiskers: Sequence[tuple[float, float]] | None = None


def _panel(ax: Axes, panel: Panel) -> None:
    # Achse über den letzten Tick hinaus verlängern, damit Werte rechts der Balken Platz haben
    _style(ax, panel.xmax * 1.22)
    ax.spines["bottom"].set_bounds(0, panel.xmax)
    for row, ((model, _), v) in enumerate(zip(MODELS, panel.values, strict=True)):
        ax.barh(row, v, height=0.42, color=color(model), linewidth=0)
        end = v
        if panel.whiskers:
            lo, hi = panel.whiskers[row]
            ax.plot([lo, hi], [row, row], color=C.INK_2, linewidth=1.2, solid_capstyle="butt")
            for x in (lo, hi):
                ax.plot([x, x], [row - 0.09, row + 0.09], color=C.INK_2, linewidth=1.2)
            end = max(v, hi)
        ax.text(end + panel.xmax * 0.035, row, panel.labels[row], va="center", ha="left", fontsize=12,
                color=C.INK, fontweight="bold" if model == "jev" else "normal")
    ax.set_xticks(list(panel.ticks))
    ax.set_xticklabels([panel.tick_fmt.format(t).replace(".", ",") for t in panel.ticks])
    ax.set_title(panel.title, loc="left", fontsize=13, fontweight="bold", color=C.INK, pad=22)
    ax.text(0, 1.035, panel.note, transform=ax.transAxes, fontsize=10, color=C.MUTED, va="bottom")


def _numbers(metrics: Mapping[str, object]) -> dict[str, object]:
    """Alle Werte für den Überblick, in der Reihenfolge von MODELS."""
    syn = metrics["datensaetze"]["synthetisch"]  # type: ignore[index]
    echt = metrics["datensaetze"]["echt"]  # type: ignore[index]
    mods, sig = syn["modelle"], syn["signifikanz"]["alle_vier_lauf1"]
    p_luna = syn["signifikanz"]["mcnemar"].get("gpt-6-luna vs jev", {}).get("p")
    if p_luna is None:
        raise SystemExit("McNemar-Test JEV vs. Luna fehlt in metrics.json – erst evaluate laufen lassen")
    return {
        "quality": [100 * sig[m]["anteil"] for m, _ in MODELS],
        "ci": [(100 * sig[m]["ki95"][0], 100 * sig[m]["ki95"][1]) for m, _ in MODELS],
        "latency": [mods[m]["latenz_ms"]["p50"] for m, _ in MODELS],
        "cost": [mods[m]["kosten_pro_1000_usd"] for m, _ in MODELS],
        "consistency": [100 * mods[m]["konstanz"]["alle_felder"] for m, _ in MODELS],
        "p_luna": float(p_luna),
        "hybrid": next(h for h in syn["hybrid"]["gpt-6-luna"] if h["schwelle"] == HYBRID_SHOWN),
        "real": {m: v["alle_vier_richtig"] for m, v in echt["modelle"].items()},  # beide Läufe, wie oben
        "later_share": echt["immer_haeufigste_klasse"]["dringlichkeit"],
        "luna_tokens": (mods["gpt-6-luna"]["tokens_in_mittel"], mods["gpt-6-luna"]["tokens_out_mittel"]),
    }


def _findings(n: Mapping[str, object], lang: str = "de") -> list[tuple[str, str]]:
    cost, hybrid, real = n["cost"], n["hybrid"], n["real"]
    assert isinstance(cost, list) and isinstance(hybrid, dict) and isinstance(real, dict)
    p_luna = float(str(n["p_luna"]))
    later = 100 * float(str(n["later_share"]))
    tokens = n["luna_tokens"]
    assert isinstance(tokens, tuple)
    ultrafast = (tokens[0] * ULTRAFAST_ASTRA_USD[0] + tokens[1] * ULTRAFAST_ASTRA_USD[1]) / 1_000_000 * 1000
    acc, esc = 100 * hybrid["genauigkeit"]["alle_vier"], 100 * hybrid["eskaliert"]
    h_cost, h_ms = hybrid["kosten_pro_1000_usd"], hybrid["latenz_mittel_ms"]
    if lang == "en":
        sig = "no significant difference" if p_luna >= 0.05 else "significant difference"
        return [
            (f"Accuracy JEV vs Luna: {sig} (McNemar p = {en(p_luna, 2)}). "
             f"Claude Haiku: significantly worse and {en(cost[2] / cost[0])}× the cost of JEV.", C.INK_2),
            (f"Hybrid – JEV decides, uncertain cases ({en(esc)}%) go to Luna: {en(acc)}% all 4 correct · "
             f"${en(h_cost, 2)} per 1,000 · avg {en(h_ms)} ms*", C.INK),
            (f"Real private inbox (300 emails, {en(later)}% of them \u201clater\u201d): "
             f"Haiku {en(100 * real['haiku-4.5'], 1)}%, Luna {en(100 * real['gpt-6-luna'], 1)}%, "
             f"JEV {en(100 * real['jev'], 1)}% all 4 correct.", C.INK_2),
            (f"For comparison, calculated**: OpenAI Ultrafast with GPT-6 Astra would cost about "
             f"${en(ultrafast)} per 1,000 emails \u2013 "
             f"roughly {en(round(ultrafast / cost[0], -2))}× JEV.", C.INK_2),
        ]
    sig = "kein signifikanter Unterschied" if p_luna >= 0.05 else "signifikanter Unterschied"
    return [
        (f"Qualität JEV vs. Luna: {sig} (McNemar p = {de(p_luna, 2)}). "
         f"Claude Haiku: signifikant schlechter und {de(cost[2] / cost[0])}× so teuer wie JEV.", C.INK_2),
        (f"Hybrid – JEV entscheidet, unsichere Fälle ({de(esc)} %) gehen an Luna: "
         f"{de(acc)} % alle 4 richtig · {de(h_cost, 2)} $ pro 1.000 · Ø {de(h_ms)} ms*", C.INK),
        (f"Echtes privates Postfach (300 E-Mails, {de(later)} % davon „später“): "
         f"Haiku {de(100 * real['haiku-4.5'], 1)} %, Luna {de(100 * real['gpt-6-luna'], 1)} %, "
         f"JEV {de(100 * real['jev'], 1)} % alle 4 richtig.", C.INK_2),
        (f"Zum Vergleich, rechnerisch**: OpenAI Ultrafast mit GPT-6 Astra käme auf ca. {de(ultrafast)} $ "
         f"pro 1.000 E-Mails – rund {de(round(ultrafast / cost[0], -2))}× JEV.", C.INK_2),
    ]


def fmt_pct(value: float, lang: str) -> str:
    return f"{de(value)} %" if lang == "de" else f"{en(value)}%"


def fmt_usd(value: float, lang: str) -> str:
    return f"{de(value, 2)} $" if lang == "de" else f"${en(value, 2)}"


OVERVIEW_TEXT: dict[str, dict[str, str]] = {
    "de": {
        "title": "JEV: ähnliche Trefferquote wie GPT-6 Luna, {speed}× schneller und {saving} % günstiger",
        "sub": "150 synthetische Geschäfts-E-Mails · je 4 Entscheidungen in einem Aufruf "
               "(Dringlichkeit, Kategorie, Antwort nötig, Phishing) · 2 Durchläufe",
        "q": "Qualität", "q_note": "alle 4 richtig · 95-%-KI", "pct_tick": "{:.0f} %",
        "l": "Geschwindigkeit", "l_note": "Median-Antwortzeit · weniger ist besser",
        "c": "Kosten", "c_note": "$ pro 1.000 E-Mails · Listenpreise",
        "k": "Konstanz", "k_note": "gleiche Antwort in Lauf 1 und 2",
        "foot1": "Referenz: Mehrheit aus GPT-6 Sol, Claude Opus 5.5 und Gemini 3.8 Flash. "
                 "Kosten: gemeldete Tokens × Listenpreise. "
                 "*Simulation, Schwelle auf denselben Daten gewählt.",
        "foot2": "**Nicht gemessen: Lunas Tokenverbrauch × Ultrafast-Preise (60/300 $ pro Mio.), "
                 "ohne Reasoning-Tokens. Code und Daten: {repo}",
    },
    "en": {
        "title": "JEV: similar accuracy to GPT-6 Luna, {speed}× faster and {saving}% cheaper",
        "sub": "150 synthetic business emails · 4 decisions per email in one call "
               "(urgency, category, reply needed, phishing) · 2 runs",
        "q": "Accuracy", "q_note": "all 4 correct · 95% CI", "pct_tick": "{:.0f}%",
        "l": "Speed", "l_note": "median latency · lower is better",
        "c": "Cost", "c_note": "$ per 1,000 emails · list prices",
        "k": "Consistency", "k_note": "same answer in run 1 and 2",
        "foot1": "Reference: majority vote of GPT-6 Sol, Claude Opus 5.5 and Gemini 3.8 Flash. "
                 "Cost: reported tokens × list prices. *Simulation, threshold chosen on the same data.",
        # \\$ statt $: zwei Dollarzeichen in einem Text liest matplotlib sonst als Formel
        "foot2": "**Not measured: Luna's token usage × Ultrafast prices (\\$60/\\$300 per million), "
                 "without reasoning tokens. Code and data: {repo}",
    },
}


def overview(metrics: Mapping[str, object], lang: str = "de") -> Path:
    n = _numbers(metrics)
    quality, ci, latency = n["quality"], n["ci"], n["latency"]
    cost, consistency = n["cost"], n["consistency"]
    assert isinstance(quality, list) and isinstance(ci, list) and isinstance(latency, list)
    assert isinstance(cost, list) and isinstance(consistency, list)
    tx, num = OVERVIEW_TEXT[lang], (de if lang == "de" else en)

    fig = plt.figure(figsize=(12, 7.2), dpi=DPI)
    fig.text(0.04, 0.945, tx["title"].format(speed=num(latency[1] / latency[0]),
                                             saving=num(100 * (1 - cost[0] / cost[1]))),
             fontsize=21, fontweight="bold", color=C.INK, va="top")
    fig.text(0.04, 0.885, tx["sub"], fontsize=12, color=C.INK_2, va="top")

    grid = fig.add_gridspec(1, 4, left=0.155, right=0.975, top=0.73, bottom=0.38, wspace=0.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    panels = (
        Panel(tx["q"], tx["q_note"], quality, [fmt_pct(v, lang) for v in quality], 100, (0, 50, 100),
              tx["pct_tick"], ci),
        Panel(tx["l"], tx["l_note"], latency, [f"{num(v)} ms" for v in latency], 1600, (0, 500, 1000, 1500),
              "{:.0f}"),
        Panel(tx["c"], tx["c_note"], cost, [fmt_usd(v, lang) for v in cost], 3.2, (0, 1, 2, 3), "{:.0f}"),
        Panel(tx["k"], tx["k_note"], consistency, [fmt_pct(v, lang) for v in consistency], 100, (0, 50, 100),
              tx["pct_tick"]),
    )
    for ax, panel in zip(axes, panels, strict=True):
        _panel(ax, panel)
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels([name for _, name in MODELS], fontsize=13)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    axes[0].get_yticklabels()[0].set_color(C.INK)
    for ax in axes[1:]:
        ax.set_yticks([])

    for i, (text, ink) in enumerate(_findings(n, lang)):
        fig.text(0.04, 0.285 - 0.05 * i, text, fontsize=12, color=ink, va="top",
                 fontweight="bold" if ink == C.INK else "normal")
    fig.text(0.04, 0.06, tx["foot1"], fontsize=9.5, color=C.MUTED, va="top")
    fig.text(0.04, 0.03, tx["foot2"].format(repo=REPO), fontsize=9.5, color=C.MUTED, va="top")

    path = OUT / f"ueberblick{'' if lang == 'de' else '-' + lang}{C.suffix}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _card_panel(ax: Axes, panel: Panel) -> None:
    """Panel für das Hochformat: Modellname über jedem Balken, Wert rechts daneben, große Schrift."""
    ax.set_xlim(0, panel.xmax * 1.45)
    ax.set_ylim(len(MODELS) - 0.45, -0.95)
    ax.axis("off")
    for row, ((model, name), v) in enumerate(zip(MODELS, panel.values, strict=True)):
        bold = "bold" if model == "jev" else "normal"
        ax.text(0, row - 0.22, name, va="bottom", ha="left", fontsize=21,
                color=C.INK if model == "jev" else C.INK_2, fontweight=bold)
        ax.barh(row, v, height=0.3, color=color(model), linewidth=0)
        ax.text(v + panel.xmax * 0.04, row, panel.labels[row], va="center", ha="left", fontsize=27,
                color=C.INK, fontweight=bold)
    ax.plot([0, 0], [-0.25, len(MODELS) - 0.7], color=C.BASELINE, linewidth=1.5)
    ax.text(0, 1.1, panel.title, transform=ax.transAxes, fontsize=31, fontweight="bold", color=C.INK,
            va="bottom")
    ax.text(0, 1.035, panel.note, transform=ax.transAxes, fontsize=19, color=C.MUTED, va="bottom")


# Texte der LinkedIn-/X-Grafik; Zahlen werden eingesetzt, nicht eingetippt
CARD_TEXT: dict[str, dict[str, str]] = {
    "de": {
        "kicker": "PRAXISTEST · 150 GESCHÄFTS-E-MAILS · 4 ENTSCHEIDUNGEN PRO E-MAIL",
        "title": "JEV: ähnliche Trefferquote\nwie GPT-6 Luna, aber {speed}× schneller\n"
                 "und {saving} % günstiger",
        "quality": "Qualität", "quality_note": "alle 4 richtig · JEV vs. Luna: p = {p}",
        "latency": "Antwortzeit", "latency_note": "Median · weniger ist besser",
        "cost": "Kosten", "cost_note": "pro 1.000 E-Mails · Listenpreise",
        "consistency": "Konstanz", "consistency_note": "gleiche Antwort in 2 Läufen",
        "pct": "{v} %", "usd": "{v} $",
        "hybrid": "Kombination (Simulation): JEV entscheidet, unsichere Fälle gehen an Luna\n"
                  "→ {acc} % alle 4 richtig, {cost} $ pro 1.000 E-Mails",
        "footer": "Referenz: Mehrheit aus GPT-6 Sol, Claude Opus 5.5, Gemini 3.8 Flash · "
                  "Code und Daten: {repo}",
    },
    "en": {
        "kicker": "HANDS-ON TEST · 150 BUSINESS EMAILS · 4 DECISIONS PER EMAIL",
        "title": "JEV: similar accuracy\nto GPT-6 Luna, but {speed}× faster\nand {saving}% cheaper",
        "quality": "Accuracy", "quality_note": "all 4 correct · JEV vs Luna: p = {p}",
        "latency": "Latency", "latency_note": "median · lower is better",
        "cost": "Cost", "cost_note": "per 1,000 emails · list prices",
        "consistency": "Consistency", "consistency_note": "same answer in 2 runs",
        "pct": "{v}%", "usd": "${v}",
        "hybrid": "Combined (simulation): JEV decides, uncertain cases go to Luna\n"
                  "→ {acc}% all 4 correct, ${cost} per 1,000 emails",
        "footer": "Reference: majority vote of GPT-6 Sol, Claude Opus 5.5, Gemini 3.8 Flash · "
                  "Code and data: {repo}",
    },
}


def linkedin_card(metrics: Mapping[str, object], lang: str = "de") -> Path:
    """Hochformat 4:5 für den Social-Media-Feed: wenige Elemente, Schrift auch auf dem Handy lesbar."""
    n = _numbers(metrics)
    quality, latency, cost, consistency = n["quality"], n["latency"], n["cost"], n["consistency"]
    hybrid = n["hybrid"]
    assert isinstance(quality, list) and isinstance(latency, list) and isinstance(hybrid, dict)
    assert isinstance(cost, list) and isinstance(consistency, list)
    tx, num = CARD_TEXT[lang], (de if lang == "de" else en)

    fig = plt.figure(figsize=(12, 15), dpi=DPI)
    fig.text(0.06, 0.955, tx["kicker"], fontsize=17, color=C.ACCENT, fontweight="bold", va="top")
    fig.text(0.06, 0.925, tx["title"].format(speed=num(latency[1] / latency[0]),
                                             saving=num(100 * (1 - cost[0] / cost[1]))),
             fontsize=46, fontweight="bold", color=C.INK, va="top", linespacing=1.15)

    p_luna = float(str(n["p_luna"]))
    panels = (
        Panel(tx["quality"], tx["quality_note"].format(p=num(p_luna, 1)), quality,
              [tx["pct"].format(v=num(v)) for v in quality], 100, (), ""),
        Panel(tx["latency"], tx["latency_note"], latency, [f"{num(v)} ms" for v in latency], 1100, (), ""),
        Panel(tx["cost"], tx["cost_note"], cost,
              [tx["usd"].format(v=num(v, 2)) for v in cost], 2.3, (), ""),
        Panel(tx["consistency"], tx["consistency_note"], consistency,
              [tx["pct"].format(v=num(v)) for v in consistency], 100, (), ""),
    )
    rects = ((0.06, 0.46, 0.41, 0.215), (0.54, 0.46, 0.41, 0.215),
             (0.06, 0.18, 0.41, 0.215), (0.54, 0.18, 0.41, 0.215))
    for rect, panel in zip(rects, panels, strict=True):
        _card_panel(fig.add_axes(rect), panel)

    fig.text(0.06, 0.13, tx["hybrid"].format(acc=num(100 * hybrid["genauigkeit"]["alle_vier"]),
                                             cost=num(hybrid["kosten_pro_1000_usd"], 2)),
             fontsize=22, color=C.INK, va="top", linespacing=1.35)
    fig.text(0.06, 0.035, tx["footer"].format(repo=REPO), fontsize=13, color=C.MUTED, va="bottom")

    path = OUT / f"linkedin{'' if lang == 'de' else '-' + lang}{C.suffix}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


HYBRID_TEXT: dict[str, dict[str, str]] = {
    "de": {"x": "Anteil der E-Mails, die an das zweite Modell weitergeleitet werden (%)",
           "y": "Alle 4 Entscheidungen richtig (%)",
           "title": "Hybrid: JEV entscheidet, bei Unsicherheit entscheidet ein LLM",
           "sub": "Links nur JEV, rechts nur das zweite Modell · 150 synthetische Geschäfts-E-Mails, "
                  "Lauf 1 · Schwelle auf denselben Daten gewählt"},
    "en": {"x": "Share of emails forwarded to the second model (%)",
           "y": "All 4 decisions correct (%)",
           "title": "Hybrid: JEV decides, an LLM steps in when JEV is unsure",
           "sub": "Left: JEV only, right: second model only · 150 synthetic business emails, "
                  "run 1 · threshold chosen on the same data"},
}


def hybrid_chart(metrics: Mapping[str, object], lang: str = "de") -> Path:
    syn = metrics["datensaetze"]["synthetisch"]  # type: ignore[index]
    num = de if lang == "de" else en
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    fig.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.14)
    for fallback, label, ink in (("gpt-6-luna", "JEV → GPT-6 Luna", C.ACCENT),
                                 ("haiku-4.5", "JEV → Claude Haiku 4.5", C.DEEMPH)):
        pts = sorted(syn["hybrid"][fallback], key=lambda h: h["eskaliert"])
        xs = [100 * h["eskaliert"] for h in pts]
        ys = [100 * h["genauigkeit"]["alle_vier"] for h in pts]
        ax.plot(xs, ys, color=ink, linewidth=2, solid_joinstyle="round", solid_capstyle="round")
        ax.scatter(xs, ys, s=40, color=ink, edgecolors=C.SURFACE, linewidths=2, zorder=3)
        ax.text(xs[-1] + 1.5, ys[-1], label, va="center", fontsize=12, color=C.INK)
    best = max(syn["hybrid"]["gpt-6-luna"], key=lambda h: h["genauigkeit"]["alle_vier"])
    bx, by = 100 * best["eskaliert"], 100 * best["genauigkeit"]["alle_vier"]
    if lang == "en":
        note = (f"{num(by)}% at {num(bx)}% forwarded\n"
                f"${num(best['kosten_pro_1000_usd'], 2)} per 1,000 · avg {num(best['latenz_mittel_ms'])} ms")
    else:
        note = (f"{num(by)} % bei {num(bx)} % Weiterleitung\n"
                f"{num(best['kosten_pro_1000_usd'], 2)} $ pro 1.000 · Ø {num(best['latenz_mittel_ms'])} ms")
    ax.annotate(note, (bx, by), xytext=(bx + 8, by + 5), fontsize=11,
                color=C.INK, arrowprops={"arrowstyle": "-", "color": C.MUTED, "linewidth": 1})
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=C.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-2, 102)
    ax.set_ylim(40, 85)
    tx = HYBRID_TEXT[lang]
    ax.set_xlabel(tx["x"])
    ax.set_ylabel(tx["y"])
    fig.text(0.1, 0.94, tx["title"], fontsize=17, fontweight="bold", color=C.INK, va="top")
    fig.text(0.1, 0.885, tx["sub"], fontsize=11, color=C.INK_2, va="top")
    path = OUT / f"hybrid{'' if lang == 'de' else '-' + lang}{C.suffix}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def latency_chart(rows: Sequence[Mapping[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=DPI)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.76, bottom=0.17)
    data = [[float(str(r["latency_ms"])) for r in rows if r["model"] == m and r.get("answer") is not None]
            for m, _ in MODELS]
    parts = ax.boxplot(data, orientation="horizontal", widths=0.42, showfliers=False, patch_artist=True,
                       medianprops={"color": C.SURFACE, "linewidth": 2},
                       whiskerprops={"color": C.INK_2, "linewidth": 1.2},
                       capprops={"color": C.INK_2, "linewidth": 1.2})
    for patch, (model, _) in zip(parts["boxes"], MODELS, strict=True):
        patch.set_facecolor(color(model))
        patch.set_linewidth(0)
    ax.set_yticks(range(1, len(MODELS) + 1))
    ax.set_yticklabels([name for _, name in MODELS], fontsize=13)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=C.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, None)
    ax.set_xlabel("Antwortzeit pro E-Mail in ms (Box: mittlere 50 %, Linie: Median, Antennen: 1,5 × IQR)")
    fig.text(0.04, 0.94, "Antwortzeiten: JEV braucht ein Viertel der Zeit",
             fontsize=17, fontweight="bold", color=C.INK, va="top")
    n = len(data[0])
    fig.text(0.04, 0.875, f"{n} Aufrufe je Modell (150 E-Mails × 2 Läufe), nacheinander gemessen, "
             "inkl. Netzwerk · Haiku und Luna über Langdock (EU), JEV direkt", fontsize=11, color=C.INK_2,
             va="top")
    path = OUT / f"latenz{C.suffix}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in BENCH.read_text(encoding="utf-8").split("\n") if line] \
        if BENCH.exists() else []
    for theme in THEMES:
        apply_theme(theme)
        paths = [overview(metrics), hybrid_chart(metrics)]
        if theme == "dunkel":
            paths += [overview(metrics, "en"), hybrid_chart(metrics, "en")]  # englischer Blog-Artikel
        paths.append(linkedin_card(metrics))  # hell für LinkedIn, dunkel für die Artikelkarte im Blog
        if theme == "hell":
            paths.append(linkedin_card(metrics, "en"))  # für X
        if rows:
            paths.append(latency_chart([r for r in rows if r["dataset"] == "synthetisch"]))
        for p in paths:
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
