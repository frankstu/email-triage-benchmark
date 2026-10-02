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
# "hell" für LinkedIn, "dunkel" passend zum Blog (Fläche wie dessen Karten).
THEMES: dict[str, dict[str, str]] = {
    "hell": {"ACCENT": "#2a78d6", "DEEMPH": "#a9a79f", "SURFACE": "#fcfcfb", "INK": "#0b0b0b",
             "INK_2": "#52514e", "MUTED": "#898781", "GRID": "#e1e0d9", "BASELINE": "#c3c2b7", "suffix": ""},
    "dunkel": {"ACCENT": "#3987e5", "DEEMPH": "#6e6e76", "SURFACE": "#121214", "INK": "#f4f4f5",
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


def _findings(n: Mapping[str, object]) -> list[tuple[str, str]]:
    cost, hybrid, real = n["cost"], n["hybrid"], n["real"]
    assert isinstance(cost, list) and isinstance(hybrid, dict) and isinstance(real, dict)
    p_luna = float(str(n["p_luna"]))
    sig_text = "kein signifikanter Unterschied" if p_luna >= 0.05 else "signifikanter Unterschied"
    later = 100 * float(str(n["later_share"]))
    tokens = n["luna_tokens"]
    assert isinstance(tokens, tuple)
    ultrafast = (tokens[0] * ULTRAFAST_ASTRA_USD[0] + tokens[1] * ULTRAFAST_ASTRA_USD[1]) / 1_000_000 * 1000
    return [
        (f"Qualität JEV vs. Luna: {sig_text} (McNemar p = {de(p_luna, 2)}). "
         f"Claude Haiku: signifikant schlechter und {de(cost[2] / cost[0])}× so teuer wie JEV.", C.INK_2),
        (f"Hybrid – JEV entscheidet, unsichere Fälle ({de(100 * hybrid['eskaliert'])} %) gehen an Luna: "
         f"{de(100 * hybrid['genauigkeit']['alle_vier'])} % alle 4 richtig · "
         f"{de(hybrid['kosten_pro_1000_usd'], 2)} $ pro 1.000 · "
         f"Ø {de(hybrid['latenz_mittel_ms'])} ms*", C.INK),
        (f"Echtes privates Postfach (300 E-Mails, {de(later)} % davon „später“): "
         f"Haiku {de(100 * real['haiku-4.5'], 1)} %, Luna {de(100 * real['gpt-6-luna'], 1)} %, "
         f"JEV {de(100 * real['jev'], 1)} % alle 4 richtig.", C.INK_2),
        (f"Zum Vergleich, rechnerisch**: OpenAI Ultrafast mit GPT-6 Astra käme auf ca. {de(ultrafast)} $ "
         f"pro 1.000 E-Mails – rund {de(round(ultrafast / cost[0], -2))}× JEV.", C.INK_2),
    ]


def overview(metrics: Mapping[str, object]) -> Path:
    n = _numbers(metrics)
    quality, ci, latency = n["quality"], n["ci"], n["latency"]
    cost, consistency = n["cost"], n["consistency"]
    assert isinstance(quality, list) and isinstance(ci, list) and isinstance(latency, list)
    assert isinstance(cost, list) and isinstance(consistency, list)

    fig = plt.figure(figsize=(12, 7.2), dpi=DPI)
    fig.text(0.04, 0.945, f"JEV: ähnliche Trefferquote wie GPT-6 Luna, {de(latency[1] / latency[0])}× "
             f"schneller und {de(100 * (1 - cost[0] / cost[1]))} % günstiger", fontsize=21, fontweight="bold",
             color=C.INK, va="top")
    fig.text(0.04, 0.885, "150 synthetische Geschäfts-E-Mails · je 4 Entscheidungen in einem Aufruf "
             "(Dringlichkeit, Kategorie, Antwort nötig, Phishing) · 2 Durchläufe",
             fontsize=12, color=C.INK_2, va="top")

    grid = fig.add_gridspec(1, 4, left=0.155, right=0.975, top=0.73, bottom=0.38, wspace=0.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    panels = (
        Panel("Qualität", "alle 4 richtig · 95-%-KI", quality, [f"{de(v)} %" for v in quality],
              100, (0, 50, 100), "{:.0f} %", ci),
        Panel("Geschwindigkeit", "Median-Antwortzeit · weniger ist besser", latency,
              [f"{de(v)} ms" for v in latency], 1600, (0, 500, 1000, 1500), "{:.0f}"),
        Panel("Kosten", "$ pro 1.000 E-Mails · Listenpreise", cost, [f"{de(v, 2)} $" for v in cost],
              3.2, (0, 1, 2, 3), "{:.0f}"),
        Panel("Konstanz", "gleiche Antwort in Lauf 1 und 2", consistency,
              [f"{de(v)} %" for v in consistency], 100, (0, 50, 100), "{:.0f} %"),
    )
    for ax, panel in zip(axes, panels, strict=True):
        _panel(ax, panel)
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels([name for _, name in MODELS], fontsize=13)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    axes[0].get_yticklabels()[0].set_color(C.INK)
    for ax in axes[1:]:
        ax.set_yticks([])

    for i, (text, ink) in enumerate(_findings(n)):
        fig.text(0.04, 0.285 - 0.05 * i, text, fontsize=12, color=ink, va="top",
                 fontweight="bold" if ink == C.INK else "normal")
    fig.text(0.04, 0.06, "Referenz: Mehrheit aus GPT-6 Sol, Claude Opus 5.5 und Gemini 3.8 Flash. "
             "Kosten: gemeldete Tokens × Listenpreise. "
             "*Simulation, Schwelle auf denselben Daten gewählt.",
             fontsize=9.5, color=C.MUTED, va="top")
    fig.text(0.04, 0.03, f"**Nicht gemessen: Lunas Tokenverbrauch × Ultrafast-Preise (60/300 $ pro Mio.), "
             f"ohne Reasoning-Tokens. Code und Daten: {REPO}", fontsize=9.5, color=C.MUTED, va="top")

    path = OUT / f"ueberblick{C.suffix}.png"
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


def linkedin_card(metrics: Mapping[str, object]) -> Path:
    """Hochformat 4:5 für den LinkedIn-Feed: wenige Elemente, Schrift auch auf dem Handy lesbar."""
    n = _numbers(metrics)
    quality, latency, cost, consistency = n["quality"], n["latency"], n["cost"], n["consistency"]
    hybrid = n["hybrid"]
    assert isinstance(quality, list) and isinstance(latency, list) and isinstance(hybrid, dict)
    assert isinstance(cost, list) and isinstance(consistency, list)

    fig = plt.figure(figsize=(12, 15), dpi=DPI)
    fig.text(0.06, 0.955, "PRAXISTEST · 150 GESCHÄFTS-E-MAILS · 4 ENTSCHEIDUNGEN PRO E-MAIL",
             fontsize=17, color=C.ACCENT, fontweight="bold", va="top")
    fig.text(0.06, 0.925, f"JEV: ähnliche Trefferquote\nwie GPT-6 Luna, aber "
             f"{de(latency[1] / latency[0])}× schneller\nund {de(100 * (1 - cost[0] / cost[1]))} % günstiger",
             fontsize=46, fontweight="bold", color=C.INK, va="top", linespacing=1.15)

    p_luna = float(str(n["p_luna"]))
    panels = (
        Panel("Qualität", f"alle 4 richtig · JEV vs. Luna: p = {de(p_luna, 1)}", quality,
              [f"{de(v)} %" for v in quality], 100, (), ""),
        Panel("Antwortzeit", "Median · weniger ist besser", latency,
              [f"{de(v)} ms" for v in latency], 1100, (), ""),
        Panel("Kosten", "pro 1.000 E-Mails · Listenpreise", cost,
              [f"{de(v, 2)} $" for v in cost], 2.3, (), ""),
        Panel("Konstanz", "gleiche Antwort in 2 Läufen", consistency,
              [f"{de(v)} %" for v in consistency], 100, (), ""),
    )
    rects = ((0.06, 0.46, 0.41, 0.215), (0.54, 0.46, 0.41, 0.215),
             (0.06, 0.18, 0.41, 0.215), (0.54, 0.18, 0.41, 0.215))
    for rect, panel in zip(rects, panels, strict=True):
        _card_panel(fig.add_axes(rect), panel)

    fig.text(0.06, 0.13, f"Kombination (Simulation): JEV entscheidet, unsichere Fälle gehen an Luna\n"
             f"→ {de(100 * hybrid['genauigkeit']['alle_vier'])} % alle 4 richtig, "
             f"{de(hybrid['kosten_pro_1000_usd'], 2)} $ pro 1.000 E-Mails",
             fontsize=22, color=C.INK, va="top", linespacing=1.35)
    fig.text(0.06, 0.035, "Referenz: Mehrheit aus GPT-6 Sol, Claude Opus 5.5, Gemini 3.8 Flash · "
             f"Code und Daten: {REPO}", fontsize=13, color=C.MUTED, va="bottom")

    path = OUT / f"linkedin{C.suffix}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def hybrid_chart(metrics: Mapping[str, object]) -> Path:
    syn = metrics["datensaetze"]["synthetisch"]  # type: ignore[index]
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
    note = (f"{de(by)} % bei {de(bx)} % Weiterleitung\n"
            f"{de(best['kosten_pro_1000_usd'], 2)} $ pro 1.000 · Ø {de(best['latenz_mittel_ms'])} ms")
    ax.annotate(note, (bx, by), xytext=(bx + 8, by + 5), fontsize=11,
                color=C.INK, arrowprops={"arrowstyle": "-", "color": C.MUTED, "linewidth": 1})
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=C.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-2, 102)
    ax.set_ylim(40, 85)
    ax.set_xlabel("Anteil der E-Mails, die an das zweite Modell weitergeleitet werden (%)")
    ax.set_ylabel("Alle 4 Entscheidungen richtig (%)")
    fig.text(0.1, 0.94, "Hybrid: JEV entscheidet, bei Unsicherheit entscheidet ein LLM",
             fontsize=17, fontweight="bold", color=C.INK, va="top")
    fig.text(0.1, 0.885, "Links nur JEV, rechts nur das zweite Modell · 150 synthetische Geschäfts-E-Mails, "
             "Lauf 1 · Schwelle auf denselben Daten gewählt", fontsize=11, color=C.INK_2, va="top")
    path = OUT / f"hybrid{C.suffix}.png"
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
        if theme == "hell":
            paths.append(linkedin_card(metrics))
        if rows:
            paths.append(latency_chart([r for r in rows if r["dataset"] == "synthetisch"]))
        for p in paths:
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
