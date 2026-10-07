"""Charts. Each function draws one figure and returns it (and saves it if given a path)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .model import Result

# Colour follows the model, never its rank, so a model keeps its colour in every chart.
MODEL_COLOURS = {"triangular": "#2a78d6", "lognormal": "#eb6834", "pert": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _money(symbol):
    return FuncFormatter(lambda v, _: f"{symbol}{v:,.0f}")


def _save(fig, path):
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def distribution(result: Result, path=None, currency="£"):
    """Histogram of total project cost with P10 / P50 / P90 marked."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(result.total, bins=60, color=MODEL_COLOURS[result.distribution],
            edgecolor=SURFACE, linewidth=0.6)
    top = ax.get_ylim()[1]
    for p, style in ((10, ":"), (50, "--"), (90, ":")):
        v = np.percentile(result.total, p)
        ax.axvline(v, color=INK_2, linestyle=style, linewidth=1.4)
        ax.text(v, top * 1.04, f"P{p}  {currency}{v:,.0f}", ha="center", va="bottom",
                fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none"))
    ax.set_ylim(0, top * 1.14)
    ax.xaxis.set_major_formatter(_money(currency))
    _style(ax, f"Total project cost: {result.name} model "
               f"({len(result.total):,} simulations)", "Total project cost", "Simulations")
    return _save(fig, path)


def s_curve(results: dict[str, Result], path=None, currency="£"):
    """Cumulative probability (S-curve) for each model on one chart."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, res in results.items():
        xs = np.sort(res.total)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=MODEL_COLOURS[key], linewidth=2, label=res.name)
        p80 = np.percentile(res.total, 80)
        ax.plot([p80], [0.8], "o", markersize=8, color=MODEL_COLOURS[key],
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axhline(0.8, color=INK_2, linewidth=0.9, linestyle="--")
    ax.text(ax.get_xlim()[0], 0.815, " P80: the usual budget confidence level",
            fontsize=9, color=INK_2, va="bottom")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(_money(currency))
    ax.legend(frameon=False, loc="lower right")
    _style(ax, "Chance the project comes in at or under each cost",
           "Total project cost", "Cumulative probability")
    return _save(fig, path)


def tornado(sens: pd.DataFrame, result: Result, path=None, top=12):
    """Horizontal bars: the tasks that drive the most uncertainty."""
    d = sens.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(d) + 1.6))
    bars = ax.barh(d["Task"], d["Share of variance (%)"], height=0.6,
                   color=MODEL_COLOURS[result.distribution])
    for bar, v in zip(bars, d["Share of variance (%)"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {v:.1f}%",
                va="center", fontsize=9, color=INK)
    ax.set_xlim(0, d["Share of variance (%)"].max() * 1.18)
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    _style(ax, f"What drives the uncertainty: {result.name} model",
           "Share of the variance in total cost", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return _save(fig, path)


def milestones(ms: pd.DataFrame, result: Result, path=None, currency="£"):
    """P50 dot with a P10-P90 whisker per milestone, base estimate as a hollow marker."""
    d = ms.iloc[::-1].reset_index(drop=True)
    colour = MODEL_COLOURS[result.distribution]
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(d) + 1.8))
    y = np.arange(len(d))
    ax.hlines(y, d["P10"], d["P90"], color=colour, linewidth=2)
    ax.plot(d["P50"], y, "o", markersize=9, color=colour, markeredgecolor=SURFACE,
            markeredgewidth=2, label="P50, with P10 to P90 range")
    ax.plot(d["Base estimate"], y, "D", markersize=7, markerfacecolor=SURFACE,
            markeredgecolor=INK_2, label="Base estimate (no risk)")
    ax.set_yticks(y, d["Milestone"])
    ax.xaxis.set_major_formatter(_money(currency))
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _style(ax, f"Cost by milestone: {result.name} model", "Cost", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Schedule, joint confidence and risk-register charts (integrated model)
# ---------------------------------------------------------------------------
def finish_s_curve(res, path=None):
    """Cumulative probability of finishing by each date, with the no-risk date marked."""
    sr = res.schedule
    days = np.sort(sr.project_days)
    dates = [sr.finish_date(d) for d in days[:: max(1, len(days) // 400)]]
    ys = np.linspace(0, 1, len(dates))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(dates, ys, color=MODEL_COLOURS[res.distribution], linewidth=2)
    det = sr.finish_date(sr.deterministic_days)
    p_det = float((sr.project_days <= sr.deterministic_days).mean())
    ax.axvline(det, color=INK_2, linestyle=":", linewidth=1.2)
    ax.text(det, 0.6, f"Plan date {det:%d %b %Y}\n{p_det:.0%} chance of making it  ", fontsize=9, color=INK, ha="right")
    for p in (50, 80):
        d = sr.finish_date(np.percentile(sr.project_days, p))
        ax.plot([d], [p / 100], "o", markersize=8, color=MODEL_COLOURS[res.distribution],
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(d, p / 100, f"  P{p}  {d:%d %b %Y}", va="center", fontsize=9, color=INK)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style(ax, "Chance of finishing by each date", "Finish date", "Cumulative probability")
    return _save(fig, path)


def criticality(sens: pd.DataFrame, path=None, top=12):
    d = sens.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(d) + 1.6))
    bars = ax.barh(d["Task"], d["Criticality index"], height=0.6, color=MODEL_COLOURS["pert"])
    for bar, v in zip(bars, d["Criticality index"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {v:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    _style(ax, "How often each task is on the critical path", "Criticality index (share of simulations)", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return _save(fig, path)


def jcl_scatter(res, budget, days, path=None, currency="£", sample=3000):
    """Each dot is one simulated project. The shaded box is 'on budget and on time'."""
    c, t = res.total_cost, res.project_days
    idx = np.random.default_rng(0).choice(len(c), min(sample, len(c)), replace=False)
    ok = (c[idx] <= budget) & (t[idx] <= days)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(t[idx][~ok], c[idx][~ok], s=7, color="#b7b6b0", alpha=0.6, linewidths=0, label="Over budget or late")
    ax.scatter(t[idx][ok], c[idx][ok], s=7, color=MODEL_COLOURS[res.distribution], alpha=0.6, linewidths=0,
               label="On budget and on time")
    ax.axhline(budget, color=INK_2, linestyle="--", linewidth=1)
    ax.axvline(days, color=INK_2, linestyle="--", linewidth=1)
    jc = float(((c <= budget) & (t <= days)).mean())
    ax.set_title("")
    ax.yaxis.set_major_formatter(_money(currency))
    ax.legend(frameon=False, loc="upper left", fontsize=9, markerscale=3)
    _style(ax, f"Joint confidence: {jc:.0%} chance of meeting both {currency}{budget:,.0f} "
               f"and {days:.0f} working days", "Project duration (working days)", "Total cost")
    return _save(fig, path)


def risk_bars(ranking: pd.DataFrame, path=None, currency="£"):
    col = [c for c in ranking.columns if c.endswith("cost saved if removed")][0]
    d = ranking.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(d) + 1.6))
    bars = ax.barh(d["Risk ID"] + "  " + d["Risk"].str.slice(0, 48), d[col], height=0.6, color=MODEL_COLOURS["lognormal"])
    for bar, v in zip(bars, d[col]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {currency}{v:,.0f}", va="center", fontsize=9)
    ax.set_xlim(0, max(d[col].max(), 1) * 1.2)
    ax.xaxis.set_major_formatter(_money(currency))
    _style(ax, "What each risk adds to the P80 budget (saving if fully mitigated)", "Reduction in P80 cost", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    return _save(fig, path)
