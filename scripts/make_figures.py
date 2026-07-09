"""Publication figures for the README, rendered from the authoritative local
JSONs. Three charts, each with the corrected framing from the re-analyses:

  fig_lora_decision_shift.png  — Exp 11 LoRA ΔP(stoic) by author × stance
  fig_exp12c_node_shift.png    — Seneca vs Marcus max node shift per item (+ medians, ties)
  fig_exp12c_delta_abs_c.png   — Δ|c| per item (diverging; mean vs median; outliers flagged)

Design follows the dataviz skill's validated default palette (light mode).
Run in an env with matplotlib (e.g. the `modellens` conda env).
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import exp12_reanalysis_modifier as RA  # read-only loaders (no model)

OUT = REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- validated palette (dataviz skill, light mode) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"   # categorical slot 1 / diverging cool pole
ORANGE = "#eb6834"  # categorical slot 8
RED = "#e34948"    # diverging warm pole
NEUTRAL = "#f0efec"  # diverging midpoint

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": INK2,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
})


def _despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        if s in keep:
            ax.spines[s].set_color(AXIS)


def _caption(fig, text):
    fig.text(0.008, 0.008, text, ha="left", va="bottom", fontsize=7.5, color=MUTED)


def _header(fig, title, subtitle):
    fig.text(0.012, 0.965, title, fontsize=13.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.895, subtitle, fontsize=9.5, color=INK2, ha="left", va="top")


# ============================================================ Fig 1
def fig_lora_decision_shift():
    d = json.load(open(REPO / "results/stage4_lora_dilemmas/lora_dilemmas_20260705_225558.json"))
    authors = ["seneca", "marcus", "epictetus"]
    labels = {"seneca": "Seneca", "marcus": "Marcus", "epictetus": "Epictetus"}
    stances = [("accepting", BLUE), ("active", ORANGE)]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    h = 0.34
    ypos = {a: i for i, a in enumerate(authors[::-1])}
    for si, (stance, color) in enumerate(stances):
        for a in authors:
            r = d["per_author"][a]["by_stance"][stance]
            dp, t = r["mean_delta"], r["t_stat"]
            y = ypos[a] + (h/2 if si == 0 else -h/2)
            sig = abs(t) >= 2
            ax.barh(y, dp, height=h, color=color, alpha=1.0 if sig else 0.42,
                    edgecolor=SURFACE, linewidth=1.5, zorder=3)
            xoff = 0.001 if dp >= 0 else -0.001
            ax.text(dp + xoff, y, f" {dp:+.3f}  t={t:.2f}{'  *' if sig else ''} ",
                    va="center", ha="left" if dp >= 0 else "right",
                    fontsize=8.5, color=INK, zorder=4)

    ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([labels[a] for a in ypos], fontsize=11, color=INK)
    ax.set_xlim(-0.03, 0.11)
    ax.set_xlabel("ΔP(stoic)  —  merged adapter vs base, forced-choice dilemmas", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax, keep=("left",))
    ax.tick_params(length=0)

    # legend (2 series → always present)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=BLUE, label="accepting stance"),
                       Patch(facecolor=ORANGE, label="active stance")],
              loc="lower right", frameon=False, fontsize=9)
    _header(fig, "LoRA reaches the decision layer — with structure",
            "Seneca moves both stance buckets; Marcus only the 'accepting' one "
            "(passivity prior); Epictetus null.   * = |t| ≥ 2,  faded = n.s.")
    _caption(fig, "Exp 11 · dilemmas_v2 (40 items) · frozen clean adapters · judge-free logits")
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.13, right=0.97)
    fig.savefig(OUT / "fig_lora_decision_shift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("↳ fig_lora_decision_shift.png")


# ============================================================ Fig 2 & 3 shared data
def _rows():
    rows = RA.collect(str(REPO / "results/exp12_circuits/exp12_sweep_20260708_124712.json"))
    for r in rows:
        r["dabs"] = abs(r["sen_c"]) - abs(r["base_c"])
    return rows


def fig_node_shift():
    rows = sorted(_rows(), key=lambda r: r["sen_shift"])
    sen_med = st.median([r["sen_shift"] for r in rows])
    mar_med = st.median([r["mar_shift"] for r in rows])
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i, r in enumerate(rows):
        s, m = r["sen_shift"], r["mar_shift"]
        ax.plot([m, s], [i, i], color=AXIS, lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter(m, i, s=70, color=ORANGE, edgecolor=SURFACE, lw=1.3, zorder=3)
        ax.scatter(s, i, s=70, color=BLUE, edgecolor=SURFACE, lw=1.3, zorder=3)
        tie = min(s, m) > 0 and (max(s, m) - min(s, m)) / min(s, m) < 0.20
        won = "marcus" if m > s else ""
        tag = "  tie" if tie else ("  < Marcus" if won else "")
        ax.text(max(s, m) * 1.04, i, f"{r['item']}{tag}", va="center", fontsize=8,
                color=MUTED if (tie or won) else INK2)

    ax.set_ylim(-2.1, len(rows) - 0.3)
    ax.axvline(sen_med, color=BLUE, lw=1.4, ls=(0, (4, 2)), zorder=1)
    ax.axvline(mar_med, color=ORANGE, lw=1.4, ls=(0, (4, 2)), zorder=1)
    ax.text(sen_med, -0.75, f"Seneca median {sen_med:.2f}", color=BLUE, fontsize=8.5, ha="center", va="top")
    ax.text(mar_med, -1.5, f"Marcus median {mar_med:.2f}", color=ORANGE, fontsize=8.5, ha="center", va="top")

    ax.set_xscale("log")
    ax.set_xlim(0.08, 4.0)
    ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1.0", "2.0"])
    ax.set_yticks([])
    ax.set_xlabel("max node shift vs base  (|Δ normalized patch effect|, log scale)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    ax.tick_params(length=0)

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0], marker="o", color="w", markerfacecolor=BLUE, markersize=9, label="Seneca"),
                       Line2D([0],[0], marker="o", color="w", markerfacecolor=ORANGE, markersize=9, label="Marcus")],
              loc="lower right", frameon=False, fontsize=9.5)
    _header(fig, "Seneca is the largest circuit modifier — robust at the median, not just outliers",
            "Median lead 2.2× survives removing the two outliers (emot_03, emot_01); "
            "3 of 9 'wins' are ties (<20%).")
    _caption(fig, "Exp 12c · 10 items (8 sweep + 2 anchors) · discover_circuit, content metric")
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.04, right=0.90)
    fig.savefig(OUT / "fig_exp12c_node_shift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("↳ fig_exp12c_node_shift.png")


def fig_delta_abs_c():
    rows = sorted(_rows(), key=lambda r: r["dabs"])
    vals = [r["dabs"] for r in rows]
    mean, med = st.mean(vals), st.median(vals)
    outliers = {"emot_03", "ctrl_03"}
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for i, r in enumerate(rows):
        v = r["dabs"]
        color = RED if v < 0 else BLUE
        is_out = r["item"] in outliers
        ax.barh(i, v, height=0.62, color=color, alpha=1.0 if is_out else 0.5,
                edgecolor=INK if is_out else SURFACE, linewidth=1.4 if is_out else 1.5, zorder=3)
        ha = "left" if v >= 0 else "right"
        ax.text(v + (0.03 if v >= 0 else -0.03), i,
                f"{r['item']}  {v:+.2f}" + ("  < outlier" if is_out else ""),
                va="center", ha=ha, fontsize=8, color=INK if is_out else INK2, zorder=4)

    ax.axvline(0, color=AXIS, lw=1.2, zorder=2)
    ax.axvline(mean, color=MUTED, lw=1.4, ls=(0, (5, 2)), zorder=1)
    ax.axvline(med, color="#006300", lw=1.6, ls=(0, (5, 2)), zorder=1)
    ax.text(mean, len(rows)-0.2, f" mean {mean:+.2f}", color=MUTED, fontsize=8.5, ha="right", va="top")
    ax.text(med, -0.9, f"median {med:+.2f} ", color="#006300", fontsize=8.5, ha="left", va="bottom")

    ax.set_xlim(-1.9, 1.5)
    ax.set_yticks([])
    ax.set_xlabel("Δ|c|  =  |Seneca content signal| − |base content signal|", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    ax.tick_params(length=0)
    ax.text(0.015, 0.5, "< flattens (erases A/B distinction)", transform=ax.transAxes,
            fontsize=8.5, color=RED, ha="left", va="center", rotation=0)
    ax.text(0.985, 0.28, "sharpens >", transform=ax.transAxes,
            fontsize=8.5, color=BLUE, ha="right", va="center")

    _header(fig, "Seneca's effect on content discrimination is scatter around zero",
            "5 down / 5 up; median +0.06. The −0.21 mean is carried by two flattening "
            "outliers — remove them and the aggregate flips to +0.60.")
    _caption(fig, "Exp 12c · Δ|c| tracks neither base sign, base magnitude, nor stance (corr −0.27)")
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.05, right=0.97)
    fig.savefig(OUT / "fig_exp12c_delta_abs_c.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("↳ fig_exp12c_delta_abs_c.png")


if __name__ == "__main__":
    fig_lora_decision_shift()
    fig_node_shift()
    fig_delta_abs_c()
    print(f"\nwrote 3 figures to {OUT.relative_to(REPO)}")
