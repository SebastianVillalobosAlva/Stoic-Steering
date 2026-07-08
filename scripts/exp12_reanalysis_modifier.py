"""Exp 12c re-analysis (READ-ONLY): is "Seneca strongest modifier" robust or
outlier-driven? Reads the saved sweep + anchor JSONs, computes per-item max
node shift (Seneca vs Marcus), medians, tie-count, and top-3 character.
No model, no new circuits. Saves a markdown report; never touches the inputs.
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import exp12_sweep as sw  # loaders + ANCHORS map only (no model)

OUT_DIR = sw.OUT_DIR
TIE_FRAC = 0.20  # within 20% counts as an effective tie


def max_shift(base_circ, adapter_circ) -> float:
    """Max |Δ normalized_effect| over nodes shared with base (same metric the
    original report used, kept for comparability)."""
    b = {n["name"]: n["normalized_effect"] for n in base_circ["nodes"]}
    diffs = [abs(n["normalized_effect"] - b[n["name"]]) for n in adapter_circ["nodes"] if n["name"] in b]
    return max(diffs) if diffs else 0.0


def collect(sweep_path: str):
    with open(sweep_path) as f:
        sweep = json.load(f)
    rows = []

    def add(iid, stance, base_c, sen_c, mar_c, is_anchor):
        rows.append(dict(
            item=iid, stance=stance, anchor=is_anchor,
            base_c=sw._c(base_c),
            sen_shift=max_shift(base_c, sen_c), mar_shift=max_shift(base_c, mar_c),
            sen_c=sw._c(sen_c), mar_c=sw._c(mar_c),
            sen_dc=sw._c(sen_c) - sw._c(base_c), mar_dc=sw._c(mar_c) - sw._c(base_c),
        ))

    for iid, entry in sweep["circuits"].items():
        add(iid, sweep["items"][iid]["stance"], entry["base"],
            entry["lora_seneca"], entry["lora_marcus"], False)
    for iid, (stance, fname) in sw.ANCHORS.items():
        with open(OUT_DIR / fname) as f:
            d = json.load(f)
        add(iid, stance, d["circuits"]["base"],
            d["circuits"]["lora_seneca"], d["circuits"]["lora_marcus"], True)
    return rows


def build_report(rows) -> str:
    rows = sorted(rows, key=lambda r: r["sen_shift"], reverse=True)
    L = ["# Exp 12c re-analysis — is 'Seneca strongest modifier' robust?",
         "",
         "Read-only, from saved circuits. Max node shift = max |Δ normalized_effect|",
         "over nodes shared with base (same metric as the original report).",
         "",
         "## Per-item max node shift (sorted by Seneca), Seneca vs Marcus",
         "",
         f"| item | stance | Seneca | Marcus | winner | margin | tie(<{int(TIE_FRAC*100)}%) |",
         "|---|---|---|---|---|---|---|"]
    ties = 0
    sen_wins = 0
    for r in rows:
        s, m = r["sen_shift"], r["mar_shift"]
        win = "seneca" if s > m else "marcus"
        sen_wins += win == "seneca"
        hi, lo = max(s, m), min(s, m)
        margin = (hi - lo) / lo if lo > 0 else float("inf")
        tie = margin < TIE_FRAC
        ties += tie and win == "seneca"
        tag = "★" if r["anchor"] else ""
        L.append(f"| {r['item']}{tag} | {r['stance']} | {s:.3f} | {m:.3f} | "
                 f"{win} | {margin*100:>5.0f}% | {'TIE' if tie else ''} |")
    L.append("")
    L.append("★ = anchor item (loaded from its own single-item JSON).")

    sen = [r["sen_shift"] for r in rows]
    mar = [r["mar_shift"] for r in rows]
    L += ["",
          "## Medians (outlier-robust) vs means",
          "",
          f"| stat | Seneca | Marcus | Seneca/Marcus |",
          "|---|---|---|---|",
          f"| median | {st.median(sen):.3f} | {st.median(mar):.3f} | {st.median(sen)/st.median(mar):.2f}× |",
          f"| mean | {st.mean(sen):.3f} | {st.mean(mar):.3f} | {st.mean(sen)/st.mean(mar):.2f}× |",
          f"| max | {max(sen):.3f} | {max(mar):.3f} | — |",
          "",
          f"Seneca wins {sen_wins}/{len(rows)} items; of those, {ties} are effective ties "
          f"(within {int(TIE_FRAC*100)}%)."]

    # Top-3 character
    L += ["", "## Top-3 Seneca effects — character (steer toward Stoic vs flatten |c|)", ""]
    L.append("| item | base c | Seneca c | Δc | Δ|c| | character |")
    L.append("|---|---|---|---|---|---|")
    for r in rows[:3]:
        d_absc = abs(r["sen_c"]) - abs(r["base_c"])
        if d_absc < -0.3 and abs(r["sen_c"]) < abs(r["base_c"]) * 0.6:
            char = "FLATTENS |c| (disrupts discrimination)"
        elif r["sen_dc"] > 0.3:
            char = "pushes TOWARD Stoic"
        elif r["sen_dc"] < -0.3:
            char = "pushes AWAY from Stoic"
        else:
            char = "mixed / small net"
        L.append(f"| {r['item']} | {r['base_c']:+.3f} | {r['sen_c']:+.3f} | "
                 f"{r['sen_dc']:+.3f} | {d_absc:+.3f} | {char} |")

    # flatten count across all items
    flat = sum(1 for r in rows if abs(r["sen_c"]) < abs(r["base_c"]))
    L += ["", f"Across all {len(rows)} items, Seneca reduces |c| on {flat} and increases it on "
          f"{len(rows)-flat}."]
    return "\n".join(L)


def main(sweep_path: str):
    rows = collect(sweep_path)
    report = build_report(rows)
    print(report)
    out = OUT_DIR / f"exp12_reanalysis_modifier_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out.write_text(report + "\n")
    print(f"\n↳ saved {out.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         str(OUT_DIR / "exp12_sweep_20260708_124712.json"))
