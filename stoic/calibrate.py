"""dilemmas_v3 calibration harness — the gate BEFORE any eval.

v3 is the 2x2 reasoning-vs-echo instrument (see CLAUDE.md): topic axis
(Letters-core vs off-topic) x phrasing axis (plain vs Stoic-idiom), with
stance balance *within each cell* so the same set serves the circuit sweep.
The calibration gate: per-cell baseline P(stoic) ~= 0.5 on the BASE model,
both label orders averaged, before the set is used for any adapter eval —
v1's 0.881 baseline is the cautionary record.

Candidate items are pipeline data: they live under `data/generated/`
(default `data/generated/dilemmas_v3_candidates.json`), never in
`data/reference/`. Only a calibrated, locked set gets frozen later.

Item schema (v2 fields plus the two v3 axes):

    {
      "id":           "core_plain_01",
      "topic_axis":   "core" | "offtopic",
      "phrasing":     "plain" | "idiom",
      "concept":      "grief",              # free text
      "stoic_stance": "accepting" | "active",
      "situation":    "...",
      "stoic":        "...",                # the Stoic option
      "nonstoic":     "..."
    }

The scoring instrument is the unchanged v2 ruler (`dilemmas.eval_dilemmas`:
one forward pass per label order, judge-free), so calibration numbers are
directly comparable to the 0.542 v2 baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

TOPIC_AXES = ("core", "offtopic")
PHRASINGS = ("plain", "idiom")
STANCES = ("accepting", "active")
CELLS = tuple(f"{t}_{p}" for t in TOPIC_AXES for p in PHRASINGS)
REQUIRED_FIELDS = (
    "id", "topic_axis", "phrasing", "stoic_stance", "situation", "stoic", "nonstoic",
)


def cell_of(item: dict) -> str:
    return f"{item['topic_axis']}_{item['phrasing']}"


def load_candidates(path: str | Path) -> list[dict]:
    with open(path) as f:
        payload = json.load(f)
    return payload["dilemmas"] if isinstance(payload, dict) else payload


def validate_items(items: list[dict], cell_size: int | None = None) -> list[str]:
    """Structural problems with a candidate set ($0, no model).

    Checks: required fields present and non-empty, axis/stance values legal,
    ids unique, every cell populated (to `cell_size` if given), and stance
    balanced within each cell (exact for even cells, off-by-one for odd).
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    for i, d in enumerate(items):
        label = d.get("id", f"item[{i}]")
        for f in REQUIRED_FIELDS:
            if not str(d.get(f, "")).strip():
                problems.append(f"{label}: missing/empty field {f!r}")
        if d.get("topic_axis") not in TOPIC_AXES:
            problems.append(f"{label}: topic_axis {d.get('topic_axis')!r} not in {TOPIC_AXES}")
        if d.get("phrasing") not in PHRASINGS:
            problems.append(f"{label}: phrasing {d.get('phrasing')!r} not in {PHRASINGS}")
        if d.get("stoic_stance") not in STANCES:
            problems.append(f"{label}: stoic_stance {d.get('stoic_stance')!r} not in {STANCES}")
        if d.get("id") in seen_ids:
            problems.append(f"{label}: duplicate id")
        seen_ids.add(d.get("id"))

    by_cell: dict[str, list[dict]] = {c: [] for c in CELLS}
    for d in items:
        if d.get("topic_axis") in TOPIC_AXES and d.get("phrasing") in PHRASINGS:
            by_cell[cell_of(d)].append(d)
    for cell, members in by_cell.items():
        n = len(members)
        if n == 0:
            problems.append(f"cell {cell}: empty")
            continue
        if cell_size is not None and n != cell_size:
            problems.append(f"cell {cell}: {n} items, expected {cell_size}")
        acc = sum(1 for d in members if d.get("stoic_stance") == "accepting")
        act = sum(1 for d in members if d.get("stoic_stance") == "active")
        allowed = 0 if n % 2 == 0 else 1
        if abs(acc - act) > allowed:
            problems.append(
                f"cell {cell}: stance imbalance {acc} accepting / {act} active"
            )
    return problems


def calibration_report(
    items: list[dict],
    scores: dict[str, float],
    *,
    tolerance: float = 0.05,
    outlier_lo: float = 0.2,
    outlier_hi: float = 0.8,
) -> dict:
    """Per-cell calibration stats from per-item P(stoic) scores.

    A cell is calibrated when |mean P(stoic) - 0.5| <= tolerance. Items
    outside [outlier_lo, outlier_hi] are flagged as replacement candidates —
    an extreme item drags its cell and adds ceiling/floor trouble for the
    later log-odds analysis (v1's failure mode).
    """
    per_cell: dict[str, dict] = {}
    for cell in CELLS:
        members = [d for d in items if cell_of(d) == cell]
        if not members:
            per_cell[cell] = {"n": 0, "mean_p_stoic": None, "abs_dev": None,
                              "within_tolerance": False, "stance_counts": {},
                              "per_item": {}}
            continue
        vals = {d["id"]: scores[d["id"]] for d in members}
        mean_p = sum(vals.values()) / len(vals)
        stance_counts = {s: sum(1 for d in members if d["stoic_stance"] == s)
                         for s in STANCES}
        per_cell[cell] = {
            "n": len(members),
            "mean_p_stoic": mean_p,
            "abs_dev": abs(mean_p - 0.5),
            "within_tolerance": abs(mean_p - 0.5) <= tolerance,
            "stance_counts": stance_counts,
            "per_item": dict(sorted(vals.items(), key=lambda kv: kv[1])),
        }

    outliers = sorted(
        (
            {"id": d["id"], "cell": cell_of(d), "p_stoic": scores[d["id"]]}
            for d in items
            if not outlier_lo <= scores[d["id"]] <= outlier_hi
        ),
        key=lambda o: o["p_stoic"],
    )
    populated = [c for c in CELLS if per_cell[c]["n"] > 0]
    all_cells_present = len(populated) == len(CELLS)
    gate = all_cells_present and all(per_cell[c]["within_tolerance"] for c in CELLS)
    return {
        "tolerance": tolerance,
        "outlier_range": [outlier_lo, outlier_hi],
        "n_items": len(items),
        "overall_mean_p_stoic": (
            sum(scores[d["id"]] for d in items) / len(items) if items else None
        ),
        "per_cell": per_cell,
        "outlier_items": outliers,
        "all_cells_present": all_cells_present,
        "calibration_gate_passed": gate,
    }
