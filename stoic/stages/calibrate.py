"""dilemmas_v3 calibration stage — validate candidates, score on base, gate.

$0, local CPU, judge-free (same one-forward-pass instrument as the v2 ruler).
Reads candidate items from data/generated/ (or --items); writes a calibration
report JSON under results/dilemmas_v3_calibration/. Never touches reference/.
"""

from __future__ import annotations

from pathlib import Path

from stoic.calibrate import (
    CELLS,
    calibration_report,
    load_candidates,
    validate_items,
)
from stoic.dilemmas import eval_dilemmas
from stoic.results_io import write_result


def _print_validation(problems: list[str]) -> None:
    if problems:
        print(f"\nStructural problems ({len(problems)}):")
        for p in problems:
            print(f"  ✗ {p}")
    else:
        print("\nStructure: valid ✓")


def calibrate_stage(
    model,
    tokenizer,
    items_path: str | Path,
    *,
    tolerance: float = 0.05,
    cell_size: int | None = None,
) -> dict:
    print("\n=== dilemmas_v3 calibration (base model, judge-free, $0) ===")
    items = load_candidates(items_path)
    print(f"candidates: {items_path}  ({len(items)} items)")

    problems = validate_items(items, cell_size=cell_size)
    _print_validation(problems)
    if problems:
        print("Fix structure before scoring — calibration not run.")
        return {"items_file": str(items_path), "structural_problems": problems,
                "passed": False}

    scores = eval_dilemmas(model, tokenizer, items)
    report = calibration_report(items, scores, tolerance=tolerance)

    print(f"\nPer-cell baseline P(stoic)  (gate: |mean − 0.5| ≤ {tolerance}):")
    for cell in CELLS:
        c = report["per_cell"][cell]
        mark = "✓" if c["within_tolerance"] else "✗"
        stance = "/".join(str(c["stance_counts"].get(s, 0))
                          for s in ("accepting", "active"))
        print(f"  {cell:16s} n={c['n']:>2}  mean={c['mean_p_stoic']:.4f}  "
              f"|dev|={c['abs_dev']:.4f}  stance {stance}  {mark}")
    print(f"overall mean P(stoic): {report['overall_mean_p_stoic']:.4f}")

    if report["outlier_items"]:
        lo, hi = report["outlier_range"]
        print(f"\nReplacement candidates (P outside [{lo}, {hi}]):")
        for o in report["outlier_items"]:
            print(f"  {o['id']:20s} {o['cell']:16s} P={o['p_stoic']:.4f}")

    gate = report["calibration_gate_passed"]
    print(f"\nCalibration gate: {'PASSED — set is eval-ready' if gate else 'NOT passed — iterate items and re-run'}")

    result = {
        "check": f"per-cell baseline P(stoic) within {tolerance} of 0.5 on the base model, "
                 "both label orders averaged, BEFORE any adapter eval (v1's 0.881 is the cautionary record)",
        "items_file": str(items_path),
        "structural_problems": [],
        **report,
        "passed": gate,
    }
    write_result("dilemmas_v3_calibration", "calibration", result)
    return result
