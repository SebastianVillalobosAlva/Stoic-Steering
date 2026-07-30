"""v3 behavioral eval — the reasoning-vs-echo verdict (judge-free, $0).

Implements docs/pre-registration-v3.md exactly. That document was frozen before
this script existed, and this script does not choose anything: the statistic,
the three planned comparisons, the multiplicity correction and the control's
role are all read off the pre-registration. If you find yourself wanting to
change a threshold here, that is an amendment to the pre-registration, dated and
reasoned, not an edit to this file.

    python scripts/v3_behavioral_eval.py --dry-run     # wiring + gates, no model
    python scripts/v3_behavioral_eval.py               # the real run

What it establishes (pre-registration §3):

  topic marginal significant, phrasing not  -> topic proximity
  phrasing marginal significant             -> LEXICAL ECHO (weakens the claim)
  offtopic_plain vs zero significant        -> REASONING (the strongest claim)
  both marginals significant                -> mixed; report both, pick neither

An arm's movement counts only if it exceeds BOTH controls on the same comparison
(§4, amended). `lora_control_nonphil` separates content from fine-tuning at all;
`lora_control_neutral` separates *Stoic* content from any strong normative
content. Without them, "LoRA moves decisions" is indistinguishable from "any
fine-tuning on formal prose moves decisions".
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- everything below is fixed by the pre-registration; do not tune here -----
PRIMARY = "logodds"          # §2: paired t on Delta log-odds
SIGN_TEST_TOL = 0.01         # §2: chosen above the observed ~4.3e-3 drift ceiling
ALPHA = 0.05                 # §2: family-wise across the three planned comparisons
# §4 (amended A1): two controls, answering different questions. Neither is
# optional — an arm's movement must exceed BOTH to count as evidence for the axis.
CONTROL_ADAPTERS = {
    "control_nonphil": ROOT / "models" / "lora_control_nonphil",
    "control_neutral": ROOT / "models" / "lora_control_neutral",
}
OUT_DIR = ROOT / "results" / "v3_behavioral"

PLANNED = (
    ("topic_marginal", "core (both phrasings) vs off-topic", "topic proximity"),
    ("phrasing_marginal", "idiom (both topics) vs plain", "lexical echo"),
    ("offtopic_plain_vs_zero", "offtopic_plain cell vs zero", "reasoning"),
)


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down across the three planned comparisons (§2).

    Ordered ascending; comparison i is rejected only if every earlier one was.
    Reported per comparison so the reader sees the threshold each faced.
    """
    ordered = sorted(pvals.items(), key=lambda kv: (kv[1] is None, kv[1]))
    m, out, still = len(ordered), {}, True
    for i, (name, p) in enumerate(ordered):
        thresh = alpha / (m - i)
        rejected = still and p is not None and p <= thresh
        still = rejected
        out[name] = {"p": p, "holm_threshold": thresh, "significant": rejected,
                     "rank": i + 1}
    return out


def gates(items: list[dict]) -> list[str]:
    """Pre-registration conditions that must hold before anything is interpreted."""
    from stoic.calibration import CELLS, cell_of, validate_items

    problems = validate_items(items, cell_size=10)
    payload = json.load(open(ROOT / "data/generated/dilemmas_v3_candidates.json"))
    if "calibration_frozen" not in payload:
        problems.append("item set is NOT frozen (no calibration_frozen block) — "
                        "pre-registration R5 forbids running before the set is frozen")
    if len(items) != 40:
        problems.append(f"expected 40 items, found {len(items)}")
    if set(cell_of(d) for d in items) != set(CELLS):
        problems.append("cells present do not match the declared 2x2")
    for label, path in CONTROL_ADAPTERS.items():
        if not path.exists():
            kind = "neutral" if "neutral" in label else "nonphilosophical"
            problems.append(
                f"control adapter missing: {path.relative_to(ROOT)} — §4 makes it a "
                f"comparison arm, not optional. Build its data with "
                f"`python scripts/prep_control_lora.py --kind {kind}`, then train on Colab.")
    try:
        import scipy  # noqa: F401
    except ImportError:
        problems.append("scipy is required for the independent-sample marginals")
    return problems


def comparisons(items: list[dict], deltas: dict[str, float]) -> dict:
    """The three planned comparisons, on whichever delta scale is primary."""
    from scipy import stats as sps

    from stoic.calibration import cell_of

    core = [deltas[d["id"]] for d in items if d["topic_axis"] == "core"]
    off = [deltas[d["id"]] for d in items if d["topic_axis"] == "offtopic"]
    idiom = [deltas[d["id"]] for d in items if d["phrasing"] == "idiom"]
    plain = [deltas[d["id"]] for d in items if d["phrasing"] == "plain"]
    op = [deltas[d["id"]] for d in items if cell_of(d) == "offtopic_plain"]

    def two_sample(a, b):
        t = sps.ttest_ind(a, b)
        return {"n_a": len(a), "n_b": len(b), "mean_a": sum(a) / len(a),
                "mean_b": sum(b) / len(b), "t": float(t.statistic),
                "p": float(t.pvalue)}

    def one_sample(a):
        t = sps.ttest_1samp(a, 0.0)
        return {"n": len(a), "mean": sum(a) / len(a), "t": float(t.statistic),
                "p": float(t.pvalue)}

    return {
        "topic_marginal": two_sample(core, off),
        "phrasing_marginal": two_sample(idiom, plain),
        "offtopic_plain_vs_zero": one_sample(op),
        # Descriptive only — each group against zero, never confirmatory.
        "descriptive": {k: one_sample(v) for k, v in
                        (("core", core), ("offtopic", off),
                         ("idiom", idiom), ("plain", plain))},
    }


def baseline_confound(items: list[dict], baseline: dict, deltas: dict) -> dict:
    """§R6: does cell-level baseline P predict cell-level movement?

    The cells sit at different baselines (offtopic_plain carries a documented
    +0.062 imbalance), so a cell-wise difference could reflect starting level
    rather than the adapter. This tests that rather than assuming it away.
    """
    from scipy import stats as sps

    from stoic.calibration import CELLS, cell_of

    xs, ys = [], []
    per_cell = {}
    for c in CELLS:
        ids = [d["id"] for d in items if cell_of(d) == c]
        b = sum(baseline[i] for i in ids) / len(ids)
        m = sum(deltas[i] for i in ids) / len(ids)
        per_cell[c] = {"baseline": b, "mean_delta": m}
        xs.append(b)
        ys.append(m)
    r = sps.pearsonr(xs, ys)
    return {"per_cell": per_cell, "pearson_r": float(r.statistic),
            "p": float(r.pvalue), "n_cells": len(xs),
            "note": "4 cells only — this is a flag, not a test with power. A large |r| "
                    "means report the imbalance alongside every cell comparison."}


def run(arms: list[str] | None = None) -> dict:
    import torch  # noqa: F401

    from stoic import config, lora
    from stoic.calibration import load_candidates
    from stoic.dilemmas import (_logit, eval_dilemmas, mean, paired_stats,
                                sign_test)
    from stoic.model import load_model

    items = load_candidates(config.GENERATED_DIR / "dilemmas_v3_candidates.json")
    problems = gates(items)
    if problems:
        print(f"GATES FAILED — {len(problems)} problem(s), nothing run:")
        for p in problems:
            print(f"  ✗ {p}")
        raise SystemExit(2)
    print("gates: item set frozen, 2x2 intact, control adapter present ✓")

    model, tokenizer = load_model()
    print("\nbase model on v3 ...")
    baseline = eval_dilemmas(model, tokenizer, items)
    print(f"  base mean P(target) = {mean(baseline):.6f}")

    # Base integrity uses the v2 ruler, whose value is the load-bearing 0.542.
    from stoic.dilemmas import load_dilemmas
    v2 = load_dilemmas()
    v2_start = mean(eval_dilemmas(model, tokenizer, v2))
    print(f"  base integrity on v2 = {v2_start:.6f} (expect {config.DILEMMA_BASELINE})")

    todo = {name: config.ARMS[name].adapter_dir for name in (arms or config.ARMS)}
    todo.update(CONTROL_ADAPTERS)

    per_arm = {}
    for name, adapter in todo.items():
        print(f"\n[{name}] {adapter.name}")
        merged = lora.merge_adapter(adapter)
        try:
            steered = eval_dilemmas(merged, tokenizer, items)
        finally:
            del merged
            gc.collect()
        dp = {i: steered[i] - baseline[i] for i in steered}
        dlo = {i: _logit(steered[i]) - _logit(baseline[i]) for i in steered}
        primary = dlo if PRIMARY == "logodds" else dp
        per_arm[name] = {
            "adapter": adapter.name,
            "steered_mean": mean(steered),
            "steered_p_target": steered,
            "overall_dP": paired_stats(list(dp.values())),
            "overall_logodds": paired_stats(list(dlo.values())),
            "sign_test_tol": sign_test(dp, tol=SIGN_TEST_TOL),
            "planned": comparisons(items, primary),
            "baseline_confound": baseline_confound(items, baseline, primary),
        }
        pv = {k: per_arm[name]["planned"][k]["p"] for k, _, _ in PLANNED}
        per_arm[name]["holm"] = holm(pv)
        st = per_arm[name]["sign_test_tol"]
        print(f"  Δlog-odds {per_arm[name]['overall_logodds']['mean_delta']:+.4f} "
              f"(t={per_arm[name]['overall_logodds']['t_stat']:+.2f})   "
              f"ΔP {per_arm[name]['overall_dP']['mean_delta']:+.4f}")
        print(f"  sign test @tol={SIGN_TEST_TOL}: +{st['pos']}/-{st['neg']} "
              f"(ties {st['ties']}, dropped as within-drift)")
        for key, desc, reading in PLANNED:
            h = per_arm[name]["holm"][key]
            print(f"    {desc:38s} p={h['p']:.4f} vs {h['holm_threshold']:.4f}  "
                  f"{'SIG' if h['significant'] else 'ns '}  ({reading})")

    v2_end = mean(eval_dilemmas(model, tokenizer, v2))
    print(f"\nbase integrity after merges: {v2_end:.6f} (drift {abs(v2_end - v2_start):.2e})")

    # §4: an arm counts only if it beats BOTH controls on the same comparison.
    def effect(a, key):
        pl = a["planned"][key]
        return abs(pl["mean"]) if "mean" in pl else abs(pl["mean_a"] - pl["mean_b"])

    for name, a in per_arm.items():
        if name in CONTROL_ADAPTERS:
            continue
        a["exceeds_control"] = {
            key: {c: effect(a, key) > effect(per_arm[c], key) for c in CONTROL_ADAPTERS}
            for key, _, _ in PLANNED
        }
        a["exceeds_all_controls"] = {
            key: all(v.values()) for key, v in a["exceeds_control"].items()
        }
        print(f"\n[{name}] exceeds both controls:")
        for key, _, reading in PLANNED:
            print(f"    {key:24s} {a['exceeds_all_controls'][key]}   ({reading})")

    result = {
        "experiment": "v3_behavioral",
        "pre_registration": "docs/pre-registration-v3.md (frozen before this run)",
        "primary_statistic": f"paired t on Delta {PRIMARY}",
        "sign_test_tolerance": SIGN_TEST_TOL,
        "alpha_familywise": ALPHA,
        "n_items": len(items),
        "base_mean_p_target": mean(baseline),
        "baseline_p_target": baseline,
        "v2_integrity": {"start": v2_start, "end": v2_end,
                         "drift": abs(v2_end - v2_start),
                         "expected": config.DILEMMA_BASELINE},
        "per_arm": per_arm,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from stoic.results_io import write_result
    write_result("v3_behavioral", "v3_eval", result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true",
                   help="check gates and wiring, load no model, write nothing")
    p.add_argument("--arms", nargs="*", default=None,
                   help="subset of philosopher arms (control is always included)")
    args = p.parse_args()

    if args.dry_run:
        from stoic.calibration import load_candidates
        from stoic import config

        items = load_candidates(config.GENERATED_DIR / "dilemmas_v3_candidates.json")
        problems = gates(items)
        print(f"items: {len(items)}")
        print(f"primary statistic: paired t on Delta {PRIMARY}")
        print(f"sign-test tolerance: {SIGN_TEST_TOL}   family-wise alpha: {ALPHA}")
        print("planned comparisons:")
        for key, desc, reading in PLANNED:
            print(f"  {key:24s} {desc:38s} -> {reading}")
        # Exercise the maths on synthetic deltas so the wiring is proven $0.
        fake = {d["id"]: 0.05 for d in items}
        c = comparisons(items, fake)
        pv = {k: c[k]["p"] for k, _, _ in PLANNED}
        h = holm(pv)
        print("\nwiring check on synthetic uniform +0.05 deltas:")
        print(f"  offtopic_plain_vs_zero mean = {c['offtopic_plain_vs_zero']['mean']:.3f} "
              f"(expect 0.050)")
        print(f"  topic marginal p = {pv['topic_marginal']:.3f} (expect ~1.0, no difference)")
        print(f"  holm ranks: {[(k, v['rank']) for k, v in h.items()]}")
        print()
        if problems:
            print(f"GATES: {len(problems)} blocker(s)")
            for pr in problems:
                print(f"  ✗ {pr}")
            return 1
        print("GATES: all clear — ready to run")
        return 0

    run(args.arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
