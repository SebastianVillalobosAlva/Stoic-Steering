"""Stages 0-2 — the local, $0 Pass A checkpoints.

Stage 0: deterministic decoding (same prompt -> identical output twice).
Stage 1: base P(stoic) == 0.542 on the v2 dilemma set (load-bearing).
Stage 2: new vectors cosine >=0.99 vs frozen; injection bites hidden_states[L+1];
         steered dilemmas flat (the CAA decision null).
"""

from __future__ import annotations

import torch

from stoic import config
from stoic.axis import ACTIVE
from stoic.dilemmas import (
    deltas_by_stance,
    eval_dilemmas,
    load_dilemmas,
    mean,
    paired_stats,
    _logit,
)
from stoic.model import generate
from stoic.results_io import write_result
from stoic.steering import extract_vector, load_pairs, load_reference_vector, steering


def stage0(model, tokenizer) -> dict:
    print("\n=== Stage 0: deterministic decoding ===")
    prompt = config.DEFAULT_PROMPTS[0]
    out1 = generate(model, tokenizer, prompt)
    out2 = generate(model, tokenizer, prompt)
    identical = out1 == out2
    print(f"prompt: {prompt!r}")
    print(f"identical twice: {identical}")
    result = {
        "stage": 0,
        "check": "same prompt -> identical output twice",
        "prompt": prompt,
        "output_1": out1,
        "output_2": out2,
        "identical": identical,
        "passed": identical,
    }
    write_result("stage0_determinism", "determinism", result)
    print(f"Stage 0: {'PASS' if identical else 'FAIL'}")
    return result


def stage1(model, tokenizer) -> dict:
    target_name = ACTIVE.target_name
    baseline_target = ACTIVE.criteria.get("decision_baseline")
    print(f"\n=== Stage 1: base P({target_name}) == {baseline_target} (load-bearing) ===")
    dilemmas = load_dilemmas()
    baseline = eval_dilemmas(model, tokenizer, dilemmas)
    base_mean = mean(baseline)
    # Checkpoint: matches reference to the 3rd decimal. An axis with no
    # established baseline yet (a new axis) has nothing to check against.
    passed = baseline_target is not None and round(base_mean, 3) == baseline_target
    print(f"n_dilemmas: {len(dilemmas)}  (x2 label orders)")
    print(f"baseline mean P({target_name}): {base_mean:.6f}  (target {baseline_target})")
    result = {
        "stage": 1,
        "check": f"base P({target_name}) == {baseline_target:g} on "
                 f"{ACTIVE.dilemmas_file.stem}" if baseline_target is not None
                 else f"base P({target_name}) on {ACTIVE.dilemmas_file.stem} (no established baseline)",
        "n_dilemmas": len(dilemmas),
        "baseline_mean": base_mean,
        "target": baseline_target,
        "reference_exact": ACTIVE.criteria.get("decision_baseline_exact"),
        "passed": passed,
        "baseline_p_stoic": baseline,
    }
    write_result("stage1_dilemma_baseline", "baseline", result)
    print(f"Stage 1: {'PASS' if passed else 'FAIL'}")
    return result, baseline


@torch.no_grad()
def _injection_site_check(model, tokenizer, layer: int, vector: torch.Tensor, coeff: float) -> dict:
    """Injecting at layer L's MLP must change hidden_states[L+1] but not [L].
    (HF hidden_states[0]=embeddings, hidden_states[i]=output of layer i-1.)"""
    prompt = "The wise person is one who"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def hs():
        return model(**inputs, output_hidden_states=True).hidden_states

    clean = hs()
    with steering(model, layer, vector, coeff):
        steered = hs()

    before = torch.equal(clean[layer], steered[layer])          # unchanged
    after_changed = not torch.equal(clean[layer + 1], steered[layer + 1])  # bitten
    return {
        "layer": layer,
        "hidden_states[L]_unchanged": before,
        "hidden_states[L+1]_changed": after_changed,
        "passed": before and after_changed,
    }


def stage2(model, tokenizer, baseline: dict | None = None) -> dict:
    print("\n=== Stage 2: vector cosine >=0.99 + steered dilemmas flat ===")
    dilemmas = load_dilemmas()
    if baseline is None:
        baseline = eval_dilemmas(model, tokenizer, dilemmas)
    base_mean = mean(baseline)

    per_author = {}
    cos = torch.nn.functional.cosine_similarity

    for name, author in config.ARMS.items():
        print(f"\n[{name}] layer {author.layer}, coeff {author.coeff}")
        pairs = load_pairs(author.pairs_file)
        new_vec = extract_vector(model, tokenizer, pairs, author.layer)
        ref_vec = load_reference_vector(author.vector_file, author.layer)

        cosine = cos(new_vec.float().unsqueeze(0), ref_vec.float().unsqueeze(0)).item()
        norm_ratio = (new_vec.float().norm() / ref_vec.float().norm()).item()
        print(f"  cosine(new, frozen) = {cosine:.4f}   |new|/|frozen| = {norm_ratio:.3f}")

        # Steer with the newly-extracted vector: tests the whole rebuilt path.
        steered = eval_dilemmas(
            model, tokenizer, dilemmas, steer=(author.layer, new_vec, author.coeff)
        )
        deltas = {i: steered[i] - baseline[i] for i in steered}
        deltas_lo = {i: _logit(steered[i]) - _logit(baseline[i]) for i in steered}
        overall = paired_stats(list(deltas.values()))
        print(
            f"  steered mean P({ACTIVE.target_name}) = {mean(steered):.4f}   "
            f"ΔP = {overall['mean_delta']:+.4f}   t = {overall['t_stat']:+.2f}"
        )

        per_author[name] = {
            "layer": author.layer,
            "coeff": author.coeff,
            "cosine_to_frozen": cosine,
            "norm_ratio_new_over_frozen": norm_ratio,
            "steered_mean": mean(steered),
            "overall": overall,
            "overall_logodds": paired_stats(list(deltas_lo.values())),
            "by_stance": deltas_by_stance(dilemmas, deltas),
        }

    # Injection-site mechanism check on the axis's designated arm (stoic: Epictetus L8).
    site_arm = config.ARMS[ACTIVE.criteria["injection_site_arm"]]
    site_vec = load_reference_vector(site_arm.vector_file, site_arm.layer)
    site = _injection_site_check(model, tokenizer, site_arm.layer, site_vec, site_arm.coeff)
    print(
        f"\ninjection site L{site_arm.layer}: hs[L] unchanged={site['hidden_states[L]_unchanged']}, "
        f"hs[L+1] changed={site['hidden_states[L+1]_changed']}"
    )

    cosine_min = ACTIVE.criteria["cosine_min"]
    flat_max = ACTIVE.criteria["caa_flat_max_abs_dp"]
    cosines_ok = all(a["cosine_to_frozen"] >= cosine_min for a in per_author.values())
    # "Flat" = every arm's |ΔP| small (Exp 10 null). Reference ΔP were ~1e-3.
    flat_ok = all(abs(a["overall"]["mean_delta"]) < flat_max for a in per_author.values())
    passed = cosines_ok and flat_ok and site["passed"]

    result = {
        "stage": 2,
        "check": f"cosine>={cosine_min:g} vs frozen; injection bites at L+1; "
                 "steered dilemmas flat (Exp 10 null)",
        "baseline_mean": base_mean,
        "per_author": per_author,
        "injection_site_check": site,
        "cosines_ok": cosines_ok,
        "flat_ok": flat_ok,
        "passed": passed,
    }
    write_result("stage2_steering", "steering", result)
    print(
        f"\nStage 2: {'PASS' if passed else 'FAIL'}  "
        f"(cosine≥0.99: {cosines_ok}, flat: {flat_ok}, site: {site['passed']})"
    )
    return result
