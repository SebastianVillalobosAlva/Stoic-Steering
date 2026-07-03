"""CLI: python -m stoic <command>

Pass A, Stage 0-2 checkpoints (all $0, local CPU):

    python -m stoic stage0     # deterministic decoding
    python -m stoic stage1     # base P(stoic) == 0.542  (load-bearing)
    python -m stoic stage2     # vector cosine >=0.99 + steered dilemmas flat
    python -m stoic all        # run 0,1,2 in one model load

Each command writes one JSON checkpoint under results/<stage>/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from stoic import config
from stoic.dilemmas import (
    eval_dilemmas,
    load_dilemmas,
    mean,
    paired_stats,
    deltas_by_stance,
    _logit,
)
from stoic.model import generate, load_model
from stoic.steering import extract_vector, load_pairs, load_reference_vector, steering


def _write(stage: str, name: str, payload: dict) -> Path:
    payload = {"timestamp": time.strftime("%Y%m%d_%H%M%S"), **payload}
    path = config.results_dir(stage) / f"{name}_{payload['timestamp']}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ↳ wrote {path.relative_to(config.PROJECT_ROOT)}")
    return path


# --- Stage 0: deterministic decoding -------------------------------------
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
    _write("stage0_determinism", "determinism", result)
    print(f"Stage 0: {'PASS' if identical else 'FAIL'}")
    return result


# --- Stage 1: base P(stoic) == 0.542 -------------------------------------
def stage1(model, tokenizer) -> dict:
    print("\n=== Stage 1: base P(stoic) == 0.542 (load-bearing) ===")
    dilemmas = load_dilemmas()
    baseline = eval_dilemmas(model, tokenizer, dilemmas)
    base_mean = mean(baseline)
    # Checkpoint: matches reference to the 3rd decimal.
    passed = round(base_mean, 3) == config.DILEMMA_BASELINE
    print(f"n_dilemmas: {len(dilemmas)}  (x2 label orders)")
    print(f"baseline mean P(stoic): {base_mean:.6f}  (target {config.DILEMMA_BASELINE})")
    result = {
        "stage": 1,
        "check": "base P(stoic) == 0.542 on v2 set",
        "n_dilemmas": len(dilemmas),
        "baseline_mean": base_mean,
        "target": config.DILEMMA_BASELINE,
        "reference_exact": 0.541601902275579,
        "passed": passed,
        "baseline_p_stoic": baseline,
    }
    _write("stage1_dilemma_baseline", "baseline", result)
    print(f"Stage 1: {'PASS' if passed else 'FAIL'}")
    return result, baseline


# --- Stage 2: vector fidelity + steered dilemmas flat --------------------
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

    for name, author in config.AUTHORS.items():
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
            f"  steered mean P(stoic) = {mean(steered):.4f}   "
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

    # Injection-site mechanism check on Epictetus L8.
    epi = config.AUTHORS["epictetus"]
    epi_vec = load_reference_vector(epi.vector_file, epi.layer)
    site = _injection_site_check(model, tokenizer, epi.layer, epi_vec, epi.coeff)
    print(
        f"\ninjection site L{epi.layer}: hs[L] unchanged={site['hidden_states[L]_unchanged']}, "
        f"hs[L+1] changed={site['hidden_states[L+1]_changed']}"
    )

    cosines_ok = all(a["cosine_to_frozen"] >= 0.99 for a in per_author.values())
    # "Flat" = every author's |ΔP| small (Exp 10 null). Reference ΔP were ~1e-3.
    flat_ok = all(abs(a["overall"]["mean_delta"]) < 0.02 for a in per_author.values())
    passed = cosines_ok and flat_ok and site["passed"]

    result = {
        "stage": 2,
        "check": "cosine>=0.99 vs frozen; injection bites at L+1; steered dilemmas flat (Exp 10 null)",
        "baseline_mean": base_mean,
        "per_author": per_author,
        "injection_site_check": site,
        "cosines_ok": cosines_ok,
        "flat_ok": flat_ok,
        "passed": passed,
    }
    _write("stage2_steering", "steering", result)
    print(
        f"\nStage 2: {'PASS' if passed else 'FAIL'}  "
        f"(cosine≥0.99: {cosines_ok}, flat: {flat_ok}, site: {site['passed']})"
    )
    return result


def main():
    parser = argparse.ArgumentParser(prog="stoic")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for c in ("stage0", "stage1", "stage2", "all"):
        sub.add_parser(c)
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.cmd == "stage0":
        stage0(model, tokenizer)
    elif args.cmd == "stage1":
        stage1(model, tokenizer)
    elif args.cmd == "stage2":
        stage2(model, tokenizer)
    elif args.cmd == "all":
        stage0(model, tokenizer)
        _, baseline = stage1(model, tokenizer)
        stage2(model, tokenizer, baseline=baseline)


if __name__ == "__main__":
    main()
