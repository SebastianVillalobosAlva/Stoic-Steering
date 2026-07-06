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
import os
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


# --- Stage 3: judge-scored content effect (Exp 9, COSTS $) ---------------
def _gemini_key() -> str:
    """Gemini key from env, else from a KEY=VALUE line in project-root .env."""
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(k):
            return os.environ[k]
    env = config.PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith(("GEMINI_API_KEY", "GOOGLE_API_KEY")) and "=" in line:
                return line.split("=", 1)[1].strip().strip("'\"")
    raise SystemExit(
        "No Gemini key found. Set GEMINI_API_KEY in the environment or in "
        f"{env} (GEMINI_API_KEY=...). Stage 3 calls the Gemini judge API ($)."
    )


def stage3(model, tokenizer, authors=None, n_seeds: int = 5, sampled: bool = False) -> dict:
    from stoic import judge

    mode = "matched SAMPLED (temp 0.6)" if sampled else "matched GREEDY"
    print(f"\n=== Stage 3: judge-scored content effect (Exp 9, Gemini judge) — {mode} ===")
    client, judge_model = judge.make_gemini_client(_gemini_key())
    authors = authors or list(config.AUTHORS)

    # Sampled mode: baselines are unsteered, so compute once per seed and share.
    baselines_by_seed = None
    if sampled:
        print(f"Generating shared unsteered baselines for {n_seeds} seeds ...")
        baselines_by_seed = {
            s: judge.generate_all_sampled(model, tokenizer, config.DEFAULT_PROMPTS, s)
            for s in range(n_seeds)
        }

    per_author, checks = {}, {}
    for name in authors:
        author = config.AUTHORS[name]
        vector = load_reference_vector(author.vector_file, author.layer)  # Exp 9 input
        if sampled:
            run = judge.seed_eval_sampled(
                model, tokenizer, client, judge_model,
                layer=author.layer, vector=vector, coeff=author.coeff,
                author=name, baselines_by_seed=baselines_by_seed,
            )
        else:
            run = judge.seed_eval(
                model, tokenizer, client, judge_model,
                layer=author.layer, vector=vector, coeff=author.coeff,
                author=name, n_seeds=n_seeds,
            )
        ref_mean, ref_std = config.EXP9_CONTENT[name]
        new_mean, new_std = run["content_mean"], run["content_std"]
        # Pattern check: positive, and ±1σ intervals overlap the reference.
        overlap = (new_mean + new_std) >= (ref_mean - ref_std) and (
            new_mean - new_std
        ) <= (ref_mean + ref_std)
        checks[name] = {
            "content_mean": new_mean,
            "content_std": new_std,
            "reference_mean": ref_mean,
            "reference_std": ref_std,
            "positive": new_mean > 0,
            "overlaps_reference": overlap,
        }
        run["reference"] = {"content_mean": ref_mean, "content_std": ref_std}
        per_author[name] = run
        print(
            f"  [{name}] content {new_mean:+.3f} ± {new_std:.3f}  "
            f"(Exp 9: {ref_mean:+.3f} ± {ref_std:.3f}, "
            f"overlap={overlap}, positive={new_mean > 0})"
        )

    all_positive = all(c["positive"] for c in checks.values())
    all_overlap = all(c["overlaps_reference"] for c in checks.values())
    passed = all_positive and all_overlap
    result = {
        "stage": 3,
        "decoding_mode": "sampled_matched" if sampled else "greedy_matched",
        "check": "content effect positive + ±1σ overlaps Exp 9 (Marcus +0.408 / Seneca +0.583 / Epictetus +0.767)",
        "judge_model": judge_model,
        "n_seeds": n_seeds,
        "per_author": per_author,
        "checks": checks,
        "all_positive": all_positive,
        "all_overlap": all_overlap,
        "passed": passed,
    }
    name_suffix = "content_sampled" if sampled else "content_greedy"
    _write("stage3_content_judge", name_suffix, result)
    print(f"\nStage 3: {'PASS' if passed else 'REVIEW'}  "
          f"(all positive: {all_positive}, all overlap Exp 9: {all_overlap})")
    return result


# --- Stage 4: LoRA decision shift (Exp 11, judge-free, $0) ----------------
def stage4(model, tokenizer) -> dict:
    """Merge each frozen clean adapter onto a fresh base and run the dilemma
    eval. Checkpoint: base integrity (0.542 -> 0.542, drift 0) and Seneca's
    shift positive in BOTH stance buckets with overall t >= ~2 (Exp 11)."""
    import gc

    from stoic import lora

    print("\n=== Stage 4: LoRA dilemma eval (judge-free, frozen adapters) ===")
    dilemmas = load_dilemmas()

    print("Base integrity (start): baseline on unmodified base ...")
    baseline = eval_dilemmas(model, tokenizer, dilemmas)
    base_mean_start = mean(baseline)
    print(f"  baseline mean P(stoic) = {base_mean_start:.6f}")

    per_author = {}
    for name, author in config.AUTHORS.items():
        print(f"\n[{name}] {author.adapter_dir.name}")
        merged = lora.merge_adapter(author.adapter_dir)
        try:
            steered = eval_dilemmas(merged, tokenizer, dilemmas)
        finally:
            del merged
            gc.collect()

        deltas = {i: steered[i] - baseline[i] for i in steered}
        deltas_lo = {i: _logit(steered[i]) - _logit(baseline[i]) for i in steered}
        overall = paired_stats(list(deltas.values()))
        overall_lo = paired_stats(list(deltas_lo.values()))
        stance = deltas_by_stance(dilemmas, deltas)
        print(f"  steered mean {mean(steered):.4f}  ΔP {overall['mean_delta']:+.4f} "
              f"(t={overall['t_stat']:.2f})  Δlo {overall_lo['mean_delta']:+.4f} "
              f"(t={overall_lo['t_stat']:.2f})")
        for k, v in sorted(stance.items()):
            print(f"    {k:10s}: ΔP {v['mean_delta']:+.4f}  t {v['t_stat']:+.2f}  (n={v['n']})")
        per_author[name] = {
            "adapter": str(author.adapter_dir.name),
            "steered_mean": mean(steered),
            "steered_p_stoic": steered,
            "overall": overall,
            "overall_logodds": overall_lo,
            "by_stance": stance,
        }

    print("\nBase integrity (end): baseline again on the same base model ...")
    baseline_end = eval_dilemmas(model, tokenizer, dilemmas)
    drift = max(abs(baseline_end[i] - baseline[i]) for i in baseline)
    base_mean_end = mean(baseline_end)
    print(f"  baseline mean {base_mean_end:.6f}  max per-item drift {drift:.2e}")

    integrity = (
        round(base_mean_start, 3) == config.DILEMMA_BASELINE
        and round(base_mean_end, 3) == config.DILEMMA_BASELINE
        and drift == 0.0
    )
    sen = per_author["seneca"]
    sen_both_positive = all(v["mean_delta"] > 0 for v in sen["by_stance"].values())
    sen_t_ok = max(sen["overall"]["t_stat"], sen["overall_logodds"]["t_stat"]) >= 2.0
    passed = integrity and sen_both_positive and sen_t_ok

    result = {
        "stage": 4,
        "check": "base integrity 0.542->0.542 drift 0; Seneca ΔP>0 in BOTH stance buckets with overall t>=2 (Exp 11 pattern)",
        "reference": "data/reference/dilemmas/v2/lora/dilemma_eval_20260701_140942.json "
                     "(marcus ΔP +0.0307 accepting-only; seneca +0.0606 both buckets; epictetus null)",
        "baseline_mean_start": base_mean_start,
        "baseline_mean_end": base_mean_end,
        "max_baseline_drift": drift,
        "base_integrity": integrity,
        "per_author": per_author,
        "seneca_both_buckets_positive": sen_both_positive,
        "seneca_t_ok": sen_t_ok,
        "passed": passed,
        "baseline_p_stoic": baseline,
    }
    _write("stage4_lora_dilemmas", "lora_dilemmas", result)
    print(f"\nStage 4: {'PASS' if passed else 'FAIL'}  (integrity: {integrity}, "
          f"seneca both buckets +: {sen_both_positive}, seneca t>=2: {sen_t_ok})")
    return result


# --- Style validation: does "CAA moves register" survive matched decoding? ---
def style_check(model, tokenizer, n_seeds: int = 5) -> dict:
    """Re-test the Exp 3b style/register claim under matched decoding.

    Greedy arm: re-scores the SAVED Stage 3 greedy generations (no generation
    cost). Sampled arm: regenerates the exact Stage 3 sampled texts (seeded)
    and judges once per seed. Pre-registered decision rule: style survives if
    the seed-averaged delta stays clearly positive by ~2σ (mean > 2*std).
    """
    import glob
    import statistics as st

    from stoic import judge

    client, judge_model = judge.make_gemini_client(_gemini_key())
    greedy_file = sorted(glob.glob(str(config.results_dir("stage3_content_judge") / "content_2*.json")))[0]
    saved = json.load(open(greedy_file))

    def seed_stats(vals):
        m = st.mean(vals)
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        return m, sd

    # Sampled baselines are unsteered → generate once per seed, share across authors.
    print(f"Generating shared sampled baselines for {n_seeds} seeds ...")
    baselines_by_seed = {
        s: judge.generate_all_sampled(model, tokenizer, config.DEFAULT_PROMPTS, s)
        for s in range(n_seeds)
    }

    per_author = {}
    for name, author in config.AUTHORS.items():
        ref3b = config.EXP3B_STYLE[name]
        entry = {"layer": author.layer, "coeff": author.coeff, "exp3b_style": ref3b}

        # -- matched greedy: pure re-scoring of saved texts --
        run = saved["per_author"][name]
        print(f"\n[{name}] matched GREEDY re-score (saved Stage 3 texts)")
        g = judge.judge_fixed_texts(
            client, judge_model, config.DEFAULT_PROMPTS,
            run["steered_outputs"], run["baseline_outputs"], n_seeds=n_seeds,
        )
        styles = [d["stylistic_authenticity"] for d in g["per_seed_deltas"]]
        m, sd = seed_stats(styles)
        entry["greedy"] = {
            "style_mean": m, "style_std": sd, "survives_2sigma": m > 2 * sd,
            "n_identical_pairs": g["n_identical"], "per_seed": g["per_seed_deltas"],
        }
        print(f"  greedy style = {m:+.3f} ± {sd:.3f}  (Exp 3b: +{ref3b:.2f}; "
              f"{g['n_identical']}/12 pairs byte-identical)")

        # -- matched sampled: regenerate seeded texts, judge once per seed --
        print(f"[{name}] matched SAMPLED (regenerating seeds 0-{n_seeds - 1})")
        vector = load_reference_vector(author.vector_file, author.layer)
        seed_deltas = []
        for s in range(n_seeds):
            steer = judge.generate_all_sampled(
                model, tokenizer, config.DEFAULT_PROMPTS, s,
                steer=(author.layer, vector, author.coeff),
            )
            er = judge.evaluate_steering(
                client, judge_model, config.DEFAULT_PROMPTS, steer, baselines_by_seed[s]
            )
            seed_deltas.append(er["avg_deltas"])
            print(f"  seed {s}: style={er['avg_deltas']['stylistic_authenticity']:+.3f}")
        styles_s = [d["stylistic_authenticity"] for d in seed_deltas]
        ms, sds = seed_stats(styles_s)
        entry["sampled"] = {
            "style_mean": ms, "style_std": sds, "survives_2sigma": ms > 2 * sds,
            "per_seed": seed_deltas,
        }
        print(f"  sampled style = {ms:+.3f} ± {sds:.3f}  (Exp 3b: +{ref3b:.2f})")
        per_author[name] = entry

    survives = {
        n: e["greedy"]["survives_2sigma"] or e["sampled"]["survives_2sigma"]
        for n, e in per_author.items()
    }
    result = {
        "check": "Exp 3b style/register claim under matched decoding; survives if seed-avg style delta > 2*std",
        "note": "Canonical clean configs (M L26/S L4/E L8, c=0.11); Exp 3b ran superseded all-L8 configs with asymmetric decoding — historical reference, not exact-config comparison.",
        "judge_model": judge_model,
        "n_seeds": n_seeds,
        "greedy_source": Path(greedy_file).name,
        "per_author": per_author,
        "survives": survives,
    }
    _write("style_validation", "style", result)
    print("\n=== Style validation ===")
    for n, e in per_author.items():
        print(f"  {n:10s} greedy {e['greedy']['style_mean']:+.3f}±{e['greedy']['style_std']:.3f}  "
              f"sampled {e['sampled']['style_mean']:+.3f}±{e['sampled']['style_std']:.3f}  "
              f"(Exp 3b +{e['exp3b_style']:.2f})  -> "
              f"{'SURVIVES' if survives[n] else 'collapses'}")
    return result


def main():
    parser = argparse.ArgumentParser(prog="stoic")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for c in ("stage0", "stage1", "stage2", "all"):
        sub.add_parser(c)
    p3 = sub.add_parser("stage3")
    p3.add_argument("--author", choices=list(config.AUTHORS), default=None,
                    help="run one author only (default: all three)")
    p3.add_argument("--seeds", type=int, default=5)
    p3.add_argument("--sampled", action="store_true",
                    help="matched-SAMPLED comparison (both baseline+steered sampled, temp 0.6)")
    ps = sub.add_parser("style")
    ps.add_argument("--seeds", type=int, default=5)
    sub.add_parser("stage4")
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.cmd == "stage0":
        stage0(model, tokenizer)
    elif args.cmd == "stage1":
        stage1(model, tokenizer)
    elif args.cmd == "stage2":
        stage2(model, tokenizer)
    elif args.cmd == "stage3":
        authors = [args.author] if args.author else None
        stage3(model, tokenizer, authors=authors, n_seeds=args.seeds, sampled=args.sampled)
    elif args.cmd == "style":
        style_check(model, tokenizer, n_seeds=args.seeds)
    elif args.cmd == "stage4":
        stage4(model, tokenizer)
    elif args.cmd == "all":
        stage0(model, tokenizer)
        _, baseline = stage1(model, tokenizer)
        stage2(model, tokenizer, baseline=baseline)


if __name__ == "__main__":
    main()
