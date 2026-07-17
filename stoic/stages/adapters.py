"""Stage 4 — LoRA decision shift (Exp 11, judge-free, $0).

Each frozen clean adapter is merged onto a FRESH base (no stacking possible)
and run through the dilemma harness; base integrity (0.542 -> 0.542, drift 0)
is asserted before and after.
"""

from __future__ import annotations

from stoic import config
from stoic.dilemmas import (
    deltas_by_stance,
    eval_dilemmas,
    load_dilemmas,
    mean,
    paired_stats,
    sign_test,
    _logit,
)
from stoic.results_io import write_result


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
        signs = sign_test(deltas)
        print(f"  steered mean {mean(steered):.4f}  ΔP {overall['mean_delta']:+.4f} "
              f"(t={overall['t_stat']:.2f})  Δlo {overall_lo['mean_delta']:+.4f} "
              f"(t={overall_lo['t_stat']:.2f})")
        print(f"    sign test: +{signs['pos']}/-{signs['neg']} "
              f"(ties {signs['ties']})  p={signs['p_two_sided']:.4f}")
        for k, v in sorted(stance.items()):
            print(f"    {k:10s}: ΔP {v['mean_delta']:+.4f}  t {v['t_stat']:+.2f}  (n={v['n']})")
        per_author[name] = {
            "adapter": str(author.adapter_dir.name),
            "steered_mean": mean(steered),
            "steered_p_stoic": steered,
            "overall": overall,
            "overall_logodds": overall_lo,
            "by_stance": stance,
            "sign_test": signs,
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
    write_result("stage4_lora_dilemmas", "lora_dilemmas", result)
    print(f"\nStage 4: {'PASS' if passed else 'FAIL'}  (integrity: {integrity}, "
          f"seneca both buckets +: {sen_both_positive}, seneca t>=2: {sen_t_ok})")
    return result
