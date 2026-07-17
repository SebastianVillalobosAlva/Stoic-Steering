"""Stage 3 + style validation — the Gemini-judged effects under matched decoding.

Both cost $ (judge API) and require GEMINI_API_KEY. These are the stages where
the original measurement artifact lived; every generation here routes through
the one canonical `generate()` so both sides of every comparison decode
identically.
"""

from __future__ import annotations

import json
from pathlib import Path

from stoic import config
from stoic.results_io import write_result
from stoic.secrets import gemini_key
from stoic.steering import load_reference_vector


def stage3(model, tokenizer, authors=None, n_seeds: int = 5, sampled: bool = False) -> dict:
    from stoic import judge

    mode = "matched SAMPLED (temp 0.6)" if sampled else "matched GREEDY"
    print(f"\n=== Stage 3: judge-scored content effect (Exp 9, Gemini judge) — {mode} ===")
    client, judge_model = judge.make_gemini_client(gemini_key())
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
    write_result("stage3_content_judge", name_suffix, result)
    print(f"\nStage 3: {'PASS' if passed else 'REVIEW'}  "
          f"(all positive: {all_positive}, all overlap Exp 9: {all_overlap})")
    return result


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

    client, judge_model = judge.make_gemini_client(gemini_key())
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
    write_result("style_validation", "style", result)
    print("\n=== Style validation ===")
    for n, e in per_author.items():
        print(f"  {n:10s} greedy {e['greedy']['style_mean']:+.3f}±{e['greedy']['style_std']:.3f}  "
              f"sampled {e['sampled']['style_mean']:+.3f}±{e['sampled']['style_std']:.3f}  "
              f"(Exp 3b +{e['exp3b_style']:.2f})  -> "
              f"{'SURVIVES' if survives[n] else 'collapses'}")
    return result
