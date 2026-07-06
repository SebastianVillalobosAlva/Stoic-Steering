# Superseded reference artifacts

**Do not use these as verification targets.** They are preserved as the
"before" side of a documented measurement artifact — provenance, not ground
truth.

## `seed_gemini_clean_{author}.json` — the Exp 9 primary result files

Byte-identical copies of the Experiment 9 seed-eval outputs from the legacy
repo (`stoic-llm-legacy/results/sweeps/`, generated 2026-06-11; Gemini judge,
clean pairs, n_seeds=5, vary=judge). They contain the original headline CAA
content effects:

| Author | Config | Reported content effect |
|---|---|---|
| Marcus Aurelius | L26, c=0.11 | +0.408 ± 0.136 |
| Seneca | L4, c=0.11 | +0.583 ± 0.121 |
| Epictetus | L8, c=0.11 | +0.767 ± 0.076 |

**Why superseded:** these numbers are a decoding-asymmetry artifact. The
legacy steered-generation path silently sampled (temp 0.6, top-p 0.9) and
truncated to ~13 tokens while the baseline was greedy at 100 tokens; the
judge scored the decoding difference, not the steering. Re-measured under
matched decoding (same frozen vectors at cosine 1.0000, same judge), the
content effect is null in both the greedy and sampled regimes, and the Exp 3b
style effect collapses the same way.

- Root cause, line-quoted: `stoic-llm-legacy/LEGACY.md` (repo tagged `v0-exp11`)
- Re-measurement record: `results/README.md` + `results/stage3_content_judge/`
  and `results/style_validation/` in this repo

Everything else under `data/reference/` remains a live verification target;
only the contents of this folder are superseded.
