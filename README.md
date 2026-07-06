# Stoic Steering

Steering Stoic philosophical reasoning into Llama-3.2-3B via activation
addition (CAA) and low-rank weight adaptation (LoRA), with mechanistic
interpretability to check what actually changes inside the model.

**Core finding:** the original evals overstated what activation steering does.
Under fair (matched-decoding) measurement, CAA at the canonical coefficient
moves *nothing* measurable — not style, not judge-scored content, not
decisions. Weight adaptation (LoRA) genuinely moves all three levels,
including the judge-free decision instrument. The clean rebuild caught the
original CAA style/content effects being a single measurement artifact.

---

## Key findings

- **CAA at the canonical coefficient does nothing measurable — at any of the
  three levels.** Decisions were always flat (forced-choice test, every
  coefficient up to 1.5, logit-measured and artifact-immune). Style and
  content *appeared* to move, but under matched decoding both collapse to
  zero (style: +1.0…+1.6 reported → −0.15…+0.05 greedy / ~0.0 sampled, all
  n.s.). At coeff 0.11 the steered greedy output is nearly byte-identical to
  baseline; whether stronger coefficients produce genuine Stoic register
  under fair measurement is an open question, not a claimed result.

- **The headline CAA "content effect" was a measurement artifact — caught by
  the clean rebuild.** The original eval silently *sampled* steered generations
  (temp 0.6, ~13 tokens) while baselines were *greedy* (100 tokens); the judge
  was scoring the decoding difference, not the steering. Re-measured with
  identical decoding on both sides (same frozen vectors, same judge), the
  content effect is null for all three philosophers — e.g. Epictetus L8:
  +0.767 ± 0.076 reported -> −0.12 ± 0.08 matched-greedy / −0.19 ± 0.33
  matched-sampled. One canonical `generate()` in the rebuild makes the original
  bug unwritable; the root cause is documented line-by-line in the legacy
  repo's LEGACY.md.

- **LoRA reaches the decision layer where CAA does not.** On the identical
  forced-choice instrument where CAA was flat, weight-level adaptation moved
  the choice: Seneca ΔP(stoic) +0.061, Δlog-odds +0.308 (t = 2.4–2.6),
  positive in *both* stance buckets; Marcus ΔP +0.031 (t = 2.00). Reproduced
  exactly (to 4 decimals) in the clean rebuild, judge-free, from the frozen
  adapters. The circuit topology difference found via interpretability
  (Exp 5/6 bridge analysis, ModelLens) *predicted* this split before it was
  measured behaviorally.

- **What LoRA installs is not (yet) uniform Stoic reasoning.** Effects are
  structured but heterogeneous: Marcus is a broad *passivity prior* — it moves
  only the "accepting" dilemmas (+0.065, t = 2.99) and is flat on "active"
  ones; Seneca is a *heavy-tailed magnitude effect* — both buckets positive,
  t = 2.4–2.6 overall, but the per-item sign test is n.s. (25/40); Epictetus
  is a *null* (ΔP +0.000; smallest corpus — 123 chunks, Enchiridion only). A
  possible Senecan-idiom lexical-echo confound in the decision instrument is
  under investigation. These are stated openly as open questions, not
  smoothed over.

---

## Method

Three depths of effect are measured separately:

- **Style / register** -- LLM-judge scoring of prose (does it sound Stoic?)
- **Content / reasoning** -- LLM-judge scoring of reasoning in prose
- **Decision / choice** -- judge-free forced-choice probe over calibrated
  dilemmas (does the model *pick* the Stoic option?)

Two interventions are compared: **CAA** (runtime activation steering,
reversible) and **LoRA** (fine-tuned adapter weights, permanent). Both are
analyzed with **ModelLens**, an architecture-agnostic interpretability
toolkit (companion project), to compare the circuit topology each method
uses to produce the same behavioral outcome.

Philosophers studied: Marcus Aurelius, Seneca, Epictetus -- three Stoic
traditions, done rigorously rather than many traditions done shallowly.

---

## Repo structure

```
stoic/
  config.py     # paths + canonical config (per-author layer/coeff, decoding)
  model.py      # model loading + the ONE generate() (decoding lives here only)
  steering.py   # CAA: extract_vector(pairs, layer) + steering() context manager
  dilemmas.py   # judge-free forced-choice harness (the 0.542 ruler)
  judge.py      # LLM-as-judge scoring (Gemini) + seed evals
  lora.py       # LoRA merge (fresh base per adapter) + prep/train for Colab
  __main__.py   # CLI: python -m stoic <stage>
  corpus.py     # (Pass B, planned) text download, slicing, chunking
  pairs.py      # (Pass B, planned) contrastive pair generation
data/
  reference/    # FROZEN artifacts (pairs, dilemma sets, vectors) — read-only
  generated/    # pipeline output (gitignored)
models/         # frozen clean LoRA adapters (not in git)
results/        # one JSON per stage checkpoint + results/README.md record
```

## Quickstart

```bash
# install (core = stages 0-2; extras: [judge] for stage 3, [lora] for stage 4)
pip install -e ".[all]"

# Pass A checkpoints
python -m stoic all       # stages 0-2: determinism, 0.542 baseline, vectors + CAA null
python -m stoic stage3    # judge-scored content effect (needs GEMINI_API_KEY, ~$1-2)
python -m stoic stage3 --sampled   # matched-sampled variant
python -m stoic style     # style/register validation under matched decoding
python -m stoic stage4    # LoRA decision shift (judge-free, $0)
```

Setup notes: `meta-llama/Llama-3.2-3B` is gated on Hugging Face (accept the
license, then `huggingface-cli login`). Stage 3 / style need `GEMINI_API_KEY`
in the environment or a project-root `.env`. Everything runs on local CPU
(~16 GB RAM); a Colab T4 is only needed to *retrain* adapters in Pass B.

---

## Status

**Done — Pass A (clean rebuild verified against frozen artifacts):**
- Everything logit-measured reproduces *exactly*: dilemma baseline 0.541602,
  CAA decision null (ΔP to 4 decimals), LoRA decision shifts (all authors,
  all stance buckets, to 4 decimals).
- Everything judge-scored on the CAA side exposed as a single
  decoding-asymmetry artifact: content (+0.41…+0.77 reported → null) and
  style (+1.0…+1.6 reported → null) both collapse under matched decoding.
  Root cause documented line-by-line in the legacy repo's `LEGACY.md`
  (tagged `v0-exp11`); full record in [`results/README.md`](results/README.md).

**In progress / next:**
- Pass B: regenerate corpus + contrastive pairs from raw Gutenberg text and
  re-run stages 2–4 on fresh data (pipeline-robustness test).
- Matched-decoding coefficient sweep: does *any* coefficient produce genuine
  Stoic register under fair measurement? (0.11 is too weak to alter greedy
  output; ≥1.0 visibly changes text but drifts off-target.)
- Seneca idiom-vs-topic discrimination (2×2 dilemma design) — tests the
  lexical-echo confound behind the strongest LoRA decision effect.
- Epictetus full-corpus retrain (Enchiridion + Discourses) — corpus-size
  hypothesis for the null.
- Expand the dilemma set 40 → 80+ items uniformly.

---

## Notes

Companion project: **ModelLens**, the architecture-agnostic interpretability
toolkit used for the circuit-topology comparison (Exp 5/6). The original
codebase that produced Experiments 1–11 is preserved unmodified as
`stoic-llm-legacy` (tagged `v0-exp11`); its `LEGACY.md` maps every experiment
to its scripts and documents the decoding-artifact root cause.
