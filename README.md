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
  forced-choice instrument where CAA was flat, weight-level adaptation moved the
  choice (<FILL: e.g. Seneca delta-log-odds / t>). The circuit topology difference
  found via interpretability (<FILL: which experiments>) *predicted* this split
  before it was measured behaviorally.

- **What LoRA installs is not (yet) uniform Stoic reasoning.** Effects are
  structured but heterogeneous: <FILL: Marcus = passivity prior; Seneca =
  heavy-tailed; Epictetus = null>. A possible idiom/style confound in the
  decision instrument is under investigation. These are stated openly as open
  questions, not smoothed over.

---

## Method

Three depths of effect are measured separately:

- **Style / register** -- LLM-judge scoring of prose (does it sound Stoic?)
- **Content / reasoning** -- LLM-judge scoring of reasoning in prose
- **Decision / choice** -- judge-free forced-choice probe over calibrated
  dilemmas (does the model *pick* the Stoic option?)

Two interventions are compared: **CAA** (runtime activation steering,
reversible) and **LoRA** (fine-tuned adapter weights, permanent). Both are
analyzed with **<FILL: ModelLens or your toolkit name>**, an
architecture-agnostic interpretability toolkit, to compare the circuit
topology each method uses to produce the same behavioral outcome.

Philosophers studied: Marcus Aurelius, Seneca, Epictetus -- three Stoic
traditions, done rigorously rather than many traditions done shallowly.

---

## Repo structure

```
stoic/
  config.py     # paths + config (per-author layer/coeff, decoding)
  model.py      # model loading + generation
  corpus.py     # text download, slicing, chunking, filtering
  pairs.py      # contrastive pair generation
  steering.py   # CAA vector extraction + steering
  lora.py       # LoRA prep, train, merge
  judge.py      # LLM-as-judge scoring
  dilemmas.py   # judge-free forced-choice harness
cli.py          # command-line entry point
data/
  reference/    # frozen artifacts (pairs, dilemma sets, vectors, adapters)
  generated/    # pipeline output
results/        # experiment results (JSON)
```

## Quickstart

```bash
# install / env setup
<FILL: e.g. pip install -e .  or  uv sync>

# extract CAA steering vectors
python -m stoic extract <FILL: args>

# run the forced-choice decision eval
python -m stoic dilemma <FILL: args>
```

<FILL: any setup notes -- model access (Llama-3.2-3B gated on HF),
API keys for judge/pair generation, Colab for LoRA training>

---

## Status

**Done:** <FILL: e.g. CAA + LoRA steering across three philosophers, content
validation with independent judge, decision-level forced-choice results, circuit
topology comparison.>

**In progress / next:**
- <FILL: Seneca idiom-vs-topic discrimination (2x2 dilemma design)>
- <FILL: Epictetus full-corpus retrain (corpus-size hypothesis)>
- <FILL: safety / robustness evaluation (jailbreak + temperature stability)>

---

## Notes

<FILL: optional -- related repos (e.g. ModelLens), context, or a one-line
pointer to the research writeup / SOP if you want one. Keep it light.>
