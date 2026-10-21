# Weight space vs Activation space - Do they install the same behavior?

Comparing activation addition (CAA) against low-rank weight adaptation (LoRA) on the same behavioral axis. In a 3B model, CAA is a no-op under matched decoding. LoRA moves forced-choice decisions, but a pre-registered powered test shows the shift is lexical echo, not reasoning.

**Core finding:** Under matched decoding, CAA (coefficient 0.11) produces no measurable change in style, content, or forced-choice decisions. An earlier positive result was a decoding artifact (docs/measurement-artifact.md). LoRA is the only intervention that moves the judge-free decision instrument (it also shifts judge-scored style and content in a single eval). A pre-registered test at n=100 (docs/pre-registration-v4.md) shows that the decision movement tracks Seneca's phrasing rather than the dilemma's content: lexical echo, not reasoning. A pilot circuit analysis shows the same split: CAA leaves the circuit essentially untouched; LoRA perturbs it.

> **⚠ Superseded in part (v4, 2026-07-31) — awaiting prose revision.** A
> pre-registered, powered eval (100 items, 25/cell) shows the LoRA decision
> shift is **lexical echo, not reasoning**: Seneca moves idiom-worded items
> (+0.099 log-odds) and moves plain-worded items the other way (−0.083), and the
> off-topic/plain reasoning cell is bounded-null (95% CI excludes an idiom-sized
> effect). The v2 numbers below stand as a v2 result; their *interpretation* has
> narrowed to "weight adaptation moves the choice through register, not
> reasoning." The claims below are left intact as receipts pending rewrite. Full
> map + numbers: [docs/claim-revision-v4.md](docs/claim-revision-v4.md),
> `results/v4_behavioral/`.

![Three-depths dissociation — CAA is flat at style, content, and decision; LoRA moves all three (Epictetus decision null)](results/figures/fig_three_depths.png)

*v2 decision result. The v4 powered test shows this movement is lexical echo, not reasoning; see docs/claim-revision-v4.md.*

------------------------------------------------------------------------

## Key findings

-   **CAA — no measurable movement.** At coefficient 0.11, decisions, style, and content show no measurable effect. A coefficient sweep at the decision level has not been done, so whether a stronger coefficient would produce real Stoic register is still unclear.

-   **CAA "content effect" — a measurement artifact.** An early measurement reported a large positive effect. Steered text was sampled and truncated to ~13 tokens while baselines were greedy at 100, so the judge was scoring the decoding difference instead of the steering. Under identical decoding, the content effect is null for all three philosophers. Full mechanism and numbers in [docs/measurement-artifact.md](docs/measurement-artifact.md).

-   **LoRA reaches decisions.** *(v2 result, n=40 — [narrowed by v4](docs/claim-revision-v4.md): on 100 items Seneca's overall effect is t=0.21, and the movement is register-driven, not reasoning. Table kept as the v2 receipt.)* On the forced-choice instrument, where CAA is flat, weight-level adaptation moved the choice — measured judge-free from the frozen adapters:

    | Author | ΔP(stoic) | Δlog-odds | t       | Stance buckets     |
    |--------|-----------|-----------|---------|--------------------|
    | Seneca | +0.061    | +0.308    | 2.4–2.6 | positive in *both* |
    | Marcus | +0.031    | +0.161    | 2.0–2.2 | accepting only     |

    The circuit analysis (Exp 12 via ModelLens) shows the same split. CAA leaves the stoic-content circuit essentially untouched. LoRA's circuit perturbation is ordered Seneca > Marcus > Epictetus ≈ 0 — the same ordering as the decision shift. It is robust at the median, and acts on content discrimination in an item-dependent way rather than as a directional push. Full analysis in [results/README.md](results/README.md).

    **Caveat:** the two methods also train on different objectives (CAA is contrastive, LoRA is continued pretraining), so method and objective are confounded. A non-philosophical control adapter separated them: it moved −0.036 (n.s.) and matched Marcus's one significant cell, so that cell is generic fine-tuning drift, not an axis effect.

![LoRA decision shift by author and stance bucket — Seneca moves both, Marcus accepting-only, Epictetus null](results/figures/fig_lora_decision_shift.png)

*v2 decision result. The v4 powered test shows this movement is lexical echo, not reasoning; see docs/claim-revision-v4.md.*

-   **LoRA does not install uniform Stoic reasoning.** The effects differ by philosopher. Marcus is a broad *passivity prior*: it moves only the "accepting" dilemmas and is flat on the "active" ones. Seneca moves the choice on average, but that average is carried by a handful of items shifting a lot rather than a steady push across all 40 — the per-item sign test is not significant (25 of 40 positive, p = 0.15). Epictetus shows no decision effect at all, but it also has by far the smallest corpus (123 chunks, the Enchiridion only) and a terse, aphoristic style unlike Seneca's letters or Marcus's reflections. A possible Senecan-idiom lexical-echo confound in the decision instrument is also open. *(v4 update: this confound is no longer open — a pre-registered powered test **confirmed it as the dominant effect**. See [docs/claim-revision-v4.md](docs/claim-revision-v4.md).)*

![Seneca vs Marcus per-item circuit node shift, with medians and tie flags](results/figures/fig_exp12c_node_shift.png)

![Seneca per-item Δ\|c\| — diverging, mean vs median, two flattening outliers flagged](results/figures/fig_exp12c_delta_abs_c.png)

------------------------------------------------------------------------

## Method

Three depths of effect are measured separately:

-   **Style / register** -- LLM-judge scoring of prose (does it sound Stoic?)
-   **Content / reasoning** -- LLM-judge scoring of reasoning in prose
-   **Decision / choice** -- judge-free forced-choice probe over calibrated dilemmas (does the model *pick* the Stoic option?)

Two methods are compared: **CAA** and **LoRA**. Both are analyzed with **ModelLens**, an architecture-agnostic interpretability toolkit (companion project) to compare the circuit topology

Corpora: Marcus Aurelius, Seneca, Epictetus. Three authors used to build contrast pairs along one behavioral axis. Every result here is in behavior (one value dimension). Sycophancy is the planned second axis, and it is what tests whether the locus result holds outside this one.

------------------------------------------------------------------------

## Repo structure

```         
stoic/
  config.py     # paths + canonical config (per-author layer/coeff, decoding)
  model.py      # model loading + the ONE generate() (decoding lives here only)
  steering.py   # CAA: extract_vector(pairs, layer) + steering() context manager
  dilemmas.py   # judge-free forced-choice harness (the 0.542 ruler) + stats
  judge.py      # LLM-as-judge scoring (Gemini) + seed evals
  lora.py       # LoRA merge (fresh base per adapter) + prep/train for Colab
  corpus.py     # Pass B: Gutenberg download, license-strip, slice, chunk
  pairs.py      # Pass B: contrastive pair generation (Claude API)
  results_io.py # checkpoint JSON writing (always under results/<stage>/)
  secrets.py    # API-key lookup (env, then .env) — stages fail fast if missing
  stages/       # stage orchestration: verify.py (0-2), content.py (3+style),
                #   adapters.py (4), passb.py (corpus/pairs)
  __main__.py   # thin CLI: python -m stoic <stage> (parse + dispatch only)
tests/          # CPU-only unit tests (no model download): hook hygiene,
                #   canonical decoding, dilemma math, stats vs published
                #   numbers, reference-wall tripwire, fixture integrity
data/
  reference/    # FROZEN artifacts (pairs, dilemma sets, vectors) — read-only
  generated/    # pipeline output (gitignored)
  MANIFEST.sha256  # checksums of the untracked frozen binaries
models/         # frozen clean LoRA adapters (not in git)
results/        # one JSON per stage checkpoint + results/README.md record
```

## Quickstart

``` bash
# install (core = stages 0-2; extras: [judge] for stage 3, [lora] for stage 4)
pip install -e ".[all]"

# fetch the frozen binaries (steering vectors + LoRA adapters) — needed for
# Stages 2 and 4 only; downloads ~26 MB and verifies against data/MANIFEST.sha256
python scripts/fetch_artifacts.py

# Pass A checkpoints
python -m stoic all       # stages 0-2: determinism, 0.542 baseline, vectors + CAA null
python -m stoic stage3    # judge-scored content effect (needs GEMINI_API_KEY, ~$1-2)
python -m stoic stage3 --sampled   # matched-sampled variant
python -m stoic style     # style/register validation under matched decoding
python -m stoic stage4    # LoRA decision shift (judge-free, $0)

# Pass B — regenerate the corpus from source (self-contained)
python -m stoic corpus    # download + slice + chunk into data/generated/, verify counts
python -m stoic pairs     # regenerate contrastive pairs (needs ANTHROPIC_API_KEY, $)

# Unit tests (CPU-only, seconds, no model download)
pip install -e ".[dev]" && pytest
```

Setup notes: `meta-llama/Llama-3.2-3B` is gated on Hugging Face (accept the license, then `huggingface-cli login`). The frozen steering vectors and LoRA adapters are hosted separately (too large for git) at [`seb-vil/llama-3.2-3b-stoic-steering`](https://huggingface.co/seb-vil/llama-3.2-3b-stoic-steering); `scripts/fetch_artifacts.py` downloads them into place and verifies against `data/MANIFEST.sha256`. Stage 3 / style need `GEMINI_API_KEY` and Pass-B pairs need `ANTHROPIC_API_KEY` (env or a project-root `.env`); `corpus` needs neither. Everything runs on local CPU (\~16 GB RAM); a Colab T4 is only needed to *retrain* adapters.

------------------------------------------------------------------------

## Status

**Verified:**

-   Dilemma baseline 0.541602, CAA decision null (ΔP to 4 decimals), LoRA decision shifts (all authors, all stance buckets, to 4 decimals).
-   The judge-scored CAA effects are a single decoding-asymmetry artifact: content (+0.41…+0.77 → null) and style (+1.0…+1.6 → null) both collapse under matched decoding. Mechanism + numbers: [docs/measurement-artifact.md](docs/measurement-artifact.md) - full record in [results/README.md](results/README.md).
-   Corpus acquisition is self-contained: `python -m stoic corpus` re-downloads and re-chunks the source texts, reproducing the frozen chunk counts exactly (437 / 540 / 123; the stage compares counts, not byte-level chunk content).

Done: axis-agnostic refactor (verified byte-identical), ModelLens import fix + regression tests (33 passed, 1 skipped), dilemmas_v3 build + calibration, v3 eval (underpowered), v4 powered eval (headline).

Next:
1. End-to-end exp12 re-run through the fixed ModelLens import
2. Stance-balanced circuit sweep on v4 items
3. Sycophancy axis
4. Figures: Figure 1 (CAA vs LoRA circuit, side by side), then Figures 2–3
5. Pass B: rebuild the contrastive pairs and re-run stages 2–4 on fresh data (built, not run; either outcome is reportable)
6. Write-up

Stability sweep: per the Tier 2 gate set in the build plan before the v4 eval, the philosophy axis showed lexical echo, so the sweep runs on the sycophancy axis only.

Descoped: retraining Epictetus on the full corpus. That adapter shows no decision effect, but it also has the smallest corpus by far (123 chunks against 437 and 540), so it's an underpowered arm rather than a real null. Whether corpus size explains it stays an open question in the write-up, not a work item.

------------------------------------------------------------------------

## Notes

Companion project: **ModelLens**, the architecture-agnostic interpretability toolkit used for the circuit-topology comparison. Some of the source texts and earlier measurements predate this repository; the measurement corrections, verification records, and corpus pipeline here are authoritative.
