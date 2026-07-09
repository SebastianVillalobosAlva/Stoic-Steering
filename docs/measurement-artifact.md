# The decoding-asymmetry artifact

This project measures steering at three levels (style, content, decision). Two of
those — the **judge-scored CAA style and content effects** — were, in an earlier
measurement, reported as large and positive. Under matched decoding they vanish.
This document records why, with the mechanism and the re-measured numbers, so the
correction lives with the code that enforces it.

## What produced the inflated numbers

The judge scores *steered* prose against an *unsteered baseline*. The size of the
effect depends entirely on the two texts being generated the same way. In the
earlier measurement they were not:

| | Steered | Baseline |
|---|---|---|
| decoding | **sampled** (temperature 0.6, top-p 0.9) | **greedy** |
| length | **~13 tokens** (`max_length` 20) | **100 tokens** |

The steered-generation path passed only the repetition controls
(`repetition_penalty`, `no_repeat_ngram_size`) to `.generate()` and dropped
`do_sample` and `max_new_tokens`, so generation silently fell back to the model's
`generation_config` — which for `meta-llama/Llama-3.2-3B` is
`do_sample=True, temperature=0.6, top_p=0.9, max_length=20`. The baseline used an
explicit greedy 100-token config. So the judge was scoring a short sampled snippet
against long greedy web-text — the decoding difference, not the steering vector.

This repo makes the bug unwritable: `generate()` is defined once
([stoic/model.py](../stoic/model.py)) and every caller — steered and baseline —
routes through it, so the two sides cannot use different decoding.

## Re-measured under matched decoding (this repo)

Same frozen steering vectors (cosine 1.0000 to the reference `.pt`), same judge
(gemini-2.5-flash), n=5 seeds. **Content effect** — (Δphilosophical_depth +
Δstoic_alignment)/2:

| Author | Layer | Earlier (asymmetric) | Matched greedy | Matched sampled |
|---|---|---|---|---|
| Marcus | 26 | +0.408 ± 0.136 | −0.175 ± 0.054 | +0.075 ± 0.165 (n.s.) |
| Seneca | 4 | +0.583 ± 0.121 | −0.117 ± 0.054 | −0.042 ± 0.204 (n.s.) |
| Epictetus | 8 | +0.767 ± 0.076 | −0.117 ± 0.080 | −0.192 ± 0.329 (n.s.) |

At coeff 0.11 the steered greedy output is byte-identical to baseline (greedy text
only visibly changes at coeff ≥ 1.0), so the residual is judge noise.

**Style / register effect** — stylistic-authenticity delta, canonical clean
configs, matched decoding, pre-registered "survives if > 2σ" rule:

| Author | Earlier (asymmetric) | Matched greedy | Matched sampled |
|---|---|---|---|
| Marcus | +1.00 | −0.150 ± 0.181 | −0.033 ± 0.139 |
| Seneca | +1.42 | −0.100 ± 0.091 | −0.017 ± 0.260 |
| Epictetus | +1.58 | +0.050 ± 0.126 | −0.017 ± 0.124 |

Nothing survives. Under fair measurement, CAA at the canonical coefficient moves
**nothing** at any level — style, content, or decision.

## What is NOT affected

- **The decision-level results.** The forced-choice dilemma probe takes a single
  forward pass and compares next-token logits over the `{A, B}` labels — no
  generation at all — so decoding settings cannot touch it. The CAA-flat /
  LoRA-moves decision split stands.
- **The LoRA content deltas.** Those were generated with `do_sample=False` on both
  sides (greedy, 100 tokens) — symmetric, and unaffected by this artifact.
- **The circuit-topology comparison.** Interpretability over activations, not
  judge-scored generation.

## Verification records in this repo

- Content re-measurement: `results/stage3_content_judge/` (greedy + sampled JSONs)
- Style re-measurement: `results/style_validation/`
- Narrative: [results/README.md](../results/README.md)
