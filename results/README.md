# Reproduction record — Pass A

One JSON per stage checkpoint. Run with the `stoic-llm` conda env (py3.11,
torch 2.5.1, transformers 4.57.3) against the frozen `data/reference/` artifacts.
Reproduce with: `python -m stoic all`.

| Stage | Checkpoint | Target | Result | Status |
|---|---|---|---|---|
| 0 | Deterministic decoding | same prompt → identical output twice | identical | ✅ PASS |
| 1 | Base P(stoic) on v2 set | 0.542 (ref 0.541601902275579) | **0.541602** | ✅ PASS |
| 2 | New vs frozen vectors | cosine ≥ 0.99 | 1.0000 (all 3) | ✅ PASS |
| 2 | Injection site (Epictetus L8) | bites `hidden_states[9]`, not `[8]` | confirmed | ✅ PASS |
| 2 | Steered dilemmas ≈ flat (Exp 10 null) | ΔP ≈ 0 all authors | see below | ✅ PASS |

### Stage 2 — CAA at coeff 0.11 reproduces the Exp 10 decision-level null

| Author | Layer | cosine→frozen | ΔP (rebuild) | ΔP (reference) |
|---|---|---|---|---|
| Marcus | 26 | 1.0000 | −0.0001 | −0.0001 |
| Seneca | 4 | 1.0000 | +0.0002 | +0.0002 |
| Epictetus | 8 | 1.0000 | −0.0007 | −0.0007 |

Rebuilt vectors are extracted at the same site they inject (`layers[L].mlp`),
from the frozen reference pairs, and match the frozen `.pt` vectors to 4 decimals
in both direction (cosine) and magnitude (norm ratio 1.000). CAA moves decisions
**not at all** — reproducing the headline Exp 10 result.

## Reference repair (Stage 2 prerequisite)

`data/reference/processed/epictetus/neutral_pairs.json` was syntactically corrupt
in the assembled reference set (malformed JSON: an object-of-objects `{ {...} }`
instead of a valid array/`{"pairs":[...]}`). Its 53 pairs were verified
**byte-identical** (after strip) to the intact legacy provenance original at
`stoic-llm-legacy/data/processed/epictetus/neutral_pairs.json`. The reference
file was restored from that original (content unchanged, JSON now valid). The
corrupt copy is archived in the session scratchpad, not in the repo.

## Stage 3 — content effect does NOT reproduce (major finding)

Exp 9's headline CAA content effect (Marcus +0.408, Seneca +0.583, Epictetus
+0.767, Gemini judge) **does not survive matched decoding**. Root cause: the
legacy steered-generation path (`run_model_with_hook`) silently dropped
`do_sample`/`max_new_tokens`, so steered text was **sampled (temp 0.6, top-p 0.9)
and truncated to ~13 tokens** while the baseline was **greedy, 100 tokens**. The
judge scored the decoding difference, not the steering. Full line-quoted root
cause: `stoic-llm-legacy/LEGACY.md`.

Rebuilt with one canonical `generate()` (both sides identical decoding), same
frozen vectors (cosine 1.0000), same judge (gemini-2.5-flash), n=5 seeds:

| Author | Layer | Exp 9 (artifact) | Matched greedy | Matched sampled |
|---|---|---|---|---|
| Marcus | 26 | +0.408 ± 0.136 | −0.175 ± 0.054 | +0.075 ± 0.165 (n.s.) |
| Seneca | 4 | +0.583 ± 0.121 | −0.117 ± 0.054 | −0.042 ± 0.204 (n.s.) |
| Epictetus | 8 | +0.767 ± 0.076 | −0.117 ± 0.080 | −0.192 ± 0.329 (n.s.) |

At coeff 0.11 the steered greedy output is byte-identical to baseline (greedy
text only visibly changes at coeff ≥ 1.0), and the sampled distribution doesn't
move either. **CAA at 0.11 is null at the content level under fair measurement.**
This strengthens rather than weakens the decision-level story: Exp 10's CAA-flat
result (reproduced exactly in Stage 2) was measured on logits, immune to the
artifact, and stands.

JSONs: `stage3_content_judge/content_20260704_192154.json` (greedy),
`content_sampled_20260705_001353.json` (sampled).

## Style validation — the register claim collapses too

Exp 3b's stylistic-authenticity deltas (Marcus +1.00, Seneca +1.42,
Epictetus +1.58) — the column the tracker called "the robust, valid result" —
were re-tested under matched decoding at the canonical clean configs
(pre-registered rule: survives if seed-averaged style delta > 2σ above zero
in either regime; n=5 seeds, same hardened Gemini judge):

| Author | Exp 3b (artifact) | Matched greedy | Matched sampled | Verdict |
|---|---|---|---|---|
| Marcus | +1.00 | −0.150 ± 0.181 | −0.033 ± 0.139 | collapses |
| Seneca | +1.42 | −0.100 ± 0.091 | −0.017 ± 0.260 | collapses |
| Epictetus | +1.58 | +0.050 ± 0.126 | −0.017 ± 0.124 | collapses |

The greedy arm re-scored the saved Stage 3 generations (byte-identical pairs
scored as delta 0 by construction); the sampled arm regenerated the seeded
Stage 3 texts. Under fair measurement CAA at coeff 0.11 moves **nothing** —
style, content, or decisions. The old "style moves robustly" signal was the
judge reacting to short sampled snippets vs long greedy baselines.

Caveats recorded in the JSON: canonical configs (Exp 3b used superseded
all-L8 picks), and coefficient 0.11 only — greedy output barely changes below
coeff ~1.0, so whether any coefficient produces genuine Stoic register under
matched decoding is untested (future sweep).

JSON: `style_validation/style_20260705_212411.json`.

## Stage 4 — LoRA decision shift reproduces EXACTLY (Exp 11)

Frozen clean adapters merged onto a fresh base each (no stacking possible),
judge-free logit eval, all local CPU. Every number matches the frozen
reference (`data/reference/dilemmas/v2/lora/dilemma_eval_20260701_140942.json`)
to the fourth decimal:

| Author | ΔP overall | t | accepting (n=22) | active (n=18) | Pattern |
|---|---|---|---|---|---|
| Marcus | +0.0307 | 2.00 | +0.0652 (t=2.99) | −0.0114 (n.s.) | passivity prior |
| **Seneca** | **+0.0606** | **2.58** | **+0.0781 (t=2.38)** | **+0.0391 (t=1.16)** | **both buckets positive ✓** |
| Epictetus | +0.0003 | 0.07 | +0.0046 | −0.0051 | null (smallest corpus) |

Base integrity held: baseline 0.541602 before AND after all three merges,
max per-item drift 0.00e+00 — the fresh-base-per-adapter rule verified.

With Stage 4 done, **Pass A is complete.** Final scoreboard: everything
logit-measured reproduces exactly (Stages 1, 2, 4 — the 0.542 baseline, the
CAA decision null, the LoRA decision shift); everything judge-scored on the
CAA side was a decoding artifact (Stage 3 + style validation). LoRA is the
only intervention with real effects, and it reaches the decision layer.

JSON: `stage4_lora_dilemmas/lora_dilemmas_20260705_225558.json`.

## Exp 12 — circuit topology: CAA leaves the circuit untouched, LoRA rewires it (new work)

First run of ModelLens `discover_circuit` (activation patching over all 56
attn/mlp sublayers) on the Stage-4-verified clean adapters — not the old
bridge scripts. Content-relevant metric: logit(" A") − logit(" B") on dilemma
`ctrl_03`, clean = stoic option as A, corrupted = options swapped — the only
input difference is *which option is Stoic*. Matched data across all seven
conditions; base model as shared control; threshold 0.15; attention-routing
edges added via an eager-attention pass.

| Condition | clean | total effect | max node shift vs base | topology |
|---|---|---|---|---|
| base | +3.88 | −8.97 | — | 33 nodes: early/mid processors (0–15), late gate cluster (MLP 23/25/26/27), booster attn 26 |
| CAA marcus | +3.91 | −8.98 | ±0.003 | identical to base |
| CAA seneca | +3.98 | −9.17 | ±0.012 | identical to base |
| CAA epictetus | +3.86 | −8.95 | ±0.015 | identical to base |
| LoRA marcus | +3.62 | −8.47 | ±0.130 | moderately shifted |
| **LoRA seneca** | **+2.45** | **−6.73** | **±0.174** | **+1 booster, −1 gate, blocks 1/5 reweighted** |
| LoRA epictetus | +4.03 | −8.88 | ±0.032 | near-base |

Findings:
1. **CAA at coeff 0.11 does not change the stoic-content circuit at all**
   (max normalized-effect shift 0.015) — the circuit-level view agrees with
   the behavioral nulls at every level.
2. **LoRA rewires the circuit in proportion to its behavioral decision
   effect: Seneca > Marcus > Epictetus ≈ 0** — the same ordering as Exp 11's
   dilemma shifts. Seneca's raw sensitivity to which-option-is-stoic
   *compresses* (−8.97 → −6.73) even as it chooses the Stoic option more.
3. The Exp 5/6 "circuit topology predicts the behavioral split" claim now
   rests on `discover_circuit` with a content metric and clean adapters.

Note: ModelLens's `_capture_activations` had a closure bug that made
activation patching unrunnable; fixed upstream (ModelLens `44d9b77`) and run
against the fix. Figures: `exp12_circuits/exp12_circuit_{author}.png`
(Base | CAA | LoRA side-by-side). JSON: `exp12_circuits/exp12_20260706_231538.json`.
Runtime ~59 min patching + ~5 min attention pass, all local CPU, $0.
