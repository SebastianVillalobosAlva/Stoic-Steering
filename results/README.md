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

## Not yet run (needs Colab, deferred)

- Stage 4 — LoRA decision shift (Seneca positive both stance buckets, Exp 11)
