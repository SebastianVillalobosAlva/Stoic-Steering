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

## Not yet run (need $ / Colab, deferred)

- Stage 3 — judge-scored content effect (Gemini judge; Epictetus L8 ≈ +0.767, Exp 9)
- Stage 4 — LoRA decision shift (Seneca positive both stance buckets, Exp 11)
