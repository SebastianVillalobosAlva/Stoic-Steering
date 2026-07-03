# Reproduction & Rebuild Plan — Stoic LLM (Clean Implementation)

**Purpose:** Rebuild the Stoic-steering codebase from scratch as a clean, understandable
pipeline, and reproduce the headline results (Exp 9, 10, 11) end-to-end. A faithful
reproduction in clean code is a *replication*, not just a refactor.

**Model scope:** Llama-3.2-3B only (float16, 28 layers, hidden_dim 3072). No 1B.

---

## Core principle: the reference wall

Two data directories, and the wall between them is sacred:

- `data/reference/`  — FROZEN. Read-only. The artifacts that produced the published
  numbers. NEVER overwrite, never regenerate, never let any script write here.
- `data/generated/`  — everything the new pipeline produces. Gitignored except final
  results JSONs.

**Pass A loads from `reference/`. Pass B writes only to `generated/`.** If any stage is
about to write into `reference/`, that is a bug — stop.

---

## Frozen reference artifacts (already assembled)

| Artifact | Role |
|---|---|
| `neutral_pairs.json` ×3 (53 entries each: Marcus, Seneca, Epictetus) | CAA extraction inputs (Exp 9) |
| `dilemmas_v2.json` (40 items) | THE RULER — forced-choice instrument, baseline 0.542. Never regenerated. |
| `dilemmas_v1.json` | Failed-calibration record (baseline 0.881). Documentation only. |
| `sources.json` | Pass B input: Gutenberg URLs + content_start/content_end slicing boundaries. Verify boundaries before trusting. |
| clean steering vectors (.pt) ×3 | Stage 2 cosine-similarity targets |
| `lora_{author}_clean` adapters ×3 | Stage 4 targets (r=8, α=32, q_proj+v_proj) |
| results JSONs: `judges/`, dilemma folder(s), `bridge/` | Reference numbers to check against |

---

## Canonical configs (ground truth)

- Base: `meta-llama/Llama-3.2-3B`, float16
- CAA clean best layers / coeff: **Marcus L26, Seneca L4, Epictetus L8, coeff 0.11**
- LoRA: r=8, alpha=32, targets q_proj + v_proj, 3 epochs
- Decoding (ONE canonical set, used everywhere): greedy — do_sample=False,
  repetition_penalty=1.3, no_repeat_ngram_size=3
- Dilemma baseline P(stoic): **0.542** (v2 set, both label orders averaged)

---

## Target structure

stoic/
  config.py     # paths + Config dataclass (model, per-author layer/coeff, gen kwargs)
  model.py      # load_model(), generate()  <- decoding lives HERE, nowhere else
  corpus.py     # download, slice (content_start/end), chunk, filter   [Pass B]
  pairs.py      # contrastive pair generation via Claude API           [Pass B]
  steering.py   # extract_vector(pairs, layer); steering() context manager
  lora.py       # prep_jsonl, train, merge (fresh base per adapter)
  judge.py      # judge scoring + seed_eval (Gemini judge)
  dilemmas.py   # forced-choice harness (both label orders averaged)
cli.py          # python -m stoic pairs|extract|steer|train|judge|dilemma

**Design rules (each kills a bug from the old repo):**
- Prefer functions over classes. Most old "classes" held no state — port them as functions.
- The steering hook is the ONLY real state → use a context manager, so a leaked hook
  is structurally impossible. No manual cleanup/__del__ dance.
- `generate()` defined once → mismatched-decoding bug becomes unwritable.
- `extract_vector(pairs, layer)` extracts AND injects at the same layer → extraction-bug fixed by signature.
- Load base tokenizer from BASE, never from the adapter folder (the 17MB tokenizer.json
  in each adapter dir is redundant).
- LoRA merge: fresh base per adapter + assert base integrity (start 0.542 / end 0.542 / drift 0).

---

## Pass A — rebuild code against frozen artifacts

Each stage has a checkpoint. Do not proceed past a failing checkpoint.

| Stage | Build | Checkpoint | Cost |
|---|---|---|---|
| 0 | `config.py`, `model.py` | same prompt → identical output twice (deterministic) | $0 |
| 1 | `dilemmas.py` | **base P(stoic) = 0.542 exactly** on v2 set | $0 |
| 2 | `steering.py` | new vectors vs frozen .pt: cosine ≥ ~0.99; inject L8 bites at hidden_states[9]; steered dilemmas at 0.11 ≈ flat (reproduces Exp 10 null) | $0 |
| 3 | `judge.py` | Epictetus L8 content effect ≈ +0.767 (error bars overlap Exp 9; NOT decimal-exact — judge is nondeterministic) | ~$3–5 |
| 4 | `lora.py` | Seneca dilemma shift positive in BOTH stance buckets, t ≈ 2+ (reproduces Exp 11; pattern not decimals) | free T4 + $0 |

Stage 1's 0.542 is the load-bearing checkpoint. If it doesn't reproduce to the third
decimal, stop — nothing downstream is trustworthy until it does.

---

## Pass B — regenerate from raw text (robustness result)

Only after Pass A passes. Build `corpus.py` + `pairs.py`, re-download from Gutenberg,
re-slice, re-filter, regenerate ~53 pairs/philosopher, re-run Stages 2–4 on FRESH data
(written to `generated/`).

- Pairs are stochastic → fresh numbers will differ from originals. That's expected.
- Pass criterion is the PATTERN, not the decimals: all three positive on content,
  CAA flat on decisions, LoRA/Seneca moves.
- Tight agreement = pipeline robust. Drift = effect is pair-sampling sensitive.
  Either is honest, reportable material.
- Budget ~$10–15 API total (pairs + judge rounds). Check console balance before Stage 3;
  June estimates in the old tracker were never reconciled.

---

## Legacy repo

Freeze, don't touch: commit as-is → `git tag v0-exp11` → archive/rename to
`stoic-llm-legacy`. It is the provenance record for Exps 1–11. Do NOT clean it up or fix
old bugs — editing it destroys its value as evidence. Add a `LEGACY.md` mapping
experiment → script → results file → valid/superseded.

---

## Logging

New repo gets `results/` from day one, one JSON per stage checkpoint. The reproduction
must be *documented*, not just done — that record is what turns "I refactored" into
"I replicated my findings in a clean implementation."