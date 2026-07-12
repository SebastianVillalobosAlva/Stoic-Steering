# CLAUDE.md — Stoic LLM (Clean Rebuild)

This project rebuilds the Stoic-steering pipeline from scratch as a clean,
understandable package and re-measures its headline results under matched
decoding. The clean reproduction (Pass A, Stages 0–4) and the clean circuit
analysis (Exp 12) are **complete and verified** — only Pass B (regenerate from
raw text) remains. The per-stage record with numbers is
[results/README.md](results/README.md); the measurement-artifact writeup is
[docs/measurement-artifact.md](docs/measurement-artifact.md).

## Non-negotiable rules

- `data/reference/` is READ-ONLY. Never write to it, never overwrite, never
  regenerate its contents. Every stage reads frozen artifacts from here.
- `data/generated/` is where ALL pipeline output goes.
- **Pass A** loads frozen artifacts from `reference/` and verifies against known
  numbers. **Pass B** (the open work) regenerates from raw text into `generated/`.
- Do NOT regenerate contrastive pairs, vectors, or adapters into `reference/`.
  If a stage is about to write into `reference/`, that is a bug — stop.

## Model

- Base: `meta-llama/Llama-3.2-3B`, float16. No 1B.
- One canonical decoding set, used everywhere (define once in model.py):
  `do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3`

## Canonical configs (ground truth)

- CAA clean best layers / coeff: **Marcus L26, Seneca L4, Epictetus L8, coeff 0.11**
- LoRA: r=8, alpha=32, targets q_proj + v_proj, 3 epochs
- Dilemma baseline P(stoic) = **0.542** (v2 set, 40 items, both label orders averaged)

## Frozen reference artifacts

- `neutral_pairs.json` ×3 (53 entries each) — CAA extraction inputs
- `dilemmas_v2.json` (40 items) — the forced-choice ruler → 0.542
- `{author}_steering_3B` steering vectors (.pt) ×3 — Stage 2 cosine targets (expect ≥0.99)
- `lora_{author}_clean` adapters ×3 — Stage 4 targets
- results JSONs (judges/, dilemmas, bridge/) — reference numbers

## Design rules (each kills a bug from the pre-rebuild code)

- Prefer functions over classes; most old "classes" held no state.
- Steering hook is the only real state → use a context manager (leaked hook
  becomes structurally impossible; no manual cleanup/__del__).
- `generate()` defined once → mismatched-decoding bug unwritable.
- `extract_vector(pairs, layer)` extracts AND injects at the same layer.
- Load tokenizer from BASE, never from the adapter folder.
- LoRA merge: fresh base per adapter + assert base integrity (0.542 → 0.542, drift 0).

## Status

- **Pass A (Stages 0–4) — complete & verified.** Deterministic decoding; base
  P(stoic) = 0.542 (load-bearing); new vectors cosine ≥0.99 vs frozen `.pt`;
  CAA null at style, content, and decision under matched decoding; LoRA moves
  decisions (Seneca both stance buckets, Marcus accepting-only, Epictetus null).
  Numbers: [results/README.md](results/README.md).
- **Exp 12 (clean circuit analysis) — complete.** `results/exp12_circuits/`.
- **Pass B — the open work.** The corpus pipeline (`stoic/corpus.py`,
  `stoic/pairs.py`) is built and verified against the frozen chunk counts; the
  remaining task is the fresh-data re-run of Stages 2–4 (writing only to
  `generated/`, ~$10–15 API for pair + judge rounds).