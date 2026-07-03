# Stoic LLM — clean rebuild

Can philosophical (Stoic) reasoning be steered into an LLM, and does it reach
the *decision* layer or only the *style* layer? This repo rebuilds the
Stoic-steering pipeline from scratch as a clean, function-first package and
reproduces the headline results on **Llama-3.2-3B** (float16, CPU):

- **CAA activation steering** moves style and judge-scored content, but moves
  forced-choice **decisions not at all** (Exp 9 / Exp 10).
- **LoRA fine-tuning** reaches the decision layer where CAA doesn't (Exp 11).

See [`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md) for the two-pass plan and
stage checkpoints, and [`CLAUDE.md`](CLAUDE.md) for the working rules.

## The reference wall

- `data/reference/` — **frozen, read-only**: the artifacts that produced the
  published numbers.
- `data/generated/` — everything the rebuilt pipeline produces.

Pass A rebuilds code against `reference/` and verifies against known numbers;
Pass B regenerates from raw text into `generated/`.

## Install

```bash
pip install -e .            # core: Stage 0-2 (steering + dilemmas)
pip install -e ".[judge]"   # + Stage 3 judge (Gemini/Anthropic)
pip install -e ".[lora]"    # + Stage 4 LoRA
pip install -e ".[all]"     # everything
```

Requires access to `meta-llama/Llama-3.2-3B` (gated) and a Hugging Face token.

## Run the checkpoints

```bash
python -m stoic stage0   # deterministic decoding
python -m stoic stage1   # base P(stoic) == 0.542   (load-bearing)
python -m stoic stage2   # rebuilt vectors cosine 1.0000 + Exp 10 null
python -m stoic all      # all of the above in one model load
```

Each writes a JSON checkpoint under `results/`. Current status:

| Stage | Checkpoint | Result |
|---|---|---|
| 0 | deterministic decoding | ✅ identical output twice |
| 1 | base P(stoic) = 0.542 | ✅ 0.541602 |
| 2 | cosine ≥ 0.99 vs frozen; Exp 10 null | ✅ cosine 1.0000; ΔP ≈ 0 all authors |

See [`results/README.md`](results/README.md) for the full record.

## Canonical configs

- Base: `meta-llama/Llama-3.2-3B`, float16
- CAA clean layers / coeff: Marcus L26, Seneca L4, Epictetus L8, coeff 0.11
- LoRA: r=8, α=32, targets q_proj + v_proj, 3 epochs
- Decoding (one canonical set): `do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3`
- Dilemma baseline P(stoic) = 0.542 (v2 set, both label orders averaged)
