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

The vectors + adapters are NOT in git (too large); fetch them from HF with
`python scripts/fetch_artifacts.py` (verifies against `data/MANIFEST.sha256`).

## Design rules (each kills a bug from the pre-rebuild code)

- Prefer functions over classes; most old "classes" held no state.
- Steering hook is the only real state → use a context manager (leaked hook
  becomes structurally impossible; no manual cleanup/__del__).
- `generate()` defined once → mismatched-decoding bug unwritable.
- `extract_vector(pairs, layer)` extracts AND injects at the same layer.
- Load tokenizer from BASE, never from the adapter folder.
- LoRA merge: fresh base per adapter + assert base integrity (0.542 → 0.542, drift 0).
- These rules are pinned by CPU-only unit tests in `tests/` (hook hygiene,
  canonical decoding, dilemma math, stats vs published numbers, reference-wall
  tripwire, fixture integrity). Run `pytest` before committing changes to
  `stoic/` — seconds, no model download.
- Stage orchestration lives in `stoic/stages/` (verify/content/adapters/passb);
  `__main__.py` is parse + dispatch only.

## Status

- **Pass A (Stages 0–4) — complete & verified.** Deterministic decoding; base
  P(stoic) = 0.542 (load-bearing); new vectors cosine ≥0.99 vs frozen `.pt`;
  CAA null at style, content, and decision under matched decoding; LoRA moves
  decisions (Seneca both stance buckets, Marcus accepting-only, Epictetus null).
  Numbers: [results/README.md](results/README.md).
- **Exp 12 (clean circuit analysis) — complete.** `results/exp12_circuits/`.
- **Pass B — built, not yet run.** The corpus pipeline (`stoic/corpus.py`,
  `stoic/pairs.py`) is built and verified against the frozen chunk counts; the
  fresh-data re-run of Stages 2–4 (writing only to `generated/`, ~$10–15 API for
  pair + judge rounds) is still open.

## Next steps (priority order)

1. **`dilemmas_v3` — the reasoning-vs-echo gate (DO FIRST).** 2×2 design:
   Letters-core vs off-topic × plain vs Stoic-idiom phrasing; 10 items/cell (40),
   stance-balanced, calibrated to per-cell P(stoic) ≈ 0.5 *before* any eval
   (v1's 0.881 is the cautionary record). Tests whether the LoRA decision shift
   is reasoning or Senecan lexical echo — the biggest live threat to the
   decision claim.
2. **Behavioral LoRA eval on v3** (all three adapters **plus the
   matched-length non-philosophical control LoRA**, $0 eval + one Colab
   training run) → the 2×2 verdict. The control belongs here, not just in the
   stability sweep: without it, "LoRA moves decisions" can't be distinguished
   from "any fine-tuning on formal prose moves decisions."
3. **Circuit sweep on v3** ($0, local) → retires the n=1-per-stance pilot caveat
   on Exp 12c. **Gated:** needs the ModelLens core regression tests first — the
   sweep runs through exactly those hooks.
4. **Stability sweep** (temperature × seed on the judge-free decision
   instrument, $0, local). **Gated on v3:** if v3 returns pure lexical echo,
   cancel — "how stable is a wording preference" is not a safety result. Needs a
   matched-length non-philosophical LoRA as the control.
5. **Pass B** (~$10–15) — regenerate pairs, re-run Stages 2–4 on fresh data.
   Either outcome (tight agreement / pair-sampling drift) is reportable.
6. **Figures #2/#3** ($0, from existing JSONs via `scripts/make_figures.py`) —
   pair-quality flip; CAA coefficient sweep flat to 1.5.
7. **Write-up** — after v3, since its verdict lands in the third claim.

Descoped: Epictetus full-corpus retrain (Enchiridion + Discourses). Decision
2026-07-16: the Epictetus adapter is a decision-level null (ΔP +0.000), so the
corpus-size hypothesis stays a named open question in the write-up, not a work
item.

## dilemmas_v3 scope (summarized from the private build plan)

The lexical-echo confound is the biggest live threat to the decision claim:
ext_02's "on loan" is Senecan idiom nearly verbatim, so the LoRA shift could be
register reaching the choice through the option's wording.

- **2×2 design.** Topic axis: Letters-core topics (grief, old age, wealth,
  ambition, illness, friendship) vs. topics Seneca barely touches. Phrasing
  axis: each dilemma's Stoic option in plain modern wording vs. Stoic/Senecan
  idiom ("on loan", "play your part").
- **Pre-registered readings:** movement concentrated in core-topic cells →
  topic proximity; movement tracking idiom across topics → lexical echo;
  plain-worded off-topic movement → strongest available reasoning claim.
- **Scale:** 10 items/cell (40 total); 20/cell only if calibration goes
  smoothly. Stance-balanced *within each cell* so the same set serves the
  circuit sweep.
- **Calibration gate:** per-cell baseline P(stoic) ≈ 0.5, both label orders
  averaged, BEFORE any eval (v1's 0.881 is the cautionary record).
- **Control adapter:** LoRA trained on matched-length non-philosophical text,
  included in the v3 behavioral eval (see step 2 above).

## Story beats (application framing — keep the write-up aligned to these)

1. "I found and fixed my own measurement artifact" — decoding-asymmetry
   writeup + the clean reproduction. Primary credibility signal; never bury it.
2. "The three levels dissociate, and only weight adaptation reaches
   decisions" — CAA null everywhere; LoRA moves decision (judge-free, exact)
   plus style + content (single merged-adapter judge eval, not seed-tested).
3. "What LoRA installs is heterogeneous, and I'm testing whether it's
   reasoning or echo" — Marcus = passivity prior, Seneca = strongest circuit
   modifier (item-dependent character), Epictetus = null; v3 is the test.
4. "I produce peer-grade research independently" — reproduction record,
   ModelLens (tests + MCP + SAE), self-contained corpus pipeline.

Housekeeping (resolved 2026-07-16): the 25/40 sign test the READMEs cite is now
computed in-repo (`dilemmas.sign_test`, wired into stage 4). Verified from the
checked-in JSONs: Marcus 27+/13− p=0.038, Seneca 25+/15− p=0.154 (the cited
n.s.), Epictetus 17+/23− p=0.430.

Frozen binaries (resolved 2026-07-17): the untracked steering vectors + LoRA
adapters are hosted at HF `seb-vil/llama-3.2-3b-stoic-steering` (public).
`python scripts/fetch_artifacts.py` downloads them into place and verifies
against `data/MANIFEST.sha256`. A fresh clone can now reproduce Stages 2 and 4,
not just 0–1.