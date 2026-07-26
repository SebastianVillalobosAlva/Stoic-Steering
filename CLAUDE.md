# CLAUDE.md — Stoic LLM (Clean Rebuild)

This project rebuilds the Stoic-steering pipeline from scratch as a clean,
understandable package and re-measures its headline results under matched
decoding. The clean reproduction (Pass A, Stages 0–4) and the clean circuit
analysis (Exp 12) are **complete and verified** — only Pass B (regenerate from
raw text) remains. The per-stage record with numbers is
[results/README.md](results/README.md); the measurement-artifact writeup is
[docs/measurement-artifact.md](docs/measurement-artifact.md).

## What this repo is about (framing — read before touching any prose)

The subject is **intervention locus**: does a behavior installed in
weight-space recruit the same circuit as one installed in activation-space,
and does the locus predict how the behavior fails? Stoic corpora are the
**instrument** used to build a behavioral axis. They are not the claim.

Every external-facing surface — README, tagline, abstract, write-up title —
leads with locus; philosophy appears in one sentence as methodology. The
reframe plan lives in `docs/reframe-brief.md`; the audit output goes to
`docs/reframe-plan.md`.

## Working rules (these override convenience)

- **Sebastian owns all external-facing prose.** Diagnose, map, propose beats,
  draft scaffolds and structure — never write README paragraphs, doc prose, or
  write-up text to be shipped as his. Give the map, not the paragraphs.
- **Retired claims are receipts, not deletions.** Anything superseded stays
  logged with a supersedes-pointer and its rationale. Never silently remove or
  edit a claim out of a doc or results file.
- **Median-based or full-N with significance testing.** No single-run deltas
  presented as findings. No outlier-carried aggregates.
- **Pre-register the criterion before the run.** Vanish branches get three-way
  framing (replication / vanish-with-clean-base / vanish-with-mushy-base — the
  last is inconclusive, not informative), never two-way.
- **Evidence altitude matches what's demonstrated.** 3B model, small-model
  probe. Never "here is a mitigation."
- Ask before any change that touches `results/` JSONs.

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
- **Behavioral axis is a config object**, not hardcoded to the philosophers:
  corpus source, contrast-pair generator, dilemma set, adapter name. Target —
  adding an axis is a config file, not a code change. Existing philosopher
  results must reproduce byte-identical where deterministic after any refactor.

## Status

- **Pass A (Stages 0–4) — complete & verified.** Deterministic decoding; base
  P(stoic) = 0.542 (load-bearing); new vectors cosine ≥0.99 vs frozen `.pt`;
  CAA null at style, content, and decision under matched decoding; LoRA moves
  decisions (Seneca both stance buckets, Marcus accepting-only, Epictetus null).
  Numbers: [results/README.md](results/README.md).
- **Exp 12 (clean circuit analysis) — complete.** `results/exp12_circuits/`.
  Status remains **n=1-per-stance pilot** until the v3 sweep.
- **Pass B — built, not yet run.** The corpus pipeline (`stoic/corpus.py`,
  `stoic/pairs.py`) is built and verified against the frozen chunk counts; the
  fresh-data re-run of Stages 2–4 (writing only to `generated/`, ~$10–15 API for
  pair + judge rounds) is still open.
- **Open discrepancy to resolve before the write-up:** story beat 3 calls
  Seneca the strongest decision-mover, but the in-repo sign test is 25+/15−,
  p=0.154 (n.s.), while Marcus is 27+/13−, p=0.038. Summary docs elsewhere cite
  a Seneca t≈2.58. Either these measure different things and both get stated,
  or the summary docs overclaim. Resolve, don't paper over.

## Next steps (priority order)

0. **Reframe audit + axis-agnostic refactor.** Audit produces
   `docs/reframe-plan.md` (map only, no prose rewrites). Refactor makes the
   behavioral axis a config object. Both gate step 6.
1. **ModelLens: upstream bugfix + core regression tests.** The
   `_capture_activations` closure-return bug is currently patched locally in
   `exp12_circuit_analysis.py` — fix it upstream before anything public. Then
   hook cleanup after exceptions, custom `metric_fn` dispatch, adapter
   dispatch, patching determinism. Gates step 4.
2. **`dilemmas_v3` — the reasoning-vs-echo gate (DO FIRST among experiments).**
   2×2 design: Letters-core vs off-topic × plain vs Stoic-idiom phrasing; 10
   items/cell (40), stance-balanced, calibrated to per-cell P(stoic) ≈ 0.5
   *before* any eval (v1's 0.881 is the cautionary record). Tests whether the
   LoRA decision shift is reasoning or Senecan lexical echo — the biggest live
   threat to the decision claim.
3. **Behavioral LoRA eval on v3** (all three adapters **plus the
   matched-length non-philosophical control LoRA**, $0 eval + one Colab
   training run) → the 2×2 verdict. The control belongs here, not just in the
   stability sweep: without it, "LoRA moves decisions" can't be distinguished
   from "any fine-tuning on formal prose moves decisions."
4. **Circuit sweep on v3** ($0, local) → retires the n=1-per-stance pilot caveat
   on Exp 12c. **Gated:** needs step 1 — the sweep runs through those hooks.
5. **Figure 1** — side-by-side CAA vs LoRA circuit graphs from the exp12 sweep
   JSONs, content logit-diff metric, base circuit as shared control panel. This
   is the image that carries the write-up and the applications. Priority over
   figures #2/#3.
6. **Second behavioral axis: sycophancy** (API gen + local). The highest-value
   remaining build. Every current claim is n=1 in *behavior* — one corpus
   family, one value dimension. Run the same three-level protocol plus the
   circuit comparison on a non-philosophical axis; contrast pairs from Chen et
   al. 2025 persona-vector traits (arXiv 2507.21509). **Gated on step 0.**
   Pre-register the three readings: replication (CAA null, LoRA reaches
   decisions → locus claim generalizes); divergence (CAA moves sycophancy where
   it couldn't move ethics → claim narrows, needs a hypothesis about what
   distinguishes the behaviors); both null (uninformative, likely a 3B capacity
   ceiling — report as such, don't spin it).
7. **Stability sweep** (temperature × seed on the judge-free decision
   instrument, $0, local), now across **both** axes. A stability result on one
   philosophy axis is still a philosophy result. **Gated on v3:** if v3 returns
   pure lexical echo on the philosophy axis, run on sycophancy only; if both
   axes are null, cancel. Needs the matched-length non-philosophical LoRA as
   control.
8. **Pass B** (~$10–15) — regenerate pairs, re-run Stages 2–4 on fresh data.
   Either outcome (tight agreement / pair-sampling drift) is reportable. Slots
   into writing time as unattended work.
9. **Figures #2/#3** ($0, from existing JSONs via `scripts/make_figures.py`) —
   pair-quality flip; CAA coefficient sweep flat to 1.5. Nice-to-have tier.
10. **Write-up** — Alignment Forum, titled on the locus claim. Neel's sequence:
    3 claims → abstract → outline → figures → intro → prose. After v3 and the
    second axis, since both land in the claims.

**Schedule anchor:** MATS Summer 2027 opens mid-December 2026; Neel Nanda's
stream historically ~January 2. The write-up should be public *before* that
application. Working back: steps 0–3 by October, 4–6 by November, draft and
post early December. The write-up is the deliverable; the repos are supporting
evidence for it, not the submission.

Descoped: Epictetus full-corpus retrain (Enchiridion + Discourses). Decision
2026-07-16: the Epictetus adapter is a decision-level null (ΔP +0.000), so the
corpus-size hypothesis stays a named open question in the write-up, not a work
item. Report corpus sizes in the results table so the power difference is
visible — Epictetus is an underpowered arm, not a finding.

Also descoped: harness-locus (scaffold-level constraints as a third
intervention locus) — a legitimate extension and a separate project; it does
not compete for time before the write-up ships. Refusal-direction interaction
(Arditi-style) — one future-work paragraph.

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
- **Calibration harness (built 2026-07-21):** `python -m stoic calibrate`
  (logic `stoic/calibrate.py`, orchestration `stoic/stages/calibrate.py`).
  Validates structure ($0, `--validate-only`), scores candidates on the base
  model with the unchanged v2 ruler, reports per-cell means + item-level
  outliers (P outside [0.2, 0.8] = replacement candidates), gates at
  |mean − 0.5| ≤ 0.05 per cell. Candidates live in
  `data/generated/dilemmas_v3_candidates.json` (never reference/). Demo run on
  the schema fixture scored 0.870 overall — the v1 failure shape, caught by
  the gate as designed (`results/dilemmas_v3_calibration/`).
- **Control adapter:** LoRA trained on matched-length non-philosophical text,
  included in the v3 behavioral eval (see step 3 above).

## Story beats (application framing — keep the write-up aligned to these)

1. "I found and fixed my own measurement artifact" — decoding-asymmetry
   writeup + the clean reproduction. Primary credibility signal; never bury it.
2. "One method changes the circuit, the other changes nothing" — CAA at coeff
   0.11 leaves the stoic-content circuit essentially untouched (deviations from
   base are within threshold-flicker size), while LoRA shifts clean logit diff
   by up to 1.43 and drops up to 7 nodes. Same split as the behavior. n=2 items
   — pilot, not settled, until the v3 sweep. **Supersedes** the earlier beat 2
   ("CAA and LoRA produce comparable surface behavior through different circuit
   topologies"), which was written pre-null and is contradicted by Exp 12: CAA
   has no behavioral effect to be comparable with, and no circuit effect to
   differ by.
3. "Only weight adaptation reaches decisions" — CAA null everywhere including
   circuit-level (±0.003–0.015); LoRA moves decision (judge-free, exact) plus
   style + content (single merged-adapter judge eval, not seed-tested). Circuit
   topology predicted the split before behavior showed it.
4. "What LoRA installs is heterogeneous, and I'm testing whether it's
   reasoning or echo" — Marcus = passivity prior, Seneca = strongest circuit
   modifier (item-dependent character; see the open sign-test discrepancy under
   Status), Epictetus = underpowered null; v3 is the test.
5. "And it replicates off the philosophy axis" — the dissociation tested on
   sycophancy. This is what converts a single-corpus finding into a claim about
   intervention loci. Pending step 6.
6. "I produce peer-grade research independently" — reproduction record,
   ModelLens (tests + MCP + SAE), self-contained corpus pipeline.

### Safety framing (keep intact wherever the project is described)

Three links, ordered by how well the evidence supports them: (1) durability may
be locus-dependent — Larsen 2025's 18–28% decision flips say RLHF refusal is
shallow, and the stability sweep asks whether weight-space installation
survives where activation-space doesn't; (2) activation steering as a runtime
safety lever gets a caution — a negative result about a proposed mitigation is
a safety contribution; (3) the artifact story is evals hygiene — a silent
decoding asymmetry manufactured an effect that wasn't there, and that bug class
produces false confidence in deployed mitigations.

Housekeeping (resolved 2026-07-16): the 25/40 sign test the READMEs cite is now
computed in-repo (`dilemmas.sign_test`, wired into stage 4). Verified from the
checked-in JSONs: Marcus 27+/13− p=0.038, Seneca 25+/15− p=0.154 (the cited
n.s.), Epictetus 17+/23− p=0.430.

Frozen binaries (resolved 2026-07-17): the untracked steering vectors + LoRA
adapters are hosted at HF `seb-vil/llama-3.2-3b-stoic-steering` (public).
`python scripts/fetch_artifacts.py` downloads them into place and verifies
against `data/MANIFEST.sha256`. A fresh clone can now reproduce Stages 2 and 4,
not just 0–1.
