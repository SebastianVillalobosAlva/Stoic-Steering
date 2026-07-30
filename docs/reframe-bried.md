## Status
- TASK 1 (reframe audit) — DONE. Output: docs/reframe-plan.md. README, CLAUDE.md,
  and ModelLens README have since been rewritten by hand; the plan's line
  anchors are stale.
- TASK 2 (axis-agnostic refactor) — IN PROGRESS
- TASK 3 (Epictetus power caveat) — not started
- TASK 4 (Figure 1) — not started
- TASK 5 (ModelLens dependency check) — DONE, folded into CLAUDE.md step 1
  (upstream _capture_activations fix still open)

Context: this repo (Stoic-Steering) currently frames itself around Stoic
philosophy. It's being repositioned around the real claim: does a behavior
installed in weight-space recruit the same circuit as one installed in
activation-space? Stoicism becomes the instrument, not the subject.

Read CLAUDE.md, README.md, results/README.md, docs/measurement-artifact.md,
and REPRODUCTION_PLAN.md before doing anything.

TASK 1 — Reframe audit (propose, don't write)
Produce docs/reframe-plan.md listing every place the repo leads with
philosophy where it should lead with intervention locus. For each: file,
line range, what the current framing claims, and a one-line note on what
the locus framing would say instead. Do NOT rewrite prose — I write all
external-facing text. Give me the map and the beats, not the paragraphs.

TASK 2 — Make the pipeline axis-agnostic
The corpus/trait is currently hardcoded to the three philosophers. Refactor
so a behavioral axis is a config object: corpus source, contrast pair
generator, dilemma set, adapter name. Target: adding a second axis
(sycophancy, via Chen et al. 2025 persona-vector trait pairs) is a config
file, not a code change. Do not break existing philosopher runs — all
current results must reproduce byte-identical where deterministic. Show me
the config schema before you refactor.

TASK 3 — Epictetus power caveat
Epictetus is null on the smallest corpus. Find every place the repo reports
this as a finding rather than an underpowered arm. List them. Propose the
caveat language for me to write. Add corpus sizes to the results table so
the power difference is visible.

TASK 4 — Figure 1 only
Extend make_figures.py to emit the side-by-side CAA vs LoRA circuit graph
from the exp12 sweep JSONs, using the content logit-diff metric. Node
color by role, edge weight by normalized effect, base circuit as shared
control panel. Skip figures 2 and 3 for now.

TASK 5 — ModelLens dependency check
exp12_circuit_analysis.py runs a local hotfix for the
_capture_activations closure-return bug. Confirm whether that fix is
upstream in ModelLens yet. If not, flag it prominently — I need it fixed
there before any of this is public.

CONSTRAINTS
- No claim gets retired or edited silently. Anything superseded stays
  logged with a supersedes-pointer, per the tracker schema.
- Nothing gets pre-registered or run in this session. This is
  restructuring and diagnosis only.
- dilemmas_v3 is still the next experiment and is unaffected by this work.
- Ask before any change that touches results JSONs.