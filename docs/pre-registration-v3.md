# Pre-registration — dilemmas_v3 behavioral eval

**Status, 2026-07-29 — split deliberately:**

- **§2 (the statistic) and §3–§5 (readings, control, invalidation): SIGNED AND
  FROZEN.** Changes from here are amendments, dated, with a reason.
- **§1's item set: FROZEN 2026-07-29** (see §7). Three of four cells meet the
  0.05 design target; `offtopic_plain` sits at 0.5623 and carries a documented
  imbalance inside the 0.10 validity limit. No item may change (R5).

The statistic was frozen *before* the item set was finalised, on purpose. Which
directional test is used does not depend on which 40 items end up in the set,
and choosing it after watching a cell fight its gate would look post-hoc — which
is exactly the failure this document exists to prevent. Nothing has been run
against v3 with any adapter.

---

## 1. What is being tested

Whether the LoRA decision shift is **reasoning** or **Senecan lexical echo**.
`ext_02`'s "on loan" is Senecan idiom nearly verbatim, so the observed shift
could be register reaching the choice through the option's wording rather than
through anything about the decision.

2×2 over 40 items, 10 per cell, stance-balanced 5/5 within every cell:

|  | plain wording | Stoic idiom |
|---|---|---|
| **Letters-core topic** | core_plain | core_idiom |
| **off-topic** | offtopic_plain | offtopic_idiom |

**Arms:** `lora_marcus_clean`, `lora_seneca_clean`, `lora_epictetus_clean`, plus
**two control adapters** (`lora_control_nonphil`, `lora_control_neutral`). The
controls are comparison arms, not sanity checks — see §4.

**Instrument:** the unchanged judge-free ruler (`eval_dilemmas`), one forward
pass per label order, both orders averaged. No judge, no generation, $0.

---

## 2. The directional statistic — the decision this document exists to fix

The sign test **cannot** be the primary statistic. On the v2 set it proved not
version-stable: under torch 2.8.0 against the pinned 2.5.1, Marcus moved from
27+/13− (p=0.038) to 24+/16− (p=0.268) because three items whose |Δ| sat inside
~4.3e-3 of fp16 drift changed sign. v3 is also 40 items. Using the same
instrument invites the same failure, and picking a tolerance after seeing a
result would be post-hoc.

**Primary statistic: paired t-test on Δ log-odds.**
- Log-odds because the analysis is about a probability shift, and because it is
  the scale that survives items sitting near the floor or ceiling — the v1
  failure mode the [0.2, 0.8] outlier band exists to prevent.
- Magnitude-based, so it is robust to the drift that breaks the sign test:
  every t-statistic in the v2 re-run reproduced to ~4 dp.
- Already computed in-repo as `overall_logodds` (`dilemmas.paired_stats`).

**Secondary, always reported, never used to rescue a null: ΔP paired t**, and a
**tolerance-aware sign test at tol = 0.01** — chosen now, and deliberately above
the observed 4.3e-3 drift ceiling with room to spare. Items dropped as ties by
that tolerance are reported by count and id.

**Planned comparisons — three, fixed in advance:**

| # | contrast | n | tests which reading |
|---|---|---:|---|
| 1 | topic marginal: core (both phrasings) vs off-topic | 20 vs 20 | topic proximity |
| 2 | phrasing marginal: idiom (both topics) vs plain | 20 vs 20 | lexical echo |
| 3 | `offtopic_plain` cell alone vs zero | 10 | the reasoning claim |

Marginals rather than four cell-wise tests, because n=10 per cell is
underpowered for a within-cell t (9 df). **Multiplicity: Holm correction across
the three planned comparisons, α = 0.05 family-wise.** No other comparison is
confirmatory; anything else is exploratory and labelled as such.

---

## 3. Pre-registered readings

Fixed before the run, per CLAUDE.md:

- **Topic proximity** — movement concentrated in the core-topic cells
  (comparison 1 significant, 2 not). The shift tracks how close the item is to
  what the corpus talks about.
- **Lexical echo** — movement tracks idiom across both topics (comparison 2
  significant). This is the confound; it would substantially weaken the decision
  claim, and the write-up says so.
- **Reasoning** — movement in `offtopic_plain` (comparison 3 significant): the
  adapter moves the choice on a topic the corpus barely covers, phrased without
  its idiom. **This is the strongest available reasoning claim** and the only
  reading that lets "LoRA reaches decisions" stand as stated.
- **Mixed** — comparisons 1 and 2 both significant. Report both; do not pick.

**Null branch, three-way** (never two-way): if an arm shows no movement anywhere,
classify as (a) replication of a genuine null, (b) vanish-with-clean-base, or
(c) vanish-with-mushy-base = inconclusive. **v3's calibration is what makes this
decidable**, and the corrected numbers still support it: three cells sit within
0.05 of P = 0.5, `offtopic_plain` at 0.5623, and **no item is outside
[0.2, 0.8]**. Branch (c) is about having no room to move — v1's 0.881 — and the
worst cell here retains headroom in both directions (highest item 0.727). So (c)
is ruled out, on the item band rather than on the cell means.

*Corrected 2026-07-29: an earlier revision of this clause claimed all four cells
were within 0.05. That was true when written and is not now.*

---

## 4. The control adapters' role

**Amended 2026-07-29 — two controls, not one.** They answer different questions
and are not substitutes:

| adapter | training text | isolates |
|---|---|---|
| `lora_control_nonphil` | external non-philosophical corpus, length-matched | content vs *any* fine-tuning |
| `lora_control_neutral` | the frozen pairs' anti-Stoic halves | *Stoic* content vs any strong normative content |

The neutral control is matched by construction on topic and generation process —
each `neutral_text` argues the same situation from a worldview that rejects
Stoicism — which makes it a tighter control than any external corpus on
everything except the axis it isolates.

**It is NOT length-matched, and cannot be.** The neutral texts run 2.2–3.1×
longer than the stoic ones (pooled mean 1436 vs 557 chars, barely overlapping
ranges) because the generator wrote fuller counterarguments than its sources.
Truncating to length would cut the conclusion, which is where the
Stoic-rejecting recommendation sits — removing the very contrast being tested.
Example count (53) and the recipe are held constant instead, and the ~2.6× token
excess is declared here in advance.

**How to read that asymmetry, fixed now:** a control with *more* training signal
that still fails to move decisions **strengthens** the content conclusion. A
control that *does* move is **ambiguous** between content and token count, and is
then followed by a token-matched variant (~21 examples, same total characters) —
run only if that ambiguity actually arises, not pre-emptively.

---

### 4.1 Original clause (unamended)

An arm's movement counts as evidence for the axis **only if it exceeds the
control adapter's movement on the same comparison.** Without that, "LoRA moves
decisions" is indistinguishable from "any fine-tuning on formal prose moves
decisions" — method and objective are already confounded (CAA is contrastive,
LoRA is continued pretraining), and the control is the only thing that separates
them.

If the control moves the instrument as much as the philosopher adapters, the
finding is about fine-tuning, not about Stoicism, and the locus claim narrows
accordingly. That outcome is registered here as a real possibility, not a
failure mode to be explained away.

---

## 5. Conditions that invalidate the run

- Base P(target) per cell drifting outside the **0.10 validity limit** at eval
  time, or any cell that met the 0.05 design target at freeze falling out of it
  → re-calibrate before interpreting anything. (Stated against the validity limit
  deliberately: `offtopic_plain` is already outside the 0.05 target by design, so
  a condition written against the target would fire on arrival and invalidate a
  run that the freeze decision explicitly accepted.)
- Base integrity check failing around the merges (must be 0.542 → 0.542, drift 0).
- Any item scoring outside [0.2, 0.8] on the base model at eval time → report it,
  and report the analysis with and without it.
- torch version differing from the run that produced the calibration → record it;
  the primary statistic is drift-robust but the tie counts are not.

---

## 6. What is NOT pre-registered here

The circuit sweep on v3 (roadmap step 4) is a separate pre-registration. It
reuses this item set — which is why stance balance is enforced *within* each
cell — but its metric and thresholds are its own.

---

## 7. Calibration state of the item set (open)

The set does **not** currently pass its gate, and the history matters because it
shows the gate working:

| date | overall | offtopic_plain | gate |
|---|---|---|---|
| 2026-07-21 | 0.4967 | 0.5163 (\|dev\| 0.016) | PASS |
| 2026-07-29 | 0.5102 | 0.5715 (\|dev\| 0.072) | **FAIL** |

The 2026-07-21 pass was an artifact. `offtopic_plain_10` scored 0.179 — outside
the [0.2, 0.8] item band — and was dragging the cell mean down. The cell's other
nine items average **0.5538**, which is |dev| 0.054 and fails on its own. Two
faults were cancelling: exactly the v1 failure shape, caught by the item-level
band rather than the cell mean.

Replacing the outlier (new item scores 0.727) removed the item-level fault and
exposed the cell-level one. With the nine fixed items summing to 4.984, item 10
would have to score ≤ 0.516 for the cell to pass — so **no well-behaved item can
centre this cell at 0.5.** The cell is biased, not the item.

**Confirmed construction fault:** a foil that argues from Stoic premises. The
retired `offtopic_plain_10` foil used impermanence ("pile-ons burn out in days")
and focus-on-what-you-control ("the help that counts happens off the timeline"),
making the foil arguably more Stoic than the target.

**Ruled out:** option-length asymmetry. Every item in the cell has a longer
nonstoic option (by 8–22 words), and `offtopic_plain_08` shares the retired
item's −22-word gap while scoring 0.694 versus 0.179. Length is not the driver.

### 7.1 Stopping rule for calibration iteration — SIGNED 2026-07-29

Calibrating to P ≈ 0.5 is, by construction, fitting the item set to
Llama-3.2-3B's preferences. That is the gate's purpose and it is legitimate
*while no adapter has been run*. But every additional round fits more closely to
this one base model's idiosyncrasies, so the budget is fixed now, before it is
convenient.

**R1 — Fault-first.** An item may be revised only if a named construction fault
and a **directional prediction** are recorded *before* re-scoring. No item may be
revised because its score is inconvenient. This converts each round into a test
of a mechanism rather than a search for a number.

**R2 — Budget.** At most **two further revision rounds per cell**, and at most
**five item revisions in total across the set**, counting from 2026-07-29. A
round is one re-score; every revision in a round is decided before that round's
scores are seen. Rounds already spent on `offtopic_plain`: 2 of 3.

**R3 — Two thresholds, both fixed now.**
- **Design target:** |cell mean − 0.5| ≤ 0.05 (the existing gate).
- **Hard validity limit:** |cell mean − 0.5| ≤ 0.10 **and** every item inside
  [0.2, 0.8].

A cell meeting the hard limit but not the target is **usable, with the imbalance
reported as a limitation.** The gate's 0.05 is a design target, not a validity
requirement: the analysis is on *deltas*, and a cell at 0.58 still has headroom
in both directions. What made v1 invalid was 0.881 — no room to move up — not
imbalance as such.

**R4 — Neither threshold may be widened.** If a cell exceeds the hard limit after
its budget is spent, the cell is not usable, and the pre-registered fallback is:
re-author that cell's *topic list* (not individual foils), or reduce the design
to a 1×2 on phrasing alone. Whichever is chosen is recorded here, with its
reason, before any adapter runs.

**R5 — Absolute.** No item may be revised, added, or removed after any adapter
has been run against v3. No exceptions, including "we noticed a bad item."

**R6 — Baseline as a confound check.** Because cells may end up at different
baseline levels, the analysis will test whether cell-level baseline P predicts
cell-level movement. If it does, baseline imbalance is confounded with the effect
and must be reported alongside every cell comparison.

**R7 — Report the trail.** The write-up states how many revision rounds each cell
took, the fault named at each round, and the resulting scores. A set that needed
three rounds is honest; a set that needed three rounds silently is not.

**Method rule (subsumed by R1):** do not iterate on a single item until the cell
mean comes out right. Repair the named construction fault, then re-score.

## Amendments

**A1, 2026-07-29 — §4 extended from one control to two.** Reason: the frozen
`neutral_pairs.json` already contains 53 anti-Stoic counterarguments per arm,
matched by construction on topic and generation process. They test a different
question than an external corpus does (Stoic content vs any normative content,
rather than content vs fine-tuning), cost one additional Colab run and no API
spend, and together the two controls bracket the question. Declared before either
adapter was trained and before any adapter touched v3.

**A2, 2026-07-29 — R2 budget exceeded by one revision, logged at the time.**
`offtopic_plain_08`'s topic was re-authored after its budget was spent, because
R4's named fallback (re-author the topic list) was triggered by a falsified
mechanism: three successive foils scored 0.694 → 0.788 → 0.800, monotonically
worse. Outcome: 0.694, no outliers, cell 0.5623. The round-4 prediction (< 0.6)
also failed, and its own falsification condition then said accept — so the set
was frozen with a documented imbalance rather than revised again.
