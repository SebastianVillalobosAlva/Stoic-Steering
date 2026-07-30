# Pre-registration — dilemmas_v4 (powered reasoning-vs-echo test)

**Status: SIGNED 2026-07-30, before any v4 item exists.**

Written after v3 returned, and deliberately before authoring, so the target
effect and the stopping rule cannot be chosen to fit items that already exist.
v3's result is final and unamended; this is a separate instrument.

---

## 1. Why v4 exists

v3 could not answer its own question. Every planned comparison failed its Holm
threshold for every arm, and the controls showed why: an adapter with **t = +0.08
overall movement** — one that provably does nothing — still produced a phrasing
marginal of p = 0.046 and a reasoning-cell p of 0.051. At 10 items per cell the
design emits p ≈ 0.05 from noise.

What v3 *did* show, descriptively, is a lopsided pattern in Seneca:

| | Δlog-odds |
|---|---|
| idiom-phrased items | **+0.2514** |
| plain-phrased items | **+0.0177** |
| off-topic **and** plain (`offtopic_plain`) | +0.0157 (t = 0.17, p = 0.87) |

That is the lexical-echo shape. It did not clear the pre-registered bar and is
**not** claimed as a finding. v4 exists to determine whether it is real.

---

## 2. The question v4 answers — an upper bound, not a detection

**This is the key design decision and it differs from v3.**

The reasoning effect appears to be near zero. No sample size can prove a null,
so "detect the reasoning effect" is the wrong target — at the observed spread
(per-item sd ≈ 0.30 in Δlog-odds) detecting an effect of +0.13 would need 55
items per cell, and +0.05 would need 379.

v4 instead asks: **is the plain/off-topic effect much smaller than the idiom
effect?** That is what separates echo from reasoning, and it is answerable.

**Primary quantity: the 95% confidence interval on the `offtopic_plain` cell's
mean Δlog-odds.**

| items/cell | CI half-width at sd = 0.30 |
|---|---|
| 10 (v3) | ±0.21 |
| **25 (v4)** | **±0.12** |
| 50 | ±0.085 |

At 25 per cell the interval excludes an idiom-sized effect (+0.25). That is the
decisive comparison.

---

## 3. Design

**100 items: 25 per cell**, same 2×2 (topic × phrasing), stance-balanced within
each cell so the set still serves the circuit sweep.

**The 40 v3 items are carried over** — they are calibrated, frozen and already
scored. 60 new items are authored. **Consequence, declared now:** v3 and v4
results are *not* independent; v4 supersedes v3 rather than replicating it, and
the write-up reports one result, not two agreeing ones.

**Statistic, unchanged from v3** (§2 of that document, still in force): paired
t on Δlog-odds primary; ΔP and a sign test at tol = 0.01 secondary; Holm across
the same three planned comparisons at α = 0.05 family-wise.

**Arms, unchanged:** three philosopher adapters plus both controls. An arm's
movement counts only if it exceeds both controls on the same comparison.

---

## 4. Pre-registered readings

Fixed before any item is written.

- **ECHO** — the `offtopic_plain` CI excludes +0.15, *and* the phrasing marginal
  is significant after Holm with idiom > plain. The decision claim is then about
  register, and story beat 3 narrows to say so explicitly.
- **REASONING** — the `offtopic_plain` mean is significantly above zero after
  Holm. The strongest available claim; "LoRA reaches decisions" stands as
  written.
- **BOUNDED NULL** — the `offtopic_plain` CI contains zero *and* excludes +0.15.
  Reported as: any reasoning effect is smaller than half the observed idiom
  effect. This is a real result, not a failure, and is the outcome the design is
  powered for.
- **STILL UNDERPOWERED** — the CI contains both zero and +0.25. The design failed
  again; report that plainly and do not iterate on item count a third time
  without a different instrument.

**Null branch stays three-way.** A whole-set null is classified as replication /
vanish-with-clean-base / vanish-with-mushy-base. The calibration gate rules out
the third in advance, as in v3.

---

## 5. Stopping rule for calibration (carried over, tightened)

R1–R7 of `docs/pre-registration-v3.md` §7.1 apply unchanged, with three
additions learned from v3:

**R8 — the budget scales with the set, not per cell.** At most **12 item
revisions total** across the 60 new items (v3 spent 6 on 40). Rounds still
require a named fault and a directional prediction recorded before scoring.

**R9 — construction rules, from v3's failures.** Every new item must satisfy all
of these, checked at authoring rather than discovered at calibration:
- the foil argues from a **competing good**, never a competing convenience —
  convenience foils are easy to reject and push P(target) up
- the foil must not **contradict a fact the situation stipulates**
- the situation must not **pre-decide the answer** by ruling out one option
  before the options are offered
- the target must not be an **unambiguous ethical duty** (harm-reporting,
  whistleblowing) — base-model priors dominate those
- both options within **±10 words** of each other

**R10 — topics with a known prior are excluded up front.** v3 burned three
rounds on a housing-office whistleblowing item before concluding the topic, not
the wording, was unsuitable. Any topic where the base model has an obvious prior
is rejected at authoring.

---

## 6. Conditions that invalidate the run

- Any cell outside the **0.10 validity limit**, or a cell that met the 0.05
  design target at freeze falling out of it.
- Base integrity failing around the merges (v2 ruler must return 0.541609,
  drift 0).
- Any item outside [0.2, 0.8] at eval time.
- torch version differing from the calibration run — record it; the primary
  statistic is drift-robust, the tie counts are not.

---

## 7. What is NOT pre-registered here

The circuit sweep (roadmap step 4) and the token-matched neutral control
(v3 §4 amendment A1, triggered by that control moving −0.2618) are separate
pre-registrations. Both reuse this item set; neither is decided here.

---

## Amendments

*(none — append dated entries below, with reasons)*
