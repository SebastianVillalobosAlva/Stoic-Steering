# Follow-up integrity audit: figures, stage scripts, build plan

**Started:** 2026-11-01. **Base:** `main` at `fad5fcb`. **Mode:** flag, never fix. This file is the only
change to the repo, and it is uncommitted.

This covers what the 2026-10-16 audit left unchecked: the figures, `make_figures.py`, the stage scripts and
the build plan. Reading those led to several checks of the headline claims, including the v4 result. It also
corrects errors in the first audit (see the end).

Every item is a candidate for Sebastian to adjudicate. Numbers marked **exploratory** are my own
recalculations from checked-in JSONs. They were not pre-registered, so they are reasons to look again, not
findings.

---

## Severity 1: credibility-threatening

### [S1-1] LoRA style/content evidence comes from the old adapters, not the clean ones

- **Where:** `README.md:5`; `fig_three_depths.png`; `results/README.md:127-128`; `docs/measurement-artifact.md:65`; `CLAUDE.md:290`
- **Claim:** "LoRA ... also shifts judge-scored style and content in a single eval"; "LoRA from the frozen merged-adapter judge evals."
- **Check:** claim traceability; confound disclosure.
- **Why:** those panels read `data/reference/judges/eval_*_20260527_*.json`, which are Exp 4 (legacy tracker: "Date: May 27, 2026", same numbers). The clean adapters are dated 2026-07-01. The legacy tracker says twice that content was "not re-scored on clean adapters" (L14, L637), and that receipt did not survive the rebuild. The figure therefore puts old-adapter judge scores beside clean-adapter decisions as one intervention.
- **Also:** those JSONs record no judge model and no decoding settings. "Symmetric decoding" and "robust to judge choice" (`results/README.md:132`) can't be traced from this repo.
- **Defensible:** "Older adapters, trained on pre-cleaning data, moved judge-scored style and content; the clean adapters have not been judge-scored."

### [S1-2] "Byte-identical" is false, and it sits in the artifact writeup

- **Where:** `docs/measurement-artifact.md:44-45`; `results/README.md:56-58`
- **Claim:** "At coeff 0.11 the steered greedy output is byte-identical to baseline (greedy text only visibly changes at coeff ≥ 1.0), so the residual is judge noise."
- **Check:** claim vs data.
- **Why:** in `content_20260704_192154.json`, identical pairs are **2/12 (Marcus), 5/12 (Seneca), 6/12 (Epictetus)**. The rest diverge mid-text. All three matched-greedy means are negative with `content_ci_excludes_zero: true`, the repo's own ±1σ rule (`judge.py`). So the premise behind "judge noise" fails for most pairs.
- **What survives:** the core correction. The +0.41 to +0.77 effect is gone under matched decoding: every greedy arm is negative, and the sampled arms are within noise of zero (Marcus +0.075 ± 0.165). Only the mechanism sentence fails.
- **Note:** this is story beat 1, so it is the worst place for a checkable error.

### [S1-3] The v4 phrasing effect leans on the items where the pattern was first seen

- **Where:** `README.md:3,5,7-16`; `docs/claim-revision-v4.md`; `CLAUDE.md` Status
- **Claim:** a pre-registered powered test shows echo: "idiom +0.099 / plain −0.083."
- **Check:** multiple comparisons; n discipline.
- **Why:** 40 of the 100 items are the v3 items that suggested the pattern. Splitting them out (**exploratory**, Δlog-odds):

| Seneca | idiom | plain | difference | t-test p |
|---|---|---|---|---|
| Carried-over 40 | +0.251 | +0.018 | +0.234 | 0.041 |
| New 60 | **−0.002** | −0.151 | +0.148 | **0.117** |

- On the new items Seneca's contrast does not exceed `control_neutral` (+0.148 vs +0.278).
- The +0.099 idiom movement comes entirely from the carried-over items. Neither marginal differs from zero alone (idiom p=0.074, plain p=0.100).
- The result is fragile: leaving out one item pushes p above the Holm threshold (0.0167) in 38 of 100 cases. Mann-Whitney gives p=0.034, which also misses it.
- **In fairness:** the protocol was followed. The ECHO branch's conditions are met as registered, and `pre-registration-v4.md` §3 declares the non-independence. The problem is that the public surfaces don't carry that caveat.
- **Defensible:** "Same direction on new items but not significant alone; the combined test includes the items that generated the hypothesis."

### [S1-4] The controls are probably not dose-matched to the philosopher adapters

- **Where:** `scripts/prep_control_lora.py:11`; `docs/pre-registration-v3.md:116,129`; `CLAUDE.md:194,218,273`
- **Claim:** "Same count. 53 examples, exactly as each philosopher adapter had."
- **Check:** confound disclosure.
- **Why:** three legacy documents say the clean adapters trained on clean chunked text (tracker L610, legacy `CLAUDE.md:18`, `AUDIT.md:13`). The controls trained on 53 examples (`checkpoint-81`, confirmed).

| Adapter | Examples | Chars | × nonphil control | Steps (batch 2, assumed) |
|---|---|---|---|---|
| Seneca | 540 | 573,041 | 19.1 | 810 |
| Marcus | 437 | 315,011 | 10.5 | 655 |
| Epictetus | 123 | 38,923 | 1.3 | 184 |
| control_nonphil | 53 | 29,951 | 1.0 | 81 (confirmed) |
| control_neutral | 53 | 75,907 | 2.5 | 81 (confirmed) |

- **Consequence:** "exceeds both controls" is load-bearing in the v4 verdict, and an under-dosed control makes it easier to pass.
- **Unresolved:** the clean adapters have no training log in this repo, so this rests on those documents. The training data needs confirming from the original Colab run. If they did train on 53 examples, the corpus-size story (S2-7) fails instead, so one of the two claims is wrong either way.

### [S1-5] "CAA is a no-op" has no dose qualifier, and CAA was never run where the headline now rests

- **Where:** `README.md:3` (no coefficient, wording applied in the follow-up); `README.md:5`
- **Check:** overclaim shape; confound disclosure.
- **Why, in three parts:**
  1. **Perturbation size (exploratory):** 0.11 × |v| is **2.6-4.2% of mean per-token MLP output norm** (8 dilemmas, MLP output rather than the residual stream). Parity would need a coefficient of about 2.6-4.3. No LoRA-equivalent dose has been defined, so locus and strength are confounded.
  2. **Selection:** layers were chosen on Exp 8/9 judge scores, and `LEGACY.md` says those runs used the asymmetric decoding. Coefficient 0.11 comes from the earlier style-inflated sweep (tracker L116-118, L369). The null is measured at a configuration picked by the broken instrument. The repo's README doesn't disclose this.
  3. **No CAA arm on v3/v4:** `behavioral_eval.py` has none. The headline is now a phrasing contrast, and CAA was never tested on it. "Only LoRA" is a v2 comparison.
- **Defensible:** "At coefficient 0.11, a perturbation of about 3-4% of MLP output norm, chosen by the earlier pipeline."

---

## Severity 2: accuracy

**[S2-1] `README.md:26`: "A coefficient sweep at the decision level has not been done."** Legacy Exp 10c did one, to 1.5, for **Epictetus only** (ΔP −0.011, t=−1.16). The raw output isn't in this repo. The legacy "at any coefficient" and `CLAUDE.md:224` "flat to 1.5" overgeneralize in the other direction. Marcus and Seneca were not swept.

**[S2-2] `CLAUDE.md:309`: "Marcus's v2 passivity prior did not replicate (−0.007...)."** −0.007 is Marcus's **v3 overall ΔP** (−0.0069), so it is neither v4 nor a stance test. **Exploratory** v4 stance contrast: Marcus accepting +0.058 / active −0.149, **p≈8e-7**; nonphil control +0.106 difference, p=0.035. So the claim looks unsupported and may be backwards. Meanwhile `README.md:45`, `results/README.md:101,305` and the figure still assert the prior. The repo disagrees with itself. Stance was not a planned v4 comparison, so this needs its own pre-registration.

**[S2-3] Seneca's stance contrast is as large as the phrasing one (exploratory).** Accepting +0.099 / active −0.090, difference 0.189, p=0.012, against phrasing 0.183, p=0.015. The echo reading doesn't mention it.

**[S2-4] `README.md:39`; `CLAUDE.md:138`: the Marcus cell and the control.** Per `exceeds_control`, Marcus's −0.102 **beats** nonphil (−0.094) and fails only against neutral (−0.402). "The reason Marcus's cell fails" names the wrong control. In the README sentence, −0.036 is the overall number, not the cell. That wording came from my first audit's suggested text.

**[S2-5] `README.md:142`: the stability sweep line.** The build plan's Tier 2 gate (L137) says **cancel** on pure echo. "Sycophancy only" is the CLAUDE.md step 7 amendment (`4255292`). The plan is gitignored, so readers can't check it. The gate was also written against v3, which came back inconclusive, and the ECHO reading applies to Seneca only. My "Case B" call conflated the two sources.

**[S2-6] `CLAUDE.md:293`: "Seneca's v2 shift (+0.1345 log-odds, t=2.34 on 40 items)."** That is **v3**. The v2 figure is +0.308, t=2.39. `docs/claim-revision-v4.md:26` has the same number without a label. v4 shares no v2-specific retest: it carries v3's items, not v2's.

**[S2-7] `README.md:45,144`: Epictetus as "underpowered."** Two issues:
- The README's 123 vs 540 chunks (4.4×) understates the gap: by characters it is **14.7×**.
- A control with *less* text moved more than Epictetus (53/100 ties vs 16), which is awkward for the size explanation. Corpus size also orders Seneca > Marcus > Epictetus, the same order as the decision shift and circuit ranking (`results/README.md:168`), and that isn't noted at the circuit claim.

**[S2-8] Labels baked into the figures.** The captions added in the fixes sit under PNGs whose pixels still say:
- `fig_lora_decision_shift`: "LoRA reaches the decision layer — with structure", "(passivity prior)"
- "Seneca moves both stance buckets", though the figure's own legend marks Seneca-active as n.s. (t=1.16). Also 6 unadjusted bucket tests.

`results/README.md:102` has a similar "both buckets positive ✓", and the Stage 4 gate only requires positive point estimates.

**[S2-9] Caption over-reach (`README.md`, `results/README.md`).** "This movement is lexical echo" sits under figures that include Marcus, whose phrasing contrast is not significant (p=0.078). Echo is a Seneca-only reading.

**[S2-10] The biggest v4 mover isn't in the README.** `control_neutral` moved −0.205 (t=−2.47, p=0.015), larger than any philosopher. Pre-registration v3 §4 says a moving control is ambiguous and is "followed by a token-matched variant." It hasn't been run and isn't on the roadmap. Meanwhile the tagline's "LoRA moves forced-choice decisions" sits beside Seneca's overall t=0.21 at n=100.

**[S2-11] "Largest circuit modifier": the claim holds, its illustration doesn't.** **Exploratory:** in raw units Seneca still leads (median 1.65 vs 0.86, 1.92×, larger on 9/10). But node shift is normalized by total effect, and it tracks how much the total effect shrank (Spearman 0.73). emot_03's 2.82 comes with a 7× shrink; in raw units it is 1.73, not an outlier. No significance test was run, and Seneca had about 1.8× Marcus's training text. Without the outliers the lead is 2.13×, not the "2.2×" in the figure subtitle.

---

## Severity 3: hygiene

- **`n=1-per-stance`** (`CLAUDE.md:140,198`; `REPRODUCTION_PLAN.md`): Exp 12c has 5 per stance. The n=1 label fits 12/12b, and the CAA circuit evidence is still only those 2 items.
- **`docs/claim-revision-v4.md:26-27`, CIs:** these match a z interval. The t interval is [−0.201, +0.061]. The conclusion is unchanged, and the CIs aren't stored in the JSON.
- **`v4_eval` JSON** records no torch version, though pre-registration v4 §6 says to record it.
- **`docs/decisions.md:17`, "The harness runs":** only the import was fixed and checked; nobody has run it end to end.
- **`CLAUDE.md:265`** cites `stoic/calibrate.py`; the module is `stoic/calibration.py`.
- **`make_figures.py:241-244`** picks inputs with `sorted(glob)[-1]`, so a new run silently redraws the headline figure. `content.py:122` takes `[0]`, the oldest. Current PNGs (Jul 11) predate the Jul 17 Stage 4 JSON; its values match.

---

## Checked, no flag found

- Figure values match their JSONs: all four figures' numbers agree with the sources I traced.
- v4 validity gates: no item outside [0.2, 0.8], all cells within 0.10, drift 0.
- Stage scripts: `verify.py` and `adapters.py` produce the keys the JSONs contain; decoding-free paths confirmed.
- Eval items remain separate from training text.
- `pytest`: 87 passed.

---

## Corrections to the 2026-10-16 audit

- It read the "byte-identical" claim and never compared the outputs (S1-2).
- It called the v4 numbers "line-for-line" exact, but I hadn't computed the CIs.
- It took "merged-adapter evals" at face value (S1-1).
- It repeated "no coefficient sweep" (S2-1).
- It offered the Marcus-cell wording now in the README (S2-4).
- It marked the Epictetus corpus caveat as well disclosed, using chunk counts only (S2-7).
- Its "claims I could defend" need new caveats: #1 from S1-2, #2 and #3 from S1-5, #4 from S1-3 and S1-4.

---

## Scope

**Read:** `scripts/make_figures.py`, `scripts/exp12_reanalysis_modifier.py`, `stoic/stages/verify.py`, `stoic/stages/adapters.py`, the build plan, and all 4 PNGs in `results/figures/`.
**Queried:** v3/v4/Stage 3/style/Exp 12 JSONs, `data/reference/judges/`, `data/generated/dilemmas_v4_candidates.json`, training metadata, git history, the frozen steering vectors, and parts of `../stoic-llm-legacy`.
**Ran:** `pytest`, plus one read-only base-model load for the perturbation ratio.
**Not read:** `stages/{passb,calibrate}.py`, `corpus.py`, `pairs.py`, `calibration.py`, `axis.py`, test bodies, `exp12_sweep.py` in full, the Exp 12 per-item PNGs, and the rest of the legacy repo.

Absence of a flag covers only what was read. This is not a pass.
