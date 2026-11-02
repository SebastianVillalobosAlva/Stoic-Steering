# Resolution of the 2026-11-01 follow-up audit

**Date:** 2026-11-02. **HEAD:** `fad5fcb`, unchanged since the audit, no commits in between.
**Mode:** read-only. This file is the only repo change. `audit/2026-11-01_followup-audit.md` is still uncommitted.
**Scratch scripts:** `/tmp/stoic_resolution/`.

This verifies the audit. It does not adjudicate flags or draft replacement wording. Numbers marked
**EXPLORATORY** were not pre-registered.

---

## 1. Summary of every flag

| Flag | Status | One line |
|---|---|---|
| S1-1 | CONFIRMED | LoRA style/content panels are Exp 4 old adapters (May 27). No judge eval of the clean adapters was found. |
| S1-2 | CONFIRMED, wording corrected | 2/5/6 of 12 identical. "Most pairs" is 23 of 36 overall; Epictetus is exactly half. |
| S1-3 | CONFIRMED | All numbers reproduce. |
| S1-4 | CORRECTED, partly UNRESOLVED | "53 examples" is contradicted by every record. But the audit's dose table and "under-dosed" were wrong: `GRAD_ACCUM = 8` changes the comparison. No direct training log exists. |
| S1-5 | CORRECTED in part | Layer selection confirmed. The audit's stated origin of 0.11 is not supported: no record gives a reason. No CAA arm on v3/v4 confirmed. Ratio not recomputed, as instructed. |
| S2-1 | CONFIRMED, raw output UNRESOLVED | Epictetus only, 5 coefficients. No raw output found; the script only prints. |
| S2-2 | CONFIRMED | −0.007 is v3 overall ΔP. Stance numbers reproduce. |
| S2-3 | CONFIRMED | Reproduces. |
| S2-4 | CONFIRMED | Marcus beats nonphil and fails only against neutral. |
| S2-5 | CONFIRMED | Plan says cancel; CLAUDE.md says sycophancy only. |
| S2-6 | CONFIRMED | +0.1345/t=2.34 is v3. No item text is shared between v2 and v4. |
| S2-7 | CORRECTED in part | 14.7× by chars holds. The "control with less text moved more" point is wrong as evidence: that control had more weight updates. |
| S2-8 | CONFIRMED | Strings at `make_figures.py:115-117`. |
| S2-9 | CONFIRMED | Marcus phrasing p=0.0776. |
| S2-10 | CORRECTED | −0.205 is Δlog-odds, not ΔP. ΔP is −0.0408. It is still the largest mover on both scales. |
| S2-11 | CONFIRMED | 2.13× should be 2.14×. |
| S3 n=1-per-stance | CONFIRMED | |
| S3 CIs | CORRECTED | The doc's CIs match a t with df≈99, not a z interval. The correct t(24) interval stands. |
| S3 v4 JSON torch | CONFIRMED | No environment key. |
| S3 "harness runs" | CONFIRMED | No end-to-end run recorded. |
| S3 `calibrate.py` | CONFIRMED | Also at `docs/naming-cleanup.md:52`. |
| S3 figure globs | CONFIRMED | |
| "Eval items separate from training" | CONFIRMED for v4 | The audit inherited this untested. Now checked: no 8-word verbatim overlap between v4 options and clean training chunks. |

New items found while checking (details in Part 1, item 7):

| ID | Status | One line |
|---|---|---|
| N-1 | NEW | Colab script says recipe is "IDENTICAL TO EXP 4", but Exp 4 ran with no gradient accumulation. |
| N-2 | NEW | Controls did not use the clean adapters' recipe, contrary to `docs/pre-registration-v3.md:129`. |
| N-3 | NEW | `stoic/lora.py` trains from pair text with no accumulation, so it can't reproduce the clean adapters. |
| N-4 | NEW | "Predicted before measured" is still live with no receipt in this repo. The first audit missed it. |
| N-5 | NEW | First audit's "LoRA training sets no seed" was wrong. |
| N-6 | NEW | First audit overstated the retirement receipts as all present. |
| N-7 | NOTE | v4 items are partly paired: 17 of 50 twin ids share a situation. |

---

## 2. Part 1: open checks

### Item 1. S1-4: what the clean adapters trained on

**Status: the "53 examples" claim is CONTRADICTED. The audit's table is CORRECTED. Direct confirmation is UNRESOLVED.**

**Records found:**

| Source | Says |
|---|---|
| `../stoic-llm-legacy/scripts/lora-train-colab.py:14,51-62,133-134` | Trains on "clean chunked text per philosopher" |
| `../stoic-llm-legacy/data/lora_training/README.md` | "Clean LoRA trains on `data/chunked/`": Marcus 437, Seneca 540, Epictetus 123 |
| `Stoic LLM - Results Tracker.md:610` | Same three counts |
| legacy `CLAUDE.md:18`, `AUDIT.md:13`, `AUDIT_REPORT.md:54` | Clean chunked text; the Colab script is what produced them |
| `scripts/prep_control_lora.py:11` | "53 examples, exactly as each philosopher adapter had" (added in `8915d07`, 2026-07-29) |

Legacy chunks and frozen `data/reference/chunked/` are identical for all three authors.

**Why no log exists:** the script sets `save_strategy="no"` (line 156), so the clean adapter folders have no
`trainer_state.json` or checkpoints. `adapter_config.json` records only r, alpha, targets and peft 0.19.1.
The legacy repo has no notebook.

**Indirect evidence (EXPLORATORY):**
- **Save timing.** Adapter mtimes follow the script's author order: Marcus 01:03:56, Seneca 01:23:20, Epictetus 01:28:00. Seneca took 1,164 s and Epictetus 280 s, a ratio of 4.16. Chunk counts predict 4.35; equal example counts predict about 1. This assumes mtimes reflect save times.
- **LoRA weight size (mean ‖BA‖_F).** Clean Seneca 0.190, Marcus 0.176, Epictetus 0.089. Controls at 81 updates: 0.131 and 0.137. Exp 4 Epictetus at 84 updates: 0.122; Exp 4 Marcus and Seneca at 150 updates: 0.163 and 0.159. Epictetus sitting below the controls fits about 24 updates, and the 2× spread among clean adapters doesn't fit equal training.

**Corrected dose table.** Clean rows assume the script ran as written.

| Adapter | Examples | Chars | Tokens kept (≤512) | Micro-batches | Weight updates | Effective batch | Seed |
|---|---|---|---|---|---|---|---|
| Seneca | 540 | 573,041 | 130,613 | 810 | 99-102 | 16 | 0 |
| Marcus | 437 | 315,011 | 72,024 | 657 | 81-84 | 16 | 0 |
| Epictetus | 123 | 38,923 | 9,314 | 186 | 21-24 | 16 | 0 |
| control_nonphil | 53 | 29,951 | 7,253 | 81 | 81 (recorded) | 2 | 42 (recorded) |
| control_neutral | 53 | 75,907 | 14,556 | 81 | 81 (recorded) | 2 | 42 (recorded) |

The update ranges depend on how the Trainer handles the last partial accumulation in each epoch. The
transformers version for that run isn't recorded.

**Corrections to the audit:**
- Its step column counted micro-batches, not weight updates.
- "An under-dosed control makes it easier to pass" holds by text (10-19× for Marcus and Seneca). By weight updates the controls are about level with Marcus, modestly below Seneca, and about 3.4-3.9× above Epictetus. Whether the controls are under-dosed depends on how dose is defined.
- Its "one of the two claims is wrong either way" framing still holds.

**What would settle it, in the Colab run or its notebook revision history:**
- Output from line 134: `N clean chunks`
- The Trainer banner: `Num examples`, `Gradient Accumulation steps`, `Total optimization steps`
- Loss logs every 10 steps, whose count gives the update total
- Drive: `stoic-llm/data/chunked/{marcus,seneca,epictetus}_clean_chunks.json`. Compare length and text with the frozen chunks; those filenames don't exist locally

### Item 2. S1-2: identical pairs

**Status: CONFIRMED, with wording corrections.** Sources: `content_20260704_192154.json` (`steered_outputs`, `baseline_outputs`), `style_20260705_212411.json` (`greedy.n_identical_pairs`, whose `greedy_source` is that same content file).

| Author | Identical, content JSON | `n_identical_pairs`, style JSON | Greedy content ±1σ | Excludes 0? | Greedy style ±1σ | Excludes 0? | `survives_2sigma` |
|---|---|---|---|---|---|---|---|
| Marcus | 2/12 | 2 | −0.175 ± 0.054 | Yes | −0.150 ± 0.181 | No | False |
| Seneca | 5/12 | 5 | −0.117 ± 0.054 | Yes | −0.100 ± 0.091 | Yes | False |
| Epictetus | 6/12 | 6 | −0.117 ± 0.080 | Yes | +0.050 ± 0.126 | No | False |

Corrections:
- "Most pairs" is 23 of 36 overall (Marcus 10/12, Seneca 7/12, Epictetus 6/12).
- "Diverge mid-text" is too general. First differences range from character 91 to 495, and two pairs differ only at character 461 of 463.
- "So the residual is judge noise" appears only at `docs/measurement-artifact.md:45`. `results/README.md:56-58` makes the byte-identical claim but not that one.
- The audit's "negative" applies to content. Greedy style for Epictetus is positive (+0.050, within noise).
- `results/README.md:86` also says "greedy output barely changes below coeff ~1.0". The audit missed this location.

### Item 3. S1-1: inputs to `fig_three_depths.png`

**Status: CONFIRMED.** Selection logic is at `scripts/make_figures.py:241-246`.

| Panel | Arm | File | Date | Adapters / condition |
|---|---|---|---|---|
| Style | CAA | `results/style_validation/style_20260705_212411.json` | 2026-07-05 | Canonical vectors, greedy arm |
| Style | LoRA | `data/reference/judges/eval_{marcus_aurelius,seneca,epictetus}_20260527_*.json` | 2026-05-27 | **Old Exp 4 adapters** |
| Content | CAA | `results/stage3_content_judge/content_20260704_192154.json` | 2026-07-04 | Canonical vectors, greedy |
| Content | LoRA | Same `eval_*_20260527_*.json` | 2026-05-27 | **Old Exp 4 adapters** |
| Decision | CAA | `results/stage2_steering/steering_20260703_132438.json` | 2026-07-03 | Canonical vectors, v2 items |
| Decision | LoRA | Latest of `results/stage4_lora_dilemmas/` | Jul 5 at render, Jul 17 now | `lora_*_clean` per JSON `adapter` key |

**Evidence for the old adapters:**
- The JSONs say only `{'model_size': '3B', 'method': 'lora', 'epochs': 3}`, with no adapter path.
- They're byte-identical to the legacy `results/judges/` copies.
- `../stoic-llm-legacy/stoic_llm/lora/runner.py:12,32` resolves `models/{size}/{author}`. `LEGACY.md:176-178` and `AUDIT_REPORT.md:29` name these as the old contaminated-data adapters.

**Judge:** not recorded in the JSONs. `run_lora_eval.py` calls `StoicJudge()`, whose default is `claude-sonnet-4-20250514` (`stoic_llm/eval/judge.py:49,60`). The CAA panels use `gemini-2.5-flash`.

**Decoding:** the audit said symmetric decoding can't be traced from this repo, which is true. It can be traced in the legacy repo (`LEGACY.md:172-177`).

**Any judge eval of the clean adapters?** None found. Legacy `results/judges/` filenames stop at 20260527, about five weeks before the clean adapters. Repo Stage 3 and style files are CAA only. A scan of 116 JSONs under both repos' `results/` and `data/` found only judge evals from before the clean adapters existed, plus CAA sweeps. I opened one sweep hit, and there "lora" matched inside "exploration". Tracker L14 and L637 say "not re-scored on clean adapters". Tracker L13 lists the Exp 4 style result under a column headed "clean data".

### Item 4. S2-1: raw Exp 10c output

**Scope: CONFIRMED. Raw output: UNRESOLVED.**
- **What was run:** Epictetus only, L8, coefficients 0.11, 0.2, 0.4, 0.8, 1.5 (`../stoic-llm-legacy/scripts/run_dilemma_eval.py:40-46`, table at tracker L559-565).
- **No raw output found:** the script prints `summarize_sweep` and never calls `save_results`. No file contains `by_coeff`, and neither dilemma JSON has a sweep key. The only record is the tracker table.
- **What would settle it:** a saved terminal log, or re-running, which is Draft (a).

### Item 5. S1-5: where 0.11 and the layers came from

**Layers: CONFIRMED.** L26, L4 and L8 are the `optimal` layers in `results/sweeps/full_{marcus_aurelius,seneca,epictetus}_2026061*.json`. They were then ranked as candidates in the Exp 9 seed eval (`scripts/run_seed_eval.py:21-28`, tracker L455-467). `LEGACY.md:45-46` says `_run_steered` was used by "all layer/coeff sweeps" and Exp 8 and 9 with asymmetric decoding. I checked those statements, not the runtime path itself.

**Coefficient: audit CORRECTED.** The audit said 0.11 "comes from the earlier style-inflated sweep". No record says that.

| Source | What it shows |
|---|---|
| `stoic_llm/config.py:41` | `COEFFICIENT = 0.11`, present since `4c10177` (2026-05-17), no rationale |
| Tracker L369 | "Coefficient fixed at 0.11", no reason given |
| `scripts/run_seed_eval.py:28` | Hard-coded 0.11 |
| Jun 10-11 sweeps, the ones matching the canonical layers | Optimal coefficients 0.2, 0.2, 0.15, **not** 0.11 |
| May 30 sweeps | 0.11 optimal for Marcus and Epictetus, 0.15 for Seneca (content selection, not style aggregate) |

The reason for 0.11 is **UNRESOLVED**. The audit's broader point stands on the records above: every sweep and seed eval involved used the decoding `LEGACY.md` calls asymmetric.

**CAA on v3/v4: CONFIRMED none.** `per_arm` keys in both JSONs are the 3 philosophers plus 2 controls. `scripts/behavioral_eval.py` never passes `steer=`. No `.axis_snap` v3 or v4 eval log mentions CAA or steering.

The perturbation ratio was not recomputed, as instructed.

### Item 6. Independent recomputation (EXPLORATORY)

Fresh script at `/tmp/stoic_resolution/recompute.py`, using arrays and scipy rather than the audit's code.
I recomputed the S1-3 split, leave-one-out and Mann-Whitney; the S2-2 and S2-3 stance contrasts; S2-10; and S2-11.

**Disagreements only:**

| Item | Audit | Recomputed | Size |
|---|---|---|---|
| S2-10 | "−0.205" with the arm's movement, unlabeled scale | −0.2047 is Δlog-odds (t=−2.473, p=0.0151). ΔP is **−0.0408** (t=−2.463, p=0.0155) | Scale label |
| S2-11 no-outlier lead | 2.13× | 2.14× (0.270 / 0.126) | 0.01 |

Everything else agrees to the precision the audit reported, including 38/100 leave-one-out exceedances and Spearman 0.73.

### Item 7. Errors in the audit and earlier audit

**In the 2026-11-01 audit:**

| # | Where | Error | Correct |
|---|---|---|---|
| 1 | S1-4 table | Steps 655 and 184 mix rounding | 657 and 186 by ceiling, as you noted. These are micro-batches. Updates are 81-84 and 21-24 |
| 2 | S1-4 | "Under-dosed" stated without defining dose | See Item 1 |
| 3 | S1-4 | Didn't read `lora-train-colab.py` | It is the main record |
| 4 | S1-5 | Origin of 0.11 | See Item 5 |
| 5 | S2-7 | Control "with less text moved more" is "awkward" for the size explanation | That control had about 3.4-3.9× more weight updates than Epictetus, assuming the script ran as written, so the comparison doesn't show what it implied |
| 6 | S2-10 | Unlabeled scale | See Item 6 |
| 7 | S3 CIs | "These match a z interval" | They match t with df≈99 (1.984): Seneca [−0.196, +0.056], Marcus [−0.179, −0.025]. z gives [−0.194, +0.054] and [−0.178, −0.026]. Correct t(24): [−0.201, +0.061] and [−0.182, −0.022] |
| 8 | S1-2 | "Most pairs", "mid-text", quote attribution | See Item 2 |
| 9 | S1-1 | "Dated 2026-07-01" | Local mtimes say Jul 1; `AUDIT_REPORT.md:29` says June 30. The conclusion is unchanged |

The 34 line references I checked all resolve to the cited text. `REPRODUCTION_PLAN.md:148` wraps onto the next line.

**New items:**
- **N-1.** `lora-train-colab.py:8-12` says "RECIPE IS HELD IDENTICAL TO EXP 4", and tracker L610 says "the only variable vs the old adapters is clean data". But line 75 sets `GRAD_ACCUM = 8`, while the Exp 4 checkpoints record accumulation 1 and seed 42. So clean vs old is not a data-only change if the script ran as written.
- **N-2.** Recipes differ between the clean adapters (script) and the controls (`training_args.bin`):

| Setting | Exp 4 adapters | Clean adapters (script) | Controls (recorded) |
|---|---|---|---|
| Batch / accumulation | 2 / 1 | 2 / 8 | 2 / 1 |
| fp16 | True | True | False |
| Seed | 42 | 0 | 42 |
| Padding and labels | Not checked | `max_length`, labels copied with no −100 masking (lines 114-116) | Consistent with `stoic/lora.py:107-127`: dynamic collator; inferred from matching args and the command printed at `prep_control_lora.py:488-491` |

- **N-3.** `stoic/lora.py:62-76` builds training data from pair `stoic_text`, and `train()` (lines 79-128) has no accumulation. Its docstring says "canonical recipe". This path can't reproduce the clean adapters, which bears on Pass B.
- **N-4.** "Predicted before measured" is listed as retired in your first audit brief. It has no retirement note in this repo and is still live: see the Part 2 tables. The first audit never reported on it.
- **N-5.** The first audit said LoRA training "sets no seed". The Colab script sets 0 and the controls' `training_args.bin` records 42. `lora.train()` itself sets none.
- **N-6.** The first audit said the listed retired claims were "present with rationale". The Part 2 receipts table shows which are not.
- **N-7.** 17 of 50 plain/idiom twin ids share an identical situation, and 83 situations are distinct. The registered marginal test treats all items as independent. Noted only.

---

## 3. Part 2: where each flagged claim appears

**Abbreviations:** `LT` = `../stoic-llm-legacy/Stoic LLM - Results Tracker.md` (the only results tracker found). `LG` = `../stoic-llm-legacy/LEGACY.md`. `Plan` = `Repo_Build_Plan_v2__Cohesive_Application_Story.md` (gitignored). `ML` = `../modellens/README.md` (`../ModelLens` and `../modellens` are the same directory). `Colab` = `../stoic-llm-legacy/scripts/lora-train-colab.py`.

`make_figures.py` strings are the text baked into figure pixels. Long lines are quoted in part. Wrapped sentences are cited at the line holding the quoted fragment.

| Claim | File:line | Exact current wording | Flag |
|---|---|---|---|
| LoRA style/content | README.md:5 | "it also shifts judge-scored style and content in a single eval" | S1-1 |
| LoRA style/content | README.md:18 | "LoRA moves all three (Epictetus decision null)" | S1-1 |
| LoRA style/content | results/README.md:122 | "LoRA moves style, content, and (Marcus/Seneca) decision" | S1-1 |
| LoRA style/content | results/README.md:127 | "LoRA from the frozen merged-adapter" | S1-1 |
| LoRA style/content | results/README.md:129 | "prompts, symmetric decoding, **not** seed-tested like the CAA side" | S1-1 |
| Robust to judge | results/README.md:132 | "judge choice; the panels use different scales." | S1-1 |
| LoRA content deltas unaffected | docs/measurement-artifact.md:65 | "**The LoRA content deltas.** Those were generated with" | S1-1 |
| LoRA style/content | CLAUDE.md:290 | "style + content (single merged-adapter judge eval, not seed-tested)." | S1-1 |
| LoRA style/content, pixels | scripts/make_figures.py:300 | "LoRA moves style & " | S1-1 |
| LoRA style/content, pixels | scripts/make_figures.py:303 | "LoRA from merged-adapter evals (rubric judge, symmetric decoding). " | S1-1 |
| LoRA style/content | Plan:24 | "LoRA moves style, content, and the" | S1-1 |
| Byte-identical | docs/measurement-artifact.md:44 | "At coeff 0.11 the steered greedy output is byte-identical to baseline (greedy text" | S1-2 |
| Judge noise | docs/measurement-artifact.md:45 | "only visibly changes at coeff ≥ 1.0), so the residual is judge noise." | S1-2 |
| Byte-identical | results/README.md:56 | "At coeff 0.11 the steered greedy output is byte-identical to baseline (greedy" | S1-2 |
| Only visibly changes | results/README.md:57 | "text only visibly changes at coeff ≥ 1.0)" | S1-2 |
| Barely changes | results/README.md:86 | "greedy output barely changes below" | S1-2 |
| Byte-identical | LG:102 | "Under matched greedy the steered text is **byte-identical to baseline** at coeff" | S1-2 |
| Echo headline | README.md:3 | "a pre-registered powered test shows the shift is lexical echo, not reasoning" | S1-3 |
| Echo headline | README.md:5 | "A pre-registered test at n=100" | S1-3 |
| Echo banner | README.md:9 | "shift is **lexical echo, not reasoning**: Seneca moves idiom-worded items" | S1-3 |
| Echo banner | README.md:10 | "(+0.099 log-odds) and moves plain-worded items the other way (−0.083), and the" | S1-3 |
| Echo | README.md:30 | "the movement is register-driven, not reasoning" | S1-3 |
| Echo | README.md:45 | "**confirmed it as the dominant effect**" | S1-3 |
| Echo | CLAUDE.md:131 | "ECHO.** Seneca's decision effect is register-dependent, not reasoning" | S1-3 |
| Echo | CLAUDE.md:132 | "(idiom +0.099 / plain −0.083, Holm-sig, exceeds both controls; off-topic/plain" | S1-3 |
| Echo | CLAUDE.md:295 | "effect: idiom-worded items move +0.099, plain-worded −0.083 (Holm-significant," | S1-3 |
| Echo | docs/claim-revision-v4.md:4 | "lexical echo, not reasoning. On a calibrated, powered set (100 items, 25/cell)" | S1-3 |
| Echo | docs/claim-revision-v4.md:6 | "+0.099 log-odds, plain-worded items move −0.083" | S1-3 |
| Control match | scripts/prep_control_lora.py:1 | "Build the matched-length non-philosophical control LoRA training set." | S1-4 |
| Control match | scripts/prep_control_lora.py:11 | "53 examples, exactly as each philosopher adapter had." | S1-4 |
| Control recipe | scripts/prep_control_lora.py:432 | "train on Colab, canonical recipe unchanged" | S1-4, N-2 |
| Control recipe | scripts/prep_control_lora.py:488 | "train on Colab with the canonical recipe" | S1-4, N-2 |
| Control match | scripts/prep_control_lora.py:483 | "matched-length non-philosophical control for the v3 behavioral eval" | S1-4 |
| Control match | docs/pre-registration-v3.md:116 | "external non-philosophical corpus, length-matched" | S1-4 |
| Control recipe | docs/pre-registration-v3.md:129 | "Example count (53) and the recipe are held constant instead" | S1-4, N-2 |
| Control match | CLAUDE.md:194 | "matched-length non-philosophical control LoRA" | S1-4 |
| Control match | CLAUDE.md:218 | "Needs the matched-length non-philosophical LoRA as" | S1-4 |
| Control match | CLAUDE.md:273 | "LoRA trained on matched-length non-philosophical text," | S1-4 |
| Canonical recipe | stoic/lora.py:88 | "Train one LoRA adapter (canonical recipe" | N-3 |
| Recipe identical | Colab:8 | "RECIPE IS HELD IDENTICAL TO EXP 4" | N-1 |
| Recipe identical | LT:610 | "recipe held identical to Exp 4" | N-1 |
| Control match | Plan:133 | "a LoRA trained on matched-length" | S1-4 |
| CAA no-op | README.md:3 | "CAA is a no-op under matched decoding" | S1-5 |
| Only LoRA | README.md:5 | "LoRA is the only intervention that moves the judge-free decision instrument" | S1-5, S2-10 |
| CAA nothing | docs/measurement-artifact.md:57 | "**nothing** at any level — style, content, or decision." | S1-5 |
| CAA no-op | results/README.md:258 | "settled no-op/null" | S1-5 |
| CAA no-op | results/README.md:305 | "(passivity prior) and CAA (no-op) remain the settled findings." | S1-5, S2-2 |
| Only LoRA | results/README.md:119 | "LoRA is the only intervention with real effects, and it reaches" | S1-5, S2-10 |
| Only weight adaptation | CLAUDE.md:288 | "Only weight adaptation reaches decisions" | S1-5 |
| CAA best | CLAUDE.md:78 | "CAA clean best layers / coeff" | S1-5 |
| CAA best | stoic/config.py:118 | "On the stoic axis this is the CAA clean best" | S1-5 |
| CAA no-op | ML:36 | "CAA at the canonical coefficient is a circuit-level no-op" | S1-5 |
| No sweep | README.md:26 | "A coefficient sweep at the decision level has not been done" | S2-1 |
| Flat to 1.5 | CLAUDE.md:224 | "CAA coefficient sweep flat to 1.5." | S2-1 |
| Sweep | Plan:150 | "CAA coefficient sweep to 1.5" | S2-1 |
| Any coefficient | LT:15 | "None, at any coefficient up to 1.5" | S2-1 |
| Any strength | LT:571 | "CAA does not move decisions at any strength." | S2-1 |
| Passivity prior | README.md:45 | "Marcus is a broad *passivity prior*" | S2-2 |
| Passivity prior | results/README.md:101 | "passivity prior" | S2-2 |
| Passivity prior | results/README.md:197 | "matches the passivity-prior prediction" | S2-2 |
| Passivity prior | results/README.md:232 | "The passivity-prior account survives the circuit-level test." | S2-2 |
| Passivity prior | results/README.md:296 | "Marcus's passivity-prior signature generalizes" | S2-2 |
| Passivity prior | CLAUDE.md:304 | "Marcus = passivity prior" | S2-2 |
| Did not replicate | CLAUDE.md:309 | "Marcus's v2 passivity prior did not replicate (−0.007, and his v4" | S2-2 |
| Passivity prior, pixels | scripts/make_figures.py:117 | "(passivity prior); Epictetus null." | S2-2, S2-8 |
| Marcus cell | README.md:39 | "it moved −0.036 (n.s.) and matched Marcus's one significant cell" | S2-4 |
| Marcus cell | CLAUDE.md:135 | "matched by the non-philosophical control" | S2-4 |
| Marcus cell | CLAUDE.md:138 | "the reason Marcus's cell fails." | S2-4 |
| Marcus cell | CLAUDE.md:299 | "significant cell fails exceeds-both-controls (matched by the non-philosophical" | S2-4 |
| Marcus cell | docs/claim-revision-v4.md:41 | "it is matched by the inert control and is generic fine-tuning drift." | S2-4 |
| Stability gate | README.md:142 | "per the Tier 2 gate set in the build plan before the v4 eval" | S2-5 |
| Stability gate | CLAUDE.md:217 | "pure lexical echo on the philosophy axis, run on sycophancy only" | S2-5 |
| Stability gate | Plan:137 | "if v3 returns pure lexical-echo, cancel" | S2-5 |
| v2 label | CLAUDE.md:293 | "Seneca's v2 shift (+0.1345 log-odds, t=2.34 on" | S2-6 |
| v2 label | docs/claim-revision-v4.md:26 | "was +0.135 (2.34) on n=40" | S2-6 |
| Underpowered | README.md:45 | "by far the smallest corpus (123 chunks, the Enchiridion only)" | S2-7 |
| Underpowered | README.md:144 | "so it's an underpowered arm rather than a real null" | S2-7 |
| Underpowered | results/README.md:103 | "null (smallest corpus)" | S2-7 |
| Underpowered | CLAUDE.md:239 | "Epictetus is an underpowered arm, not a finding." | S2-7 |
| Underpowered | CLAUDE.md:306 | "Epictetus = underpowered null" | S2-7 |
| Underpowered | docs/claim-revision-v4.md:91 | "remains an underpowered-corpus null" | S2-7 |
| Figure title, pixels | scripts/make_figures.py:115 | "LoRA reaches the decision layer — with structure" | S2-8 |
| Both buckets, pixels | scripts/make_figures.py:116 | "Seneca moves both stance buckets" | S2-8 |
| Both buckets | README.md:34 | "positive in *both*" | S2-8 |
| Both buckets | README.md:41 | "Seneca moves both, Marcus accepting-only, Epictetus null" | S2-8 |
| Both buckets | results/README.md:102 | "both buckets positive ✓" | S2-8 |
| Both buckets | results/README.md:140 | "Seneca moves both, Marcus accepting-only, Epictetus null" | S2-8 |
| Both buckets | CLAUDE.md:123 | "Seneca both stance buckets" | S2-8 |
| Stage 4 gate | stoic/stages/adapters.py:29 | "shift positive in BOTH stance buckets with overall t >= ~2" | S2-8 |
| Caption | README.md:20 | "this movement is lexical echo, not reasoning" | S2-9 |
| Caption | README.md:43 | "this movement is lexical echo, not reasoning" | S2-9 |
| Caption | results/README.md:124 | "this movement is lexical echo, not reasoning" | S2-9 |
| Caption | results/README.md:142 | "this movement is lexical echo, not reasoning" | S2-9 |
| LoRA moves | README.md:3 | "LoRA moves forced-choice decisions" | S2-10 |
| LoRA reaches | README.md:30 | "**LoRA reaches decisions.**" | S2-10 |
| Circuit ordering | README.md:37 | "LoRA's circuit perturbation is ordered Seneca > Marcus > Epictetus ≈ 0" | S2-11 |
| Median | README.md:37 | "It is robust at the median" | S2-11 |
| In proportion | results/README.md:168 | "LoRA rewires the circuit in proportion to its behavioral decision" | S2-11, S2-7 |
| Strongest | results/README.md:225 | "strongest circuit modifier by every measure" | S2-11 |
| Strongest | results/README.md:245 | "Seneca = strongest modifier, character item-dependent" | S2-11 |
| Median | results/README.md:280 | "Seneca produces the largest circuit perturbations" | S2-11 |
| 2.2× | results/README.md:282 | "2.2×; survives removing the two outliers" | S2-11 |
| Strongest | CLAUDE.md:304 | "Seneca = strongest circuit" | S2-11 |
| Title, pixels | scripts/make_figures.py:172 | "Seneca is the largest circuit modifier — robust at the median, not just outliers" | S2-11 |
| 2.2×, pixels | scripts/make_figures.py:173 | "Median lead 2.2× survives removing the two outliers" | S2-11 |
| Largest | ML:37 | "LoRA is the largest circuit modifier" | S2-11 |
| Largest | Plan:68 | "Seneca is the largest circuit modifier robust" | S2-11 |
| n=1 label | CLAUDE.md:140 | "n=1-per-stance pilot" | S3 |
| n=1 label | CLAUDE.md:198 | "retires the n=1-per-stance pilot caveat" | S3 |
| n=1 label | REPRODUCTION_PLAN.md:148 | "12c is a pilot (n=1 per" | S3 |
| CI | docs/claim-revision-v4.md:26 | "−0.070, [−0.196, +0.056] (0.282)" | S3 |
| CI | docs/claim-revision-v4.md:27 | "−0.102, [−0.179, −0.025] (0.015)" | S3 |
| Harness | docs/decisions.md:17 | "The harness runs" | S3 |
| Path | CLAUDE.md:265 | "logic `stoic/calibrate.py`" | S3 |
| Path | docs/naming-cleanup.md:52 | "`stoic/calibrate.py` is the logic" | S3 |
| Predicted before | CLAUDE.md:291 | "topology predicted the split before behavior showed it." | N-4 |
| Predicted before | ML:39 | "circuit split predicted the behavior before it was measured" | N-4 |
| Predicts | results/README.md:172 | "circuit topology predicts the behavioral split" | N-4 |

S2-3 has no row because the flag is an omission: no document mentions Seneca's v4 stance contrast. S2-10's other half is also an omission: `README.md` does not mention `control_neutral` or a token-matched control.

**Not searched:** the Hugging Face card, GitHub About field and portfolio blurb, because they are off-repo.

### Retirement receipts

| Claim | Receipt? | Where |
|---|---|---|
| "reaches decisions" (v2) | Present | `README.md:7-16,30`; `CLAUDE.md:292-302`; `docs/claim-revision-v4.md` R1-R6 |
| Earlier beat 2 | Present | `CLAUDE.md:283` |
| "stance modulation" | Present | `results/README.md:245,303` |
| "9/10 wins" | Present | `results/README.md:282` |
| "compress-on-accepting / amplify-on-active" | Present | `results/README.md:250-251` |
| ΣΔ\|c\| aggregates | Present | `REPRODUCTION_PLAN.md:147-148` |
| Marcus p=0.038 | Present | `docs/decisions.md`; `CLAUDE.md:333` |
| "predicted before measured" | **MISSING and still live** | No note in this repo. Live at `CLAUDE.md:291`, `ML:39`, `results/README.md:172`. See N-4 |
| "Stoic vector reliably shifts register, not decisions" | **MISSING in this repo** | The claim exists only at `LT:414,589`, with no retirement note found in this repo |
| "disruption signature" | **Partial** | Phrase absent from all docs. Commit subject `9b45ac7` names it. `results/README.md:293` says "uniform disruption" without naming the retired claim |
| "compress the high-\|c\| tail" | **MISSING** | The phrase occurs only in the first audit file (`git log -S`) |
| "net exactly zero" | **MISSING** | Same. `results/README.md:294` addresses net flattening without naming it |
| LoRA style/content on clean adapters (S1-1) | **MISSING in this repo** | `LT:14,637`, and `LG:176-178` flags the old adapters; neither was carried over |
| Byte-identical (S1-2) | **MISSING** | |
| "53 examples" (S1-4) | **MISSING** | |
| CAA selection via asymmetric decoding (S1-5) | **MISSING in this repo** | `LG` documents the decoding, not its effect on selection. `results/README.md:86-88` covers coefficient only |
| No decision-level sweep (S2-1) | **MISSING** | |
| Passivity prior (S2-2) | **Partial and inconsistent** | `CLAUDE.md:309` retires it with the wrong number; `README.md:45` and `results/README.md:101,296,305` have no pointer |
| Stability gate wording (S2-5) | **MISSING** | Commit `142a5b9` records the wording change, not the cancel/sycophancy difference |
| v3 labeled as v2 (S2-6) | **MISSING** | |
| Figure pixel text (S2-8) | **Partial** | Added captions sit under the PNGs; the pixels are unchanged |
| "Largest modifier" illustration (S2-11) | **MISSING** | |

---

## 4. Part 3: draft pre-registrations

**All four are DRAFT. None has been run, and none is signed.** Bracketed fields are yours to fill. No threshold here was chosen by looking at Part 1. Where one is given, it is carried from the signed v3/v4 documents, as noted.

**Common to all, carried from `docs/pre-registration-v4.md` §3 and §6:** primary statistic is a t-test on Δlog-odds. Family-wise α=0.05 with Holm. Invalidate if a cell falls outside the 0.10 limit, any item falls outside [0.2, 0.8], or v2 base integrity fails. Record the torch version in the JSON. Vanish outcomes use the three-way framing from `CLAUDE.md`.

### Draft (a): CAA coefficient sweep on v4, generation-free

**Contamination: not compromised on its outcome**, since CAA has never been run on these items. Three things are already known and must be disclosed: the v4 LoRA and control results, the legacy Epictetus v2 sweep, and the audit's perturbation ratio.

| Field | Draft |
|---|---|
| Question | Does CAA move forced-choice decisions on v4 items at any coefficient, overall or on the phrasing contrast? |
| Items | `data/generated/dilemmas_v4_candidates.json`, frozen, 100 items |
| Conditions | Base; Marcus L26, Seneca L4, Epictetus L8; coefficients [S to fix. The legacy grid was 0.11, 0.2, 0.4, 0.8, 1.5] |
| Primary tests | For each arm × coefficient: overall paired t vs 0; phrasing marginal, as in `behavioral_eval.py` |
| Threshold | α=0.05 family-wise. Bound for "no movement": [S to fix. The only already-registered value is +0.15 from v4 §4] |
| Correction | Holm across all arm × coefficient × test cells: [S to confirm family size] |
| Moves | Significant after Holm at some coefficient: the "CAA is a no-op" claim is limited to the coefficients where it held |
| Bounded null | Every CI excludes the bound: no movement at these coefficients on these items |
| Inconclusive | Some CI contains both 0 and the bound: report plainly |
| Vanish branch | Three-way, per `CLAUDE.md` |
| To fix first | Whether to add a control, what to do if a high coefficient breaks the model, whether layers stay canonical given S1-5 |

### Draft (b): Marcus/Seneca stance test on v4

**Contamination: COMPROMISED as a confirmatory test.** The audit and this report both computed stance contrasts for every arm on these exact items. The 40 carried-over items were also scored in v3. Nothing run on these 100 items can be confirmatory for stance.

| Field | Draft |
|---|---|
| Question | Do Marcus and Seneca shift accepting-stance items differently from active-stance items? |
| Items, as requested | v4, 52 accepting / 48 active. **Can only be labeled exploratory** |
| Items, if confirmatory | [S to decide] a new stance-balanced set, authored and calibrated under v3 R1-R7 and v4 R8-R10, with no v4 items and frozen before any adapter runs |
| Conditions | Marcus, Seneca, control_nonphil, control_neutral, subject to the S1-4 decision |
| Primary test | Two-sample t on Δlog-odds, accepting vs active, per arm. Direction: accepting > active, from the v2 pattern |
| Threshold | α=0.05 family-wise |
| Correction | Holm across the two arms |
| Control rule | Must exceed both controls: [S to fix whether sign-aware. `behavioral_eval.py:238-240` compares absolute values] |
| Holds for arm | Significant and exceeds both controls: a stance-dependent effect specific to that arm |
| Fails on control | Significant but doesn't exceed: generic fine-tuning effect |
| Null | Three-way, per `CLAUDE.md` |

### Draft (c): token-matched neutral control

**Contamination: partly compromised.** The new adapter's outcome is unseen, but `control_neutral`'s v3 and v4 results, including the new-items split, are known. The selection rule and comparison must be fixed before training.

**Source of the requirement:** `docs/pre-registration-v3.md:132-136`: a moving control "is then followed by a token-matched variant (~21 examples, same total characters)". `docs/pre-registration-v4.md` §7 makes this a separate pre-registration.

| Field | Draft |
|---|---|
| Question | Is `control_neutral`'s movement due to its content or its 2.6× token excess? |
| Items | v4, frozen |
| Training text | `neutral_text`, about 21 examples, same total characters as [S to fix: 29,502 pooled pair chars, per §4; S1-4 shows this isn't the clean adapters' training size] |
| Selection | [S to fix before training. Existing rule: stratified by author, quantile-spaced, deterministic] |
| Recipe | [S to fix: same as existing controls, or the Colab recipe. See N-2] |
| Primary tests | Overall paired t on Δlog-odds; the three v4 planned comparisons |
| Threshold / correction | α=0.05 family-wise, Holm. "Same movement" criterion: [S to fix. §4 gives no number] |
| Still moves | Evidence for content rather than token count |
| Stops moving | Evidence for token count; the existing control's movement stays ambiguous as to content |
| Inconclusive | CI contains both 0 and the criterion: report plainly |

### Draft (d): dose-matched control

**Contamination: partly compromised.** New control outcomes are unseen, but the comparators' v4 effects are known, so "exceeds" has a known target. This must be disclosed.

**Decide first, under both branches: what "dose" means.** Options: examples, tokens kept, weight updates, micro-batches, or effective batch. They disagree (Part 1, Item 1 table).

| Field | Branch A: chunk counts are right (what the records say) | Branch B: all adapters trained on 53 |
|---|---|---|
| Consequence | `prep_control_lora.py:11` is wrong | The tracker, README and script are wrong; the corpus-size account of Epictetus (`README.md:45,144`) lacks a basis |
| Question | Does the v4 verdict hold against controls matched on [dose]? | Same, after aligning recipe (N-2) |
| Conditions | One non-philosophical control per arm at that arm's dose, or [S to fix] | One control under the clean recipe |
| Text supply | Vitruvius Book X slice is 73,443 chars, so it can't reach Marcus or Seneca by text. Whole book is about 600k chars, but `docs/decisions.md` rejected other books on content. [S to fix source] | Existing 53 examples suffice |
| Recipe | [S to fix: Colab script as written, including padding and labels] | Same |
| Items | v4, frozen | v4, frozen |
| Primary tests | Three v4 planned comparisons on the new controls; re-evaluate "exceeds both" for Seneca's phrasing marginal | Same |
| Threshold / correction | α=0.05 family-wise, Holm | Same |
| Still exceeds | The ECHO reading's control condition holds under that dose | Same |
| No longer exceeds | The control condition fails under that dose: a registered reading must then be re-stated | Same |
| Inconclusive | Report plainly; don't iterate on dose after seeing a result | Same |
| Prerequisite | Settle S1-4 from the Colab record, if available | Same |

---

## 5. Decisions Sebastian needs to make

| # | Decision | Options | Flags |
|---|---|---|---|
| 1 | LoRA style/content claim | Re-judge clean adapters; state the claim as old-adapter; remove the panels from the headline | S1-1 |
| 2 | Artifact writeup | Correct the sentence only; add per-pair analysis; re-measure | S1-2 |
| 3 | How to treat the v4 split | Note it as exploratory; author a new item set; no change | S1-3, S2-3 |
| 4 | Definition of dose | Examples; tokens; weight updates; a combination | S1-4, S2-7, N-1, N-2 |
| 5 | Control response | Keep controls with disclosure; retrain under Draft (d); retrain with the clean recipe only | S1-4, N-2 |
| 6 | Canonical recipe | Declare the Colab script canonical; declare `stoic/lora.py` canonical; document both | N-1, N-2, N-3 |
| 7 | CAA scope | Limit the claim to 0.11; run Draft (a); both | S1-5, S2-1, S2-10 |
| 8 | Layers and 0.11 going forward | Keep canonical with disclosure; re-select under matched decoding | S1-5 |
| 9 | Token-matched control | Run Draft (c); withdraw that step with a recorded reason | S2-10 |
| 10 | Marcus claim | Keep v2 wording as v2; retire with corrected number; run a new-items test | S2-2, S2-3 |
| 11 | Epictetus framing | Keep "underpowered" with chars and updates; drop the size explanation; settle by retrain | S2-7 |
| 12 | Stability sweep gate | Follow the plan (cancel); follow CLAUDE.md (sycophancy only); record an amendment | S2-5 |
| 13 | Figures | Regenerate with new text; keep with captions; retire | S2-8, S2-9, S1-1, S2-11 |
| 14 | Circuit claim metric | Keep normalized; report raw too; add a significance test | S2-11 |
| 15 | Missing receipts | Where they go, and whether to carry over legacy text | N-4, N-6, Part 2 |
| 16 | ModelLens README | Edit in that repo, and when | S1-5, S2-11, N-4 |
| 17 | Build plan visibility | Keep gitignored; commit it, so citations can be checked | S2-5 |
| 18 | Audit files | Commit the two uncommitted files, and whether to amend the earlier audit | N-5, N-6 |

---

## 6. Scope

**Read in full:** `audit/2026-11-01_followup-audit.md`; `../stoic-llm-legacy/scripts/lora-train-colab.py`; `../stoic-llm-legacy/data/lora_training/README.md`.

**Read in part:** `scripts/prep_control_lora.py`; `docs/corpus-sources.md`; `docs/pre-registration-v3.md:132-136`; `../modellens/README.md:28-42`; `../stoic-llm-legacy/`: tracker, `LEGACY.md`, `AUDIT_REPORT.md`, `scripts/run_lora_eval.py`, `run_dilemma_eval.py`, `run_seed_eval.py`, `stoic_llm/eval/{sweep,dilemma,judge}.py`, `stoic_llm/lora/runner.py`, `stoic_llm/config.py`.

**Read earlier this session and relied on:** `stoic/lora.py`, `stoic/stages/adapters.py`, `scripts/make_figures.py`, `scripts/behavioral_eval.py`, the build plan, `docs/pre-registration-v4.md`.

**Data queried:** v3, v4, Stage 2, Stage 3 (greedy and sampled), Stage 4, style and Exp 12 JSONs; `data/reference/judges/`; `data/reference/chunked/`; `data/generated/dilemmas_v4_candidates.json`; `data/generated/lora_training/`; `data/generated/raw/control/pg20239.txt`; control `trainer_state.json` and `training_args.bin`; Exp 4 checkpoints; clean adapter folders and weights; legacy `results/sweeps/` and `results/judges/`; `.axis_snap/` v3 and v4 logs (grep).

**Ran, all read-only:** seven scripts in `/tmp/stoic_resolution/`; grep and `git log`; the base tokenizer for token counts. No model weights were loaded for inference, and no experiment was run.

**Not re-run:** `pytest`; the perturbation ratio, as instructed.

**Not read:** the Colab run itself, Google Drive, the Hugging Face card, GitHub About and portfolio; ModelLens beyond those README lines; legacy `app/`, `stoic_llm/steering/` and `run_model_with_hook` (so `LEGACY.md`'s runtime claim wasn't independently traced); other legacy docs; v3-only item checks.

Absence of a flag covers only what was read. This report is not a clearance of the repo.
