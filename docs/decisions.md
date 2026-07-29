# Decision log

Engineering decisions and environment facts that the code cannot explain about
itself. `CLAUDE.md` says what is true now; this says *why*, and what each choice
replaced.

Same rule as the claim tracker: **nothing is deleted.** A superseded decision
stays, with a pointer to the entry that replaced it.

Entry format — date, decision, why, and what it costs. One entry per decision,
newest first.

---

## 2026-07-27 — Behavioral axis becomes a config directory

**Decision.** An axis (arms, layers, coefficients, dilemma set, field maps,
judge rubric, thresholds, reference targets) is declared in `axes/<name>/` and
loaded by `stoic/axis.py`. The pipeline reads `ACTIVE`; no module hardcodes the
instrument.

**Why.** Roadmap step 6 (sycophancy) is what turns a single-corpus result into a
claim about intervention loci. Before this, adding an axis meant editing six
modules, so the second axis was gated on a refactor nobody had scoped.

**Cost.** Two indirection layers between a constant and its use. Reading
`config.AUTHORS` no longer tells you the layers; `axes/stoic/axis.json` does.

---

## 2026-07-27 — `axis.py` imports nothing from the package

**Decision.** Path roots (`PROJECT_ROOT`, `REFERENCE_DIR`, `GENERATED_DIR`,
`MODELS_DIR`) are defined in `stoic/axis.py` and re-exported by `stoic/config.py`.

**Why.** `config` depends on the axis, so the axis cannot depend on `config`
without a cycle. Defining the roots twice would give two sources of truth for
the reference wall.

**Cost.** `config.REFERENCE_DIR` and `axis.REFERENCE_DIR` are the same object
under two names — a naming cleanup candidate, not a correctness issue.

---

## 2026-07-27 — The axis is bound from `sys.argv`, above the imports

**Decision.** `stoic/__main__.py` scans `sys.argv` for `--axis` and sets
`STOIC_AXIS` *before* any `from stoic…` import. argparse still declares the flag,
with its default read back from the environment.

**Why.** `ACTIVE` is resolved at import time. argparse parses `--axis` long after
`stoic` is imported, so a flag handled only by argparse would have silently done
nothing while the env var worked. Any check exercising only the env var would
have passed.

**Related.** A flag that disagrees with an already-set `STOIC_AXIS` is a hard
error, not a precedence rule — a silently chosen axis attributes real numbers to
the wrong axis. Pinned by `tests/test_axis.py::test_flag_and_env_resolve_the_same_axis_end_to_end`.

**Cost.** `stoic/__init__.py` must stay import-free of `config`, or the axis
binds too early. Pinned by a test.

---

## 2026-07-27 — Pass B stays hardcoded to the stoic axis

**Decision.** `corpus.py` and `pairs.py` are not generalized. `load_axis()`
rejects any non-stoic axis declaring `corpus.kind == "gutenberg"` or
`pairs.kind == "corpus_contrastive"`, pointing at `pass_b._limit` in the axis file.

**Why.** Sycophancy supplies prepared trait pairs (`kind: "prepared"`), so
generalizing Pass B does not unblock the second axis. It serves Pass B on the
philosophy axis, which is roadmap item 9.

**Cost.** "Adding an axis is a config file" holds only for prepared-pair axes. A
future corpus-derived axis needs this work first. Recorded rather than hidden.

---

## 2026-07-27 — Result JSONs keep saying `per_author`

**Decision.** The per-arm key in stage result JSONs stays `per_author` on every
axis, as do `baseline_p_stoic` / `steered_p_stoic`.

**Why.** Six checked-in result files under `results/` use these keys, and they
are the replication record. Renaming them makes new runs non-comparable with the
published ones, and `tests/test_stats.py` reproduces cited numbers from them.

**Cost.** The name is wrong for a non-philosophy axis. Deliberate debt; revisit
only with an explicit decision about the record.

**Superseded by.** *(nothing yet — see the naming-cleanup inventory)*

---

## 2026-07-27 — Verification is before↔after on one machine, not vs the record

**Decision.** `scripts/axis_snapshot.py` captures every deterministic quantity
with floats as `float.hex()`. Two comparisons are kept separate: `--compare`
(before↔after, same machine) is the claim that the refactor changed nothing;
`--check-published` (vs checked-in `results/`) is an environment check.

**Why.** This machine runs torch 2.8.0 while `pyproject.toml` pins 2.5.1. Mixing
the two comparisons would let environment drift be read as refactor damage, or
hide refactor damage inside expected drift.

**Cost.** Two full runs (~3 h each, $0) per verified refactor.

---

## 2026-07-27 — fp16 CPU drift between torch 2.5.1 and 2.8.0 is accepted

**Facts.** With no pipeline file modified, on torch 2.8.0 / transformers 4.57.3:

| Quantity | Published | This machine |
|---|---|---|
| Dilemma baseline mean | 0.541601902275579 | 0.5416093931009527 |
| Per-item baseline exact matches | — | 19 / 40 |
| Largest per-item baseline deviation | — | **4.26e-3** (`ext_06`); median 2.2e-4 |
| Largest steered deviation | — | 5.6e-3 (Seneca), 3.1e-3 (Marcus) |
| Stage-3 greedy generations reproduced | — | 9 / 12 per set |
| Marcus sign test | 27+/13−, p = 0.038 | 24+/16−, p = 0.268 |
| Seneca sign test | 25+/15−, p = 0.154 | identical |
| Epictetus sign test | 17+/23−, p = 0.430 | identical |
| Marcus / Seneca / Epictetus ΔP | +0.0307 / +0.0606 / +0.0003 | identical to 4 dp |
| All t-statistics and bucket p-values | — | reproduce to ~4 dp |

*(An earlier revision of this entry said the ceiling was ~1.6e-3. That came from
inspecting only the first 5 of 21 differing items. The true maximum is 4.26e-3 —
roughly 3× wider, which matters for choosing any tolerance.)*

**Decision.** Accepted for refactor verification. Every magnitude-based
statistic reproduces to ~4 dp, because a per-item drift of ~4e-3 largely cancels
when averaged over 40 items. That cancellation is exactly why means survive and
the sign test does not.

**Why it still matters.** The sign test discards magnitude, so any item whose Δ
sits inside numerical noise contributes a coin flip. Exactly three Marcus items
flipped — `ctrl_05`, `emot_05`, `ext_06` — and they are precisely the three whose
|Δ| was below the drift scale. `emot_05` is the clean case: its *steered* value
is bit-identical across torch versions, so the flip comes entirely from baseline
drift in the paired quantity.

Seneca held, but by one item: its smallest |Δ| (−0.00234, `fate_03`) is also
inside the drift band and simply did not flip. Stable by luck, not by margin.

**Open, and not testable here.** Whether `torch==2.5.1` restores 27+/13− is
unverified: 2.5.1 has no wheel for this interpreter (Python 3.13.5, macOS
arm64 — PyPI's set starts at 2.6.0), so the pin in `pyproject.toml` is currently
uninstallable on this machine. Confirming it needs a Python 3.12 environment.
Do not restate p = 0.038 as settled until then — and note that an effect visible
only under one torch build is not one to lean on regardless.

---

## 2026-07-28 — The CLAUDE.md "open discrepancy" resolves as *both get stated*

**The discrepancy.** CLAUDE.md Status asked whether the summary docs overclaim
by citing a Seneca t≈2.58 while the in-repo sign test is 25+/15−, p = 0.154 (n.s.).

**Resolution: they measure different things, and both are correct.** The t≈2.58
is in the checked-in JSON — `seneca.overall.t_stat = 2.576132231272829,
p = 0.0139`. The summary docs cite the **paired t-test on ΔP** (magnitude); the
READMEs cite the **sign test** (direction consistency). This is the "both get
stated" branch, not the overclaim branch.

**And the re-run sharpens it.** On the magnitude statistic Seneca (t = 2.58) is
the stronger mover, ahead of Marcus (t = 2.00) — which matches story beat 3. The
*only* statistic that made Marcus look like the significant arm is the sign test,
and it is the one that does not survive a torch bump.

**Framing for the write-up.** The honest reading is that the sign test was the
wrong instrument for a 40-item paired comparison containing several near-zero
deltas — not that Marcus's effect vanished; its mean ΔP reproduced to 4 dp. That
is a methodology upgrade, consistent with story beat 1, rather than a retraction.
Untouched either way: Marcus's accepting-only pattern (bucket p = 0.007) and the
CAA-null / LoRA-moves split are magnitude claims.

**Action before the v3 sweep.** Pre-register a tolerance-aware or
magnitude-based directional statistic, so the same failure cannot recur on 40
fresh items. Choosing a tolerance *after* seeing the flip would be post-hoc.

---

## 2026-07-28 — `tests/test_stats.py` will fail the next time stage 4 runs

**Fact.** `test_sign_test_reproduces_published_numbers` hardcodes
`marcus: (27, 13, 0.0385)` and resolves its input through `_latest()`, which is
the last file by sorted glob. It passes today only because the newest stage-4
JSON is the frozen 2026-07-17 one.

**Consequence.** The moment any new `results/stage4_lora_dilemmas/lora_dilemmas_*.json`
lands, the test picks it up and compares it against the old expectation, and the
suite goes red for a reason that has nothing to do with the code.

**Not fixed yet** — it touches how the replication record is read, so it should
be a deliberate choice: pin the test to the specific frozen filename, or carry
expectations keyed by file so both runs are asserted.

---

## 2026-07-27 — `USE_TF=0` is required to load any model here

**Fact.** `transformers` probes for TensorFlow during model resolution, and this
machine's TensorFlow build segfaults inside `pywrap_tensorflow.preload_check`.
The process dies with exit 139 before any weight is read — `python -m stoic stage1`
included.

**Decision.** `scripts/axis_snapshot.py` sets `os.environ.setdefault("USE_TF", "0")`
above its imports. It touches no numerics; it only stops `transformers` importing
TensorFlow.

**Open.** Where the guard should permanently live — `stoic/model.py`, the CLI, or
the shell environment — is undecided. Until then the repo's own CLI segfaults on
this machine.

---

## 2026-07-27 — `peft` pin is older than the adapters were saved with

**Fact.** `pyproject.toml` pins `peft==0.18.1`. The frozen adapters were saved by
a newer version, so loading them warns that `lora_ga_config` and `use_bdlora` are
ignored.

**Assessment.** Neither key affects the canonical recipe (r=8, α=32,
q_proj + v_proj), so the merge math is unchanged and Stage-4 ΔP reproduces to 4 dp.
Recorded because a version mismatch on an adapter config is exactly the class of
thing that silently shifts numbers.

---

## 2026-07-27 — ModelLens closure-return bug is fixed upstream

**Fact.** Fixed in the ModelLens repo as commit `44d9b77` (2026-07-06),
*"Fix _capture_activations: return hook_fn from make_hook, not from hook_fn"* —
on `main`, pushed to `origin/main`. Covered by `tests/test_core_regression.py`
(12 tests naming the commit); full suite 33 passed, 1 skipped. Roadmap step 1's
other items — hook cleanup after exceptions, custom `metric_fn` dispatch, adapter
dispatch — also have tests.

**Consequence.** The local hotfix in `scripts/exp12_circuit_analysis.py` is dead
code. The two implementations differ only in parameter names, so published exp12
results stand. It should be removed: it *overwrites* upstream's function at
runtime, so future improvements to that module would be silently clobbered, and
`exp12_circuit_analysis.py:214` writes `"upstream fix pending"` into every future
result JSON, which is now false.

**New blocker, previously unrecorded.** `exp12_circuit_analysis.py:37` hardcodes
a ModelLens path under `DSAN/Spring 2026/Neural Nets - 6600/…` that no longer
exists, and `modellens` is not installed. `import modellens` raises
`ModuleNotFoundError`, so exp12 and the sweep cannot run at all today. Suggested
fix: `pip install -e ../modellens` and delete the `sys.path` insertion.
