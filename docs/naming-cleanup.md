# Naming cleanup — inventory

The axis refactor made the instrument configurable but left every name saying
"stoic". Some of that is free to fix; some is load-bearing on the published
record; one part cannot be touched at all. This separates them.

Verification for all of it: renames are behaviour-preserving, so the existing
`.axis_snap/before.json` remains the anchor — no new baseline is needed. Use
`--cheap` (~1 min) after each step and one full `--replay seneca` run at the end.

---

## Tier 0 — unblock first

Nothing else can be verified end-to-end until this is done.

| Action | Where |
|---|---|
| Fix the ModelLens import | `exp12_circuit_analysis.py:37` hardcodes a path under `DSAN/Spring 2026/…` that no longer exists. Replace with `pip install -e ../modellens` and delete the `sys.path` insertion |
| Delete the dead hotfix | `exp12_circuit_analysis.py:49,149`; `exp12_sweep.py:63` |
| Fix the false provenance string | `exp12_circuit_analysis.py:214` writes `"upstream fix pending"` into every future result JSON; the fix landed upstream as `44d9b77` |

**Why first:** 20 of the `AUTHORS` call sites are in the exp12 scripts, and they
cannot even be imported today. Renaming them blind is the riskiest move in this
whole inventory — a mistake stays invisible until the v3 circuit sweep.

---

## Tier 1 — free, mechanical

Code-only. No frozen data, no result-JSON keys, no published numbers.

| Current | Proposed | stoic/ | tests/ | scripts/ |
|---|---|---:|---:|---:|
| `Author` / `AUTHORS` | `Arm` / `ARMS` | 10 | 6 | 20 |
| `STOIC_AXIS` env var | `BEHAVIOR_AXIS` | 7 | 14 | 0 |
| `judge.STOIC_RUBRIC` | `judge.RUBRIC` | 2 | 0 | 1 |
| `p_stoic()` function | `p_target()` | 2 | 3 | 4 |
| `--author` flag, `stage3(authors=…)` | `--arm`, `arms=` | 1 | 1 | 0 |
| `calibrate.TOPIC_AXES` / `PHRASINGS` | delete — superseded by `CELL_AXES` | 2 | 0 | 2 |
| `config.*_DIR` re-exports of `axis.*_DIR` | pick one home, import from it | 5 | — | — |

Note `p_stoic()` the *function* is free; `baseline_p_stoic` / `steered_p_stoic`
the *JSON keys* are not — see Tier 3.

---

## Tier 2 — structural

Judgement calls, not mechanical.

**Two files named `calibrate.py`.** `stoic/calibrate.py` is the logic,
`stoic/stages/calibrate.py` is the stage. Suggest renaming the logic module to
`stoic/calibration.py` so imports read unambiguously.

**Stage files are named by mechanism, the CLI by number.** `verify.py` is stages
0–2, `content.py` is stage 3 + style, `adapters.py` is stage 4. The mapping only
exists in the `stages/__init__.py` docstring. Either rename the files after what
they measure (`determinism_and_vectors`, `judged_effects`, `decisions`) or make
the docstring mapping authoritative and leave it.

**The `stoic/` package name.** The subject is intervention locus; stoicism is one
instrument. Renaming touches every import, `pyproject.toml`
(`[project.scripts]`, `packages.find`), `tests/conftest.py`, all of `scripts/`,
and the `python -m stoic` invocation quoted throughout README, CLAUDE.md and
results/README.md. Purely mechanical but the largest blast radius — do it last,
as its own commit, or not at all. Counter-argument: the repo directory is
`llm-stoic` and the HF artifact repo is `llama-3.2-3b-stoic-steering`, so a
package-only rename leaves the naming half-migrated.

---

## Tier 3 — blocked by the published record

These appear inside checked-in `results/` JSONs.

| Key | Result files affected |
|---|---|
| `per_author` | 6 |
| `baseline_p_stoic` | 4 |
| `steered_p_stoic` | 2 |

**Recommendation: leave them.** Renaming does not touch the old files — it
changes what *future* runs write, which means new runs stop being directly
comparable with the published ones, `tests/test_stats.py` (which reproduces the
cited sign-test numbers from those files) needs to handle both spellings, and
`--check-published` needs a key map. The cost is real and recurring; the benefit
is cosmetic. Recorded as deliberate debt in `docs/decisions.md`.

---

## Tier 4 — cannot be renamed

`stoic`, `nonstoic`, `stoic_stance` are keys inside
`data/reference/config/dilemmas_v2.json`, which is frozen and read-only.

**No action needed, and none possible.** The refactor already solved this the
only way it can be solved: the code reaches those keys exclusively through
`axis.fields`, so a sycophancy set can use entirely different names. The two
remaining mentions in `stoic/` are docstrings; the 19 in `tests/` are test
fixtures legitimately building stoic-axis items.

---

## Suggested order

1. Tier 0 — unblocks verification of the exp12 call sites
2. Tier 1 renames, one commit per name, `--cheap` after each
3. Tier 2 structural moves
4. Package rename, last and alone (if doing it)
5. One full `--replay seneca` snapshot → `--compare` against the existing
   `before.json`, which must still print IDENTICAL
