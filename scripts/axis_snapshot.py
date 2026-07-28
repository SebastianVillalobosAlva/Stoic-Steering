"""Deterministic-output snapshot — the proof that a refactor changed nothing.

Emits ONE canonical JSON holding every deterministic quantity the pipeline
produces. Run it at HEAD before touching anything, run it again after, diff the
two: an empty diff is the claim "existing philosopher results reproduce
byte-identical where deterministic", made checkable instead of asserted.

    python scripts/axis_snapshot.py --out .axis_snap/before.json --replay seneca
    ... refactor ...
    python scripts/axis_snapshot.py --out .axis_snap/after.json  --replay seneca
    python scripts/axis_snapshot.py --compare .axis_snap/before.json .axis_snap/after.json
    python scripts/axis_snapshot.py --check-published .axis_snap/after.json

Two comparisons, deliberately separate so environment drift can never be
mistaken for refactor damage:

  --compare          before vs after, same machine   -> "the refactor changed nothing"
  --check-published  after vs the checked-in results -> "still matches the record"

Design constraints:

- **Floats are compared bit-exactly**, serialized as `float.hex()`. No
  tolerances — a tolerance is a place for a real regression to hide.
- **Never calls a stage function.** The stages write checkpoint JSONs into
  `results/`; a verification harness must not produce results. Every quantity
  is recomputed here from the library primitives (`dilemmas`, `steering`,
  `lora`, `calibrate`, `corpus`, `model`), which also keeps the snapshot
  independent of how stage internals get reorganized.
- **Nothing is written outside `--out`.** No API is called; $0.

`--cheap` collects only the artifacts that need no model (11, 9, 12) — used at
a mid-refactor checkpoint where nothing model-dependent can have moved yet.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Loading any model through transformers probes for TensorFlow, and this
# machine's TF build segfaults inside pywrap_tensorflow.preload_check — taking
# the whole process with it (exit 139) before a single weight is read. Skipping
# the probe touches no numerics; it only stops transformers importing TF.
# Pre-existing environment fault, not a property of the pipeline.
os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Artifact numbering follows the verification plan; the names are the JSON keys.
ARTIFACTS = {
    1: "stage1_baseline",
    2: "stage2_steering",
    3: "stage2_injection_site",
    4: "stage4_lora",
    5: "stage0_determinism",
    6: "stage3_greedy_replay",
    7: "stage3_sampled_replay",
    8: "calibration",
    9: "corpus_chunks",
    11: "config_surface",
    12: "pytest",
}
CHEAP_ARTIFACTS = (9, 11, 12)
MODEL_ARTIFACTS = (1, 2, 3, 4, 5, 8)          # plus 6, 7 when --replay is given


# --- canonical encoding ---------------------------------------------------

def enc(o):
    """Encode to a canonical, bit-exact JSON form.

    Floats become {"f": "<hex>"} so equality is on the bits, not the decimal
    rendering — 0.1 + 0.2 and 0.30000000000000004 must never compare equal by
    accident of formatting. Dict keys are sorted so ordering can't create a
    false diff.
    """
    if isinstance(o, bool) or o is None:
        return o
    if isinstance(o, float):
        return {"f": o.hex()}
    if isinstance(o, int) or isinstance(o, str):
        return o
    if isinstance(o, dict):
        return {str(k): enc(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))}
    if isinstance(o, (list, tuple)):
        return [enc(v) for v in o]
    return str(o)


def sha(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str | None:
    p = Path(path)
    return sha(p.read_bytes()) if p.exists() else None


def rel(path) -> str:
    """Path relative to the repo root — machine-independent, so a snapshot
    taken in a worktree still compares against one taken in the checkout."""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# --- artifact 11: config surface (no model) -------------------------------

def collect_config_surface() -> dict:
    from stoic import calibrate as cal
    from stoic import config, judge, lora, pairs
    from stoic.dilemmas import PROMPT_TEMPLATE, load_dilemmas

    dilemmas = load_dilemmas()
    stances: dict[str, int] = {}
    for d in dilemmas:
        stances[d["stoic_stance"]] = stances.get(d["stoic_stance"], 0) + 1

    arms = {}
    for name, a in config.AUTHORS.items():
        arms[name] = {
            "key": a.key,
            "label": a.label,
            "layer": a.layer,
            "coeff": a.coeff,
            "pairs_file": rel(a.pairs_file),
            "vector_file": rel(a.vector_file),
            "adapter_dir": rel(a.adapter_dir),
            "pairs_sha256": sha_file(a.pairs_file),
            "vector_sha256": sha_file(a.vector_file),
            "adapter_weights_sha256": sha_file(a.adapter_dir / "adapter_model.safetensors"),
        }

    return {
        "model": {
            "model_name": config.MODEL_NAME,
            "dtype": str(config.DTYPE),
            "device": config.DEVICE,
            "num_layers": config.NUM_LAYERS,
            "hidden_dim": config.HIDDEN_DIM,
        },
        "gen_kwargs": config.GEN_KWARGS,
        "arms": arms,
        "digests": {
            "neutral_pair_prompt": sha(config.NEUTRAL_PAIR_PROMPT),
            "judge_rubric": sha(judge.STOIC_RUBRIC),
            "default_prompts": sha("\n".join(config.DEFAULT_PROMPTS)),
            "dilemma_prompt_template": sha(PROMPT_TEMPLATE),
            "dilemmas_v2_file": sha_file(config.DILEMMAS_V2),
            "sources_json": sha_file(config.SOURCES_JSON),
        },
        "decision_instrument": {
            "file": rel(config.DILEMMAS_V2),
            "n_dilemmas": len(dilemmas),
            "ids": [d["id"] for d in dilemmas],
            "stance_counts": stances,
        },
        "prompts": list(config.DEFAULT_PROMPTS),
        "judge": {
            "model": judge.JUDGE_MODEL,
            "dimensions": list(judge.DIMENSIONS),
        },
        "calibration": {
            "cells": list(cal.CELLS),
            "required_fields": list(cal.REQUIRED_FIELDS),
            "topic_axes": list(cal.TOPIC_AXES),
            "phrasings": list(cal.PHRASINGS),
            "stances": list(cal.STANCES),
        },
        "criteria": {
            "dilemma_baseline": config.DILEMMA_BASELINE,
            "exp9_content": {k: list(v) for k, v in config.EXP9_CONTENT.items()},
            "exp3b_style": dict(config.EXP3B_STYLE),
        },
        "pass_b": {
            "pair_authors": pairs.PAIR_AUTHORS,
            "n_pairs": pairs.N_PAIRS,
            "pair_model": pairs.PAIR_MODEL,
            "lora_config": lora.LORA_CONFIG,
        },
    }


# --- artifact 9: corpus chunking (no model, no network) -------------------

def collect_corpus_chunks() -> dict:
    """Re-chunk the already-downloaded processed texts and digest the result.

    Pure function over local files: no re-download, so this isolates
    `chunk_paragraphs` from Gutenberg availability. The on-disk chunk JSON is
    digested alongside, so a snapshot also records whether the two agree.
    """
    from stoic import config, corpus

    out = {}
    for txt in sorted(config.GEN_PROCESSED_DIR.glob("*/*.txt")):
        author, stem = txt.parent.name, txt.stem
        paras = corpus.chunk_paragraphs(txt.read_text(encoding="utf-8"))
        entry = {
            "n_chunks_recomputed": len(paras),
            "chunks_sha256": sha("\n\x00\n".join(paras)),
            "source_text_sha256": sha_file(txt),
        }
        on_disk = config.GEN_CHUNKED_DIR / author / f"{stem}.json"
        if on_disk.exists():
            payload = json.load(open(on_disk))
            texts = [c["text"] for c in payload["chunks"]]
            entry["n_chunks_on_disk"] = payload["total_chunks"]
            entry["on_disk_chunks_sha256"] = sha("\n\x00\n".join(texts))
            entry["matches_on_disk"] = (
                entry["chunks_sha256"] == entry["on_disk_chunks_sha256"]
            )
        out[f"{author}/{stem}"] = entry
    return out


# --- artifact 12: test suite ----------------------------------------------

def collect_pytest() -> dict:
    """Run the suite and record pass/fail plus the collected-test count.

    The count matters: a refactor that silently drops a test looks green.
    The comparison rule (see `compare`) fails on a drop, never on an increase.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped|xfailed)", proc.stdout)}
    return {
        "returncode": proc.returncode,
        "summary_line": tail,
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "errors": counts.get("errors", counts.get("error", 0)),
        "skipped": counts.get("skipped", 0),
    }


# --- model-dependent artifacts --------------------------------------------

def collect_stage0(model, tokenizer) -> dict:
    from stoic import config
    from stoic.model import generate

    prompt = config.DEFAULT_PROMPTS[0]
    out1 = generate(model, tokenizer, prompt)
    out2 = generate(model, tokenizer, prompt)
    return {"prompt": prompt, "output_1": out1, "output_2": out2, "identical": out1 == out2}


def collect_stage1(model, tokenizer, dilemmas) -> dict:
    from stoic.dilemmas import eval_dilemmas, mean

    baseline = eval_dilemmas(model, tokenizer, dilemmas)
    return {"baseline_p_stoic": baseline, "baseline_mean": mean(baseline)}


def _injection_site(model, tokenizer, layer, vector, coeff) -> dict:
    """Inlined from stages/verify.py on purpose: the snapshot must not depend
    on stage internals, which are exactly what the refactor reorganizes."""
    import torch

    from stoic.steering import steering

    inputs = tokenizer("The wise person is one who", return_tensors="pt").to(model.device)

    def hs():
        with torch.no_grad():
            return model(**inputs, output_hidden_states=True).hidden_states

    clean = hs()
    with steering(model, layer, vector, coeff):
        steered = hs()
    return {
        "layer": layer,
        "hidden_states[L]_unchanged": torch.equal(clean[layer], steered[layer]),
        "hidden_states[L+1]_changed": not torch.equal(clean[layer + 1], steered[layer + 1]),
    }


def collect_stage2(model, tokenizer, dilemmas, baseline) -> tuple[dict, dict]:
    import torch

    from stoic import config
    from stoic.dilemmas import deltas_by_stance, eval_dilemmas, mean, paired_stats, _logit
    from stoic.steering import extract_vector, load_pairs, load_reference_vector

    cos = torch.nn.functional.cosine_similarity
    per_arm = {}
    for name, arm in config.AUTHORS.items():
        print(f"  [snapshot] stage2 {name} L{arm.layer}")
        pairs_ = load_pairs(arm.pairs_file)
        new_vec = extract_vector(model, tokenizer, pairs_, arm.layer)
        ref_vec = load_reference_vector(arm.vector_file, arm.layer)
        steered = eval_dilemmas(
            model, tokenizer, dilemmas, steer=(arm.layer, new_vec, arm.coeff)
        )
        deltas = {i: steered[i] - baseline[i] for i in steered}
        deltas_lo = {i: _logit(steered[i]) - _logit(baseline[i]) for i in steered}
        per_arm[name] = {
            "layer": arm.layer,
            "coeff": arm.coeff,
            "cosine_to_frozen": cos(
                new_vec.float().unsqueeze(0), ref_vec.float().unsqueeze(0)
            ).item(),
            "norm_ratio_new_over_frozen": (
                new_vec.float().norm() / ref_vec.float().norm()
            ).item(),
            "steered_mean": mean(steered),
            "steered_p_stoic": steered,
            "overall": paired_stats(list(deltas.values())),
            "overall_logodds": paired_stats(list(deltas_lo.values())),
            "by_stance": deltas_by_stance(dilemmas, deltas),
        }

    epi = config.AUTHORS["epictetus"]
    site = _injection_site(
        model, tokenizer, epi.layer,
        load_reference_vector(epi.vector_file, epi.layer), epi.coeff,
    )
    return per_arm, site


def collect_stage4(model, tokenizer, dilemmas, baseline) -> dict:
    from stoic import config, lora
    from stoic.dilemmas import (
        deltas_by_stance, eval_dilemmas, mean, paired_stats, sign_test, _logit,
    )

    per_arm = {}
    for name, arm in config.AUTHORS.items():
        print(f"  [snapshot] stage4 {name} ({arm.adapter_dir.name})")
        merged = lora.merge_adapter(arm.adapter_dir)
        try:
            steered = eval_dilemmas(merged, tokenizer, dilemmas)
        finally:
            del merged
            gc.collect()
        deltas = {i: steered[i] - baseline[i] for i in steered}
        deltas_lo = {i: _logit(steered[i]) - _logit(baseline[i]) for i in steered}
        per_arm[name] = {
            "adapter": arm.adapter_dir.name,
            "steered_mean": mean(steered),
            "steered_p_stoic": steered,
            "overall": paired_stats(list(deltas.values())),
            "overall_logodds": paired_stats(list(deltas_lo.values())),
            "by_stance": deltas_by_stance(dilemmas, deltas),
            "sign_test": sign_test(deltas),
        }

    baseline_end = eval_dilemmas(model, tokenizer, dilemmas)
    return {
        "per_arm": per_arm,
        "baseline_mean_end": mean(baseline_end),
        "max_baseline_drift": max(abs(baseline_end[i] - baseline[i]) for i in baseline),
    }


def collect_calibration(model, tokenizer) -> dict:
    from stoic import config
    from stoic.calibrate import calibration_report, load_candidates, validate_items
    from stoic.dilemmas import eval_dilemmas

    path = config.GENERATED_DIR / "dilemmas_v3_candidates.json"
    if not path.exists():
        return {"__missing__": f"{rel(path)} not found"}
    items = load_candidates(path)
    problems = validate_items(items)
    scores = eval_dilemmas(model, tokenizer, items)
    return {
        "items_file": rel(path),
        "structural_problems": problems,
        "scores": scores,
        "report": calibration_report(items, scores),
    }


def collect_greedy_replay(model, tokenizer, arm_name: str) -> dict:
    """Regenerate the Stage-3 greedy texts — the exact strings the judge saw.

    The judge's own scores are not reproducible (nondeterministic API, costs
    money), but its *input* is, and that input is everything the refactor can
    reach: prompt set, canonical decoding, and steering during generation.
    """
    from stoic import config
    from stoic.model import generate
    from stoic.steering import load_reference_vector, steering

    arm = config.AUTHORS[arm_name]
    prompts = config.DEFAULT_PROMPTS
    print(f"  [snapshot] greedy replay {arm_name}: {len(prompts)} baseline + {len(prompts)} steered")
    baseline = [generate(model, tokenizer, p) for p in prompts]
    vector = load_reference_vector(arm.vector_file, arm.layer)
    with steering(model, arm.layer, vector, arm.coeff):
        steered = [generate(model, tokenizer, p) for p in prompts]
    return {"arm": arm_name, "prompts": list(prompts),
            "baseline_outputs": baseline, "steered_outputs": steered}


def collect_sampled_replay(model, tokenizer, arm_name: str, seed: int = 4) -> dict:
    """Regenerate the Stage-3 sampled steered texts for one seed.

    `generate_all_sampled` reseeds at entry, so seed 4 reproduces standalone
    without replaying seeds 0-3.
    """
    import torch

    from stoic import config
    from stoic.model import generate
    from stoic.steering import load_reference_vector, steering

    arm = config.AUTHORS[arm_name]
    prompts = config.DEFAULT_PROMPTS
    print(f"  [snapshot] sampled replay {arm_name} seed {seed}: {len(prompts)} generations")
    vector = load_reference_vector(arm.vector_file, arm.layer)
    torch.manual_seed(seed)
    with steering(model, arm.layer, vector, arm.coeff):
        steered = [
            generate(model, tokenizer, p, do_sample=True, temperature=0.6,
                     top_p=0.9, max_new_tokens=100)
            for p in prompts
        ]
    return {"arm": arm_name, "seed": seed, "decoding":
            {"do_sample": True, "temperature": 0.6, "top_p": 0.9, "max_new_tokens": 100},
            "steered_outputs_last_seed": steered}


# --- orchestration --------------------------------------------------------

def preflight(cheap: bool, replay: str | None) -> list[str]:
    """Everything the run needs, checked before the first forward pass.

    The full snapshot costs hours; discovering a missing import at the last
    artifact wastes all of it. Every dependency this run will touch is proven
    reachable here, cheaply, up front.
    """
    from stoic import config

    problems = []
    if not config.DILEMMAS_V2.exists():
        problems.append(f"decision instrument missing: {rel(config.DILEMMAS_V2)}")
    if cheap:
        return problems

    for name, arm in config.AUTHORS.items():
        if not arm.pairs_file.exists():
            problems.append(f"{name}: pairs file missing ({rel(arm.pairs_file)})")
        if not arm.vector_file.exists():
            problems.append(f"{name}: steering vector missing ({rel(arm.vector_file)}) "
                            "— run scripts/fetch_artifacts.py")
        if not arm.adapter_dir.exists():
            problems.append(f"{name}: adapter missing ({rel(arm.adapter_dir)}) "
                            "— run scripts/fetch_artifacts.py")
    try:
        import peft  # noqa: F401  (artifact 4 merges adapters through it)
    except ImportError:
        problems.append("peft not importable — artifact 4 (stage4 LoRA merges) "
                        "cannot run. Install the repo's own extra: pip install -e '.[lora]'")
    if not (config.GENERATED_DIR / "dilemmas_v3_candidates.json").exists():
        problems.append("dilemmas_v3_candidates.json missing — artifact 8 cannot run")
    if replay and replay not in config.AUTHORS:
        problems.append(f"--replay {replay!r} is not an arm: {sorted(config.AUTHORS)}")
    return problems


def build_snapshot(cheap: bool, replay: str | None, out: Path) -> dict:
    """Collect every artifact, writing the partial snapshot to `out` after each.

    Each artifact is isolated: a failure records `__error__` and the run
    continues. A late failure must never destroy the artifacts already paid
    for — the first version of this script lost ~2.5 h of completed work to a
    missing import in the final step.
    """
    snap: dict = {"artifacts": {}, "mode": "cheap" if cheap else "full",
                  "replay_arm": replay, "collected": []}
    art = snap["artifacts"]

    def step(num: int, label: str, fn):
        print(f"[snapshot] artifact {num}: {label}")
        snap["collected"] = sorted(set(snap["collected"]) | {num})
        try:
            art[ARTIFACTS[num]] = fn()
        except Exception as e:  # isolate: one artifact must not sink the rest
            art[ARTIFACTS[num]] = {"__error__": f"{type(e).__name__}: {e}"}
            print(f"  ✗ artifact {num} failed: {type(e).__name__}: {e}")
        _write(snap, out)      # checkpoint after every artifact
        return art[ARTIFACTS[num]]

    step(11, "config surface", collect_config_surface)
    step(9, "corpus chunking", collect_corpus_chunks)
    step(12, "pytest", collect_pytest)
    if cheap:
        return snap

    from stoic.dilemmas import load_dilemmas
    from stoic.model import load_model

    model, tokenizer = load_model()
    dilemmas = load_dilemmas()

    step(5, "stage0 determinism", lambda: collect_stage0(model, tokenizer))
    s1 = step(1, "stage1 baseline (40 items x 2 label orders)",
              lambda: collect_stage1(model, tokenizer, dilemmas))
    baseline = s1.get("baseline_p_stoic")

    if baseline is None:
        print("  ✗ stage1 produced no baseline — stages 2 and 4 depend on it; skipping both")
    else:
        holder: dict = {}

        def _stage2():
            per_arm, site = collect_stage2(model, tokenizer, dilemmas, baseline)
            holder["site"] = site
            return per_arm

        step(2, "stage2 vectors", _stage2)
        step(3, "stage2 injection site", lambda: holder.get("site", {}))

    step(8, "dilemmas_v3 calibration", lambda: collect_calibration(model, tokenizer))

    if replay:
        step(6, f"stage3 greedy replay ({replay})",
             lambda: collect_greedy_replay(model, tokenizer, replay))
        step(7, f"stage3 sampled replay ({replay})",
             lambda: collect_sampled_replay(model, tokenizer, replay))

    if baseline is not None:
        step(4, "stage4 LoRA merges (fresh base per adapter)",
             lambda: collect_stage4(model, tokenizer, dilemmas, baseline))
    return snap


def _write(snap: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(enc(snap), f, indent=2, sort_keys=True)


def completeness_problems(snap: dict) -> list[str]:
    """Name every artifact that is absent or hollow.

    A partial before.json silently narrows the whole verification, and it
    cannot be recaptured once the refactor lands — so this exits nonzero
    rather than warning.
    """
    expected = set(snap.get("collected", []))
    problems = []
    for num in sorted(expected):
        key = ARTIFACTS[num]
        val = snap["artifacts"].get(key)
        if not val:
            problems.append(f"artifact {num} ({key}): missing or empty")
            continue
        if isinstance(val, dict) and "__missing__" in val:
            problems.append(f"artifact {num} ({key}): {val['__missing__']}")
        if isinstance(val, dict) and "__error__" in val:
            problems.append(f"artifact {num} ({key}): {val['__error__']}")

    surface = snap["artifacts"].get(ARTIFACTS[11], {})
    for name, arm in surface.get("arms", {}).items():
        for field in ("pairs_sha256", "vector_sha256", "adapter_weights_sha256"):
            if arm.get(field) is None:
                problems.append(f"artifact 11: {name}.{field} is null (file absent)")

    tests = snap["artifacts"].get(ARTIFACTS[12], {})
    if tests.get("returncode") != 0:
        problems.append(f"artifact 12: pytest failed ({tests.get('summary_line')!r})")
    return problems


# --- comparison -----------------------------------------------------------

def diff(a, b, path="") -> list[tuple[str, object, object]]:
    if type(a) is not type(b):
        return [(path, a, b)]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append((f"{path}.{k}", "<absent>", b[k]))
            elif k not in b:
                out.append((f"{path}.{k}", a[k], "<absent>"))
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        out = []
        if len(a) != len(b):
            out.append((f"{path}.__len__", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [(path, a, b)]


def compare(before: dict, after: dict) -> int:
    """before vs after. Artifact 12 is special-cased: a *higher* test count is
    expected whenever a step adds tests, so only a failure or a drop counts."""
    b_art, a_art = before["artifacts"], after["artifacts"]
    shared = [n for n in sorted(ARTIFACTS) if ARTIFACTS[n] in b_art and ARTIFACTS[n] in a_art]
    only_before = [ARTIFACTS[n] for n in ARTIFACTS if ARTIFACTS[n] in b_art and ARTIFACTS[n] not in a_art]

    failures, notes = [], []
    for num in shared:
        key = ARTIFACTS[num]
        if num == 12:
            bt, at = b_art[key], a_art[key]
            if at.get("returncode") != 0:
                failures.append((num, key, [("pytest", bt.get("summary_line"), at.get("summary_line"))]))
            elif at.get("passed", 0) < bt.get("passed", 0):
                failures.append((num, key, [("passed_count_dropped", bt.get("passed"), at.get("passed"))]))
            else:
                delta = at.get("passed", 0) - bt.get("passed", 0)
                notes.append(f"artifact 12: {at.get('passed')} passed "
                             f"({'+' if delta >= 0 else ''}{delta} vs before) — increase is expected, not a diff")
            continue
        d = diff(b_art[key], a_art[key], key)
        if d:
            failures.append((num, key, d))

    print(f"compared {len(shared)} artifact(s): "
          + ", ".join(f"{n}={ARTIFACTS[n]}" for n in shared))
    if only_before:
        print(f"\n⚠ present in before, absent in after (not compared): {', '.join(only_before)}")
    for note in notes:
        print(f"  · {note}")

    if not failures:
        print("\nIDENTICAL — every compared artifact is bit-for-bit equal.")
        return 0

    print(f"\nDIFFERENT — {len(failures)} artifact(s) changed:")
    for num, key, d in failures:
        print(f"\n  artifact {num} ({key}): {len(d)} differing field(s)")
        for path, x, y in d[:12]:
            print(f"    {path}\n      before: {x!r}\n      after:  {y!r}")
        if len(d) > 12:
            print(f"    ... and {len(d) - 12} more")
    return 1


# --- comparison against the checked-in results record ---------------------

def _latest(pattern: str) -> Path | None:
    hits = sorted(ROOT.glob(pattern))
    return hits[-1] if hits else None


def check_published(snap: dict) -> int:
    """Snapshot vs the checked-in results/ JSONs. This is the *environment*
    check, reported separately from --compare on purpose: if --compare is clean
    and this is not, the drift predates the refactor."""
    art = snap["artifacts"]
    checks: list[tuple[str, Path | None, object, object]] = []

    f = _latest("results/stage1_dilemma_baseline/baseline_*.json")
    if f and ARTIFACTS[1] in art:
        pub = json.load(open(f))
        checks.append(("stage1 per-item P(stoic)", f,
                       art[ARTIFACTS[1]]["baseline_p_stoic"], pub["baseline_p_stoic"]))
        checks.append(("stage1 mean", f,
                       art[ARTIFACTS[1]]["baseline_mean"], pub["baseline_mean"]))

    f = _latest("results/stage2_steering/steering_*.json")
    if f and ARTIFACTS[2] in art:
        pub = json.load(open(f))
        for name, got in art[ARTIFACTS[2]].items():
            want = pub["per_author"].get(name, {})
            subset = {k: got[k] for k in
                      ("cosine_to_frozen", "norm_ratio_new_over_frozen", "steered_mean",
                       "overall", "overall_logodds", "by_stance") if k in got}
            checks.append((f"stage2 {name}", f, subset,
                           {k: want[k] for k in subset if k in want}))
        if ARTIFACTS[3] in art:
            site = {k: v for k, v in art[ARTIFACTS[3]].items()}
            pub_site = {k: v for k, v in pub["injection_site_check"].items() if k in site}
            checks.append(("stage2 injection site", f, site, pub_site))

    f = _latest("results/stage4_lora_dilemmas/lora_dilemmas_*.json")
    if f and ARTIFACTS[4] in art:
        pub = json.load(open(f))
        for name, got in art[ARTIFACTS[4]]["per_arm"].items():
            want = pub["per_author"].get(name, {})
            subset = {k: got[k] for k in
                      ("steered_mean", "steered_p_stoic", "overall", "overall_logodds",
                       "by_stance", "sign_test") if k in got}
            checks.append((f"stage4 {name}", f, subset,
                           {k: want[k] for k in subset if k in want}))

    f = _latest("results/stage0_determinism/determinism_*.json")
    if f and ARTIFACTS[5] in art:
        pub = json.load(open(f))
        checks.append(("stage0 outputs", f,
                       {k: art[ARTIFACTS[5]][k] for k in ("output_1", "output_2")},
                       {k: pub[k] for k in ("output_1", "output_2")}))

    f = _latest("results/stage3_content_judge/content_2*.json")
    if f and ARTIFACTS[6] in art:
        pub = json.load(open(f))
        arm = art[ARTIFACTS[6]]["arm"]
        want = pub["per_author"].get(arm, {})
        checks.append((f"stage3 greedy generations ({arm})", f,
                       {k: art[ARTIFACTS[6]][k] for k in ("baseline_outputs", "steered_outputs")},
                       {k: want.get(k) for k in ("baseline_outputs", "steered_outputs")}))

    f = _latest("results/stage3_content_judge/content_sampled_*.json")
    if f and ARTIFACTS[7] in art:
        pub = json.load(open(f))
        arm = art[ARTIFACTS[7]]["arm"]
        want = pub["per_author"].get(arm, {})
        checks.append((f"stage3 sampled generations ({arm}, seed 4)", f,
                       art[ARTIFACTS[7]]["steered_outputs_last_seed"],
                       want.get("steered_outputs_last_seed")))

    f = _latest("results/dilemmas_v3_calibration/calibration_*.json")
    if f and ARTIFACTS[8] in art and "__missing__" not in art[ARTIFACTS[8]]:
        pub = json.load(open(f))
        got = art[ARTIFACTS[8]]["report"]
        checks.append(("v3 calibration report", f,
                       {k: got[k] for k in ("per_cell", "overall_mean_p_stoic",
                                            "outlier_items", "calibration_gate_passed")
                        if k in got},
                       {k: pub[k] for k in ("per_cell", "overall_mean_p_stoic",
                                            "outlier_items", "calibration_gate_passed")
                        if k in pub}))

    if ARTIFACTS[9] in art:
        for key, entry in art[ARTIFACTS[9]].items():
            if "on_disk_chunks_sha256" in entry:
                checks.append((f"corpus chunks {key}",
                               Path("data/generated/chunked"),
                               entry["chunks_sha256"], entry["on_disk_chunks_sha256"]))

    if not checks:
        print("nothing to check — snapshot has no artifacts with a published counterpart")
        return 0

    n_diff = 0
    print(f"snapshot vs checked-in record — {len(checks)} comparison(s)\n")
    for label, src, got, want in checks:
        if want is None:
            print(f"  SKIP   {label}  (no published counterpart)")
            continue
        d = diff(enc(got), enc(want), label)
        if not d:
            print(f"  MATCH  {label}   [{rel(src) if src else '?'}]")
        else:
            n_diff += 1
            print(f"  DIFFER {label}   [{rel(src) if src else '?'}] — {len(d)} field(s)")
            for path, x, y in d[:6]:
                print(f"           {path}\n             snapshot:  {x!r}\n             published: {y!r}")
            if len(d) > 6:
                print(f"           ... and {len(d) - 6} more")

    print(f"\n{len(checks) - n_diff} match, {n_diff} differ.")
    if n_diff:
        print("Differences here are NOT automatically refactor damage — compare "
              "before.json ↔ after.json first. If that is clean, this is "
              "pre-existing environment drift (pyproject pins torch==2.5.1).")
    return 0


# --- CLI ------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(prog="axis_snapshot", description=__doc__.splitlines()[0])
    p.add_argument("--out", help="write a snapshot to this path")
    p.add_argument("--replay", metavar="ARM",
                   help="also replay Stage-3 generations for this arm (e.g. seneca)")
    p.add_argument("--cheap", action="store_true",
                   help="no model: collect only artifacts 11, 9, 12 (~1 min)")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="run even if preflight finds missing dependencies")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                   help="diff two snapshots; exits nonzero on any difference")
    p.add_argument("--check-published", metavar="SNAPSHOT",
                   help="compare a snapshot against the checked-in results/ JSONs")
    args = p.parse_args()

    if args.compare:
        before = json.load(open(args.compare[0]))
        after = json.load(open(args.compare[1]))
        return compare(before, after)

    if args.check_published:
        return check_published(json.load(open(args.check_published)))

    if not args.out:
        p.error("one of --out, --compare, --check-published is required")

    pre = preflight(cheap=args.cheap, replay=args.replay)
    if pre:
        print(f"PREFLIGHT FAILED — {len(pre)} problem(s), nothing run:")
        for pr in pre:
            print(f"  ✗ {pr}")
        print("\nThe full snapshot costs hours; these are checked up front so a "
              "missing dependency cannot surface at the last artifact.")
        if not args.allow_incomplete:
            return 2
        print("\n--allow-incomplete: continuing anyway; affected artifacts will "
              "record __error__ and the snapshot will be marked INCOMPLETE.")

    out = Path(args.out)
    snap = build_snapshot(cheap=args.cheap, replay=args.replay, out=out)
    print(f"\n↳ wrote {rel(out)}")

    problems = completeness_problems(snap)
    print(f"\ncompleteness: {len(snap['collected'])} artifact(s) requested "
          f"({', '.join(str(n) for n in snap['collected'])})")
    if problems:
        print(f"INCOMPLETE — {len(problems)} problem(s):")
        for pr in problems:
            print(f"  ✗ {pr}")
        print("\nA partial snapshot silently narrows the verification. Fix these "
              "before relying on it as a baseline.")
        return 1
    print("COMPLETE — every requested artifact present and non-empty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
