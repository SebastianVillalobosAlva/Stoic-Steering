"""The axis config object: schema validation, and the binding the CLI depends on.

Two things are pinned here.

1. **The stoic axis still says exactly what config.py used to say.** The axis
   files are a new source for old constants; if they drift, every published
   number silently belongs to a different configuration. The prompt/rubric
   digests are pinned to the values captured from the pre-refactor tree.

2. **`--axis` and BEHAVIOR_AXIS resolve the same axis.** The axis is bound at
   import, so argparse cannot decide it — a flag parsed after `stoic` is
   imported would look like it worked while doing nothing. That failure is
   invisible to any check that only exercises the env var, so it is checked
   here on both paths, in-process and through a real CLI dispatch.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stoic import axis as A
from stoic.axis import ACTIVE, AxisError, load_axis

ROOT = Path(__file__).resolve().parent.parent

# Digests of the verbatim text files, captured from the tree BEFORE the
# refactor (.axis_snap/before.json -> config_surface.digests). These are the
# whole point of keeping the prompts as files: drift is visible.
RUBRIC_SHA = "bec2de3cb5f09d8e753097e860f5d51873819aa916bf5f1694fccbe3b9f8db89"
PROMPTS_SHA = "3433cf092848a5824742e0720edfc8e953e2827bb9fc12b441ce99282b6681af"


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the stoic axis reproduces the historical constants -------------------

def test_default_axis_is_stoic():
    assert ACTIVE.name == "stoic"


def test_canonical_layers_and_coeff():
    """CLAUDE.md ground truth: Marcus L26, Seneca L4, Epictetus L8, coeff 0.11."""
    assert ACTIVE.arms["marcus"].layer == 26
    assert ACTIVE.arms["seneca"].layer == 4
    assert ACTIVE.arms["epictetus"].layer == 8
    assert all(a.coeff == 0.11 for a in ACTIVE.arms.values())


def test_arm_layers_are_within_the_model():
    from stoic import config

    assert all(0 <= a.layer < config.NUM_LAYERS for a in ACTIVE.arms.values())


def test_criteria_match_published_numbers():
    c = ACTIVE.criteria
    assert c["decision_baseline"] == 0.542
    assert c["decision_baseline_exact"] == 0.541601902275579
    assert c["cosine_min"] == 0.99
    assert c["injection_site_arm"] == "epictetus"
    assert c["stage4_criterion_arm"] == "seneca"


def test_reference_targets_match_exp9_and_exp3b():
    t = ACTIVE.reference_targets
    assert t["exp9_content"] == {"marcus": [0.408, 0.136], "seneca": [0.583, 0.121],
                                 "epictetus": [0.767, 0.076]}
    assert t["exp3b_style"] == {"marcus": 1.00, "seneca": 1.42, "epictetus": 1.58}


def test_verbatim_text_files_have_not_drifted():
    assert _sha(ACTIVE.judge_rubric) == RUBRIC_SHA
    assert _sha("\n".join(ACTIVE.prompts)) == PROMPTS_SHA
    assert len(ACTIVE.prompts) == 12


def test_dilemma_field_map_matches_the_frozen_v2_set():
    f = ACTIVE.fields
    assert (f.target, f.foil, f.stance) == ("stoic", "nonstoic", "stoic_stance")
    items = json.load(open(ACTIVE.dilemmas_file))[ACTIVE.dilemmas_collection_key]
    assert len(items) == 40
    for d in items:
        assert d[f.target] and d[f.foil] and d[f.stance]


def test_arm_paths_resolve_where_they_always_did():
    a = ACTIVE.arms["marcus"]
    assert ACTIVE.pairs_file(a) == A.REFERENCE_DIR / "processed/marcus_aurelius/neutral_pairs.json"
    assert ACTIVE.vector_file(a) == A.REFERENCE_DIR / "steering_vectors/marcus_aurelius_steering_3B.pt"
    assert ACTIVE.adapter_dir(a) == A.MODELS_DIR / "lora_marcus_clean"
    assert ACTIVE.pairs_file("seneca") == ACTIVE.pairs_file(ACTIVE.arms["seneca"])


def test_results_arm_key_is_unchanged_for_byte_identity():
    """Result JSONs keep saying `per_author` on the stoic axis; renaming it
    would break every checked-in comparison for no benefit."""
    assert ACTIVE.results_arm_key == "per_author"


def test_reference_inputs_are_read_only():
    for a in ACTIVE.arms.values():
        assert ACTIVE.is_read_only(ACTIVE.pairs_file(a))
        assert ACTIVE.is_read_only(ACTIVE.vector_file(a))
        assert not ACTIVE.is_read_only(ACTIVE.adapter_dir(a))


# --- a second axis loads with no code change ------------------------------

def test_example_axis_loads_and_uses_its_own_field_names():
    ex = load_axis("_example")
    assert ex.name == "_example"
    assert (ex.fields.target, ex.fields.foil, ex.fields.stance) == \
        ("honest", "flattering", "pressure")
    assert (ex.pair_fields.target, ex.pair_fields.foil) == \
        ("honest_text", "flattering_text")
    assert ex.results_arm_key == "per_arm"


def test_example_axis_is_self_contained():
    """The shipped template must not reach outside its own directory for data,
    or it is not a template."""
    ex = load_axis("_example")
    for p in (ex.dilemmas_file, ex.candidates_file, ex.pairs_file("agreeable"),
              ex.directory / ex.judge["rubric_file"]):
        assert p.is_relative_to(ex.directory), p
        assert p.exists(), p


def test_unknown_axis_names_what_is_available():
    with pytest.raises(AxisError) as e:
        load_axis("does_not_exist")
    assert "stoic" in str(e.value)


# --- validation refuses malformed configs ---------------------------------

def _write_axis(tmp_path, monkeypatch, mutate, name="tmpaxis"):
    """Copy the example axis, mutate its config, and load it from a temp tree.

    EVERY root moves to tmp_path together. Patching only PROJECT_ROOT leaves
    REFERENCE_DIR and friends pointing at the real repo, so every case trips
    the repo-escape check first and appears to pass for the wrong reason —
    which is exactly what the first version of this fixture did.
    """
    import shutil

    d = tmp_path / name
    shutil.copytree(A.AXES_DIR / "_example", d)
    cfg = json.loads((d / "axis.json").read_text())
    cfg["name"] = name
    mutate(cfg)
    (d / "axis.json").write_text(json.dumps(cfg))

    monkeypatch.setattr(A, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(A, "AXES_DIR", tmp_path)
    monkeypatch.setattr(A, "REFERENCE_DIR", tmp_path / "data" / "reference")
    monkeypatch.setattr(A, "GENERATED_DIR", tmp_path / "data" / "generated")
    monkeypatch.setattr(A, "MODELS_DIR", tmp_path / "models")
    load_axis.cache_clear()
    try:
        return load_axis(name)
    finally:
        load_axis.cache_clear()


BAD_CONFIGS = [
    ("bad schema_version", lambda c: c.update(schema_version=99), "schema_version"),
    ("stoic-only corpus kind", lambda c: c["pass_b"].update(corpus={"kind": "gutenberg"}), "stoic-only"),
    ("stoic-only pairs kind", lambda c: c["pass_b"].update(pairs={"kind": "corpus_contrastive"}), "stoic-only"),
    ("dilemma field map wrong", lambda c: c["decision_instrument"]["fields"].update(target="nope"), "fields"),
    ("pair field map wrong", lambda c: c["contrast_pairs"]["fields"].update(target="nope"), "contrast_pairs.fields"),
    ("unknown path root", lambda c: c["paths"]["pairs"].update(root="elsewhere"), "unknown root"),
    ("criteria names ghost arm", lambda c: c["criteria"].update(stage4_criterion_arm="ghost"), "not an arm"),
    ("content dim not in dims", lambda c: c["judge"].update(content_dimensions=["nope"]), "content_dimensions"),
    ("negative layer", lambda c: c["arms"]["agreeable"].update(layer=-1), "non-negative"),
    ("no arms", lambda c: c.update(arms={}), "at least one arm"),
    ("missing rubric file", lambda c: c["judge"].update(rubric_file="missing.txt"), "does not exist"),
]


@pytest.mark.parametrize("label,mutate,expected", BAD_CONFIGS, ids=[b[0] for b in BAD_CONFIGS])
def test_malformed_axis_is_rejected(tmp_path, monkeypatch, label, mutate, expected):
    with pytest.raises(AxisError) as e:
        _write_axis(tmp_path, monkeypatch, mutate)
    assert expected in str(e.value), f"{label}: unexpected message {e.value}"


def test_name_must_match_directory(tmp_path, monkeypatch):
    with pytest.raises(AxisError) as e:
        _write_axis(tmp_path, monkeypatch, lambda c: c.update(name="mismatch"))
    assert "lives in" in str(e.value)


def test_path_cannot_escape_the_repo(tmp_path, monkeypatch):
    with pytest.raises(AxisError) as e:
        _write_axis(tmp_path, monkeypatch,
                    lambda c: c["paths"]["pairs"].update(template="../../../escape.json"))
    assert "escapes the repo" in str(e.value)


# --- the binding: flag and env var must agree -----------------------------

def test_package_init_does_not_import_config():
    """`import stoic` must not pull in config, or the axis binds before the CLI
    has had a chance to set BEHAVIOR_AXIS from --axis."""
    tree = ast.parse((ROOT / "stoic" / "__init__.py").read_text())
    imported = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not imported, f"stoic/__init__.py must stay import-free, found {len(imported)}"


@pytest.mark.parametrize("argv", [
    ["stoic", "--axis", "_example", "stage1"],
    ["stoic", "--axis=_example", "stage1"],
])
def test_bind_axis_from_argv_sets_the_env_var(monkeypatch, argv):
    from stoic.__main__ import _bind_axis_from_argv

    monkeypatch.delenv("BEHAVIOR_AXIS", raising=False)
    assert _bind_axis_from_argv(argv) == "_example"
    assert os.environ["BEHAVIOR_AXIS"] == "_example"


def test_bind_axis_returns_none_without_the_flag(monkeypatch):
    from stoic.__main__ import _bind_axis_from_argv

    monkeypatch.delenv("BEHAVIOR_AXIS", raising=False)
    assert _bind_axis_from_argv(["stoic", "stage1"]) is None
    assert "BEHAVIOR_AXIS" not in os.environ


def test_flag_and_env_resolve_the_same_axis_in_process(monkeypatch):
    from stoic.__main__ import _bind_axis_from_argv

    monkeypatch.delenv("BEHAVIOR_AXIS", raising=False)
    _bind_axis_from_argv(["stoic", "--axis", "_example", "stage1"])
    via_flag = load_axis(os.environ["BEHAVIOR_AXIS"])

    monkeypatch.setenv("BEHAVIOR_AXIS", "_example")
    via_env = load_axis(os.environ["BEHAVIOR_AXIS"])

    assert via_flag is via_env
    assert via_flag.name == "_example"


def _cli_axis_line(env_extra: dict, args: list[str]) -> str:
    """Run a real CLI dispatch and read back which axis it actually loaded.

    `--help` is the probe on purpose: it exercises the true import order and
    argparse construction, needs no model, and costs nothing.
    """
    env = {**os.environ, "USE_TF": "0", **env_extra}
    env.pop("BEHAVIOR_AXIS", None)
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-m", "stoic", *args, "--help"],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    lines = [ln.strip() for ln in out.stdout.splitlines() if "active axis" in ln]
    assert lines, f"no active-axis line in --help output: {out.stdout[:400]}{out.stderr[:400]}"
    return lines[0]


def test_flag_and_env_resolve_the_same_axis_end_to_end():
    """The regression this whole binding path exists for: argparse parses
    --axis long after `stoic` is imported, so a flag handled only by argparse
    would silently no-op while BEHAVIOR_AXIS worked."""
    via_flag = _cli_axis_line({}, ["--axis", "_example"])
    via_env = _cli_axis_line({"BEHAVIOR_AXIS": "_example"}, [])
    default = _cli_axis_line({}, [])

    assert via_flag == via_env, f"flag {via_flag!r} != env {via_env!r}"
    assert "_example" in via_flag
    assert "stoic" in default and "_example" not in default


def test_conflicting_flag_and_env_is_an_error():
    env = {**os.environ, "USE_TF": "0", "BEHAVIOR_AXIS": "stoic"}
    out = subprocess.run([sys.executable, "-m", "stoic", "--axis", "_example", "stage1"],
                         cwd=ROOT, capture_output=True, text=True, env=env)
    assert out.returncode != 0
    assert "disagrees with" in (out.stdout + out.stderr)
