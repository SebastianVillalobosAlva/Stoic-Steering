"""Static tripwire: nothing in the stoic package may open a reference path
for writing. CLAUDE.md's rule — 'if a stage is about to write into reference/,
that is a bug' — enforced by AST scan, so it fails at test time instead of
after the frozen artifacts are gone."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "stoic"
REFERENCE_MARKERS = ("REF", "reference")


def _writeful_open_targets(tree):
    """Yield (lineno, unparsed-path) for every open(...) with a write/append mode
    and every .write_text/.write_bytes/torch.save call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if fname == "open" and node.args:
            mode = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                yield node.lineno, ast.unparse(node.args[0])
        elif fname in ("write_text", "write_bytes"):
            yield node.lineno, ast.unparse(node.func.value)
        elif fname == "save" and isinstance(node.func, ast.Attribute):
            owner = getattr(node.func.value, "id", None)
            if owner == "torch" and len(node.args) >= 2:
                yield node.lineno, ast.unparse(node.args[1])


def test_no_module_writes_into_reference():
    offenders = []
    for py in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for lineno, target in _writeful_open_targets(tree):
            if any(m in target for m in REFERENCE_MARKERS):
                offenders.append(f"{py.name}:{lineno} writes to {target}")
    assert not offenders, "reference/ is READ-ONLY:\n" + "\n".join(offenders)


def test_no_axis_declares_a_writable_path_into_reference():
    """The reference wall now has a second face: an axis config could point a
    pipeline OUTPUT at data/reference/ without any module containing a literal
    reference path, so the AST scan above would never see it.

    Read-only inputs (pairs, vectors, the dilemma set) may live under
    reference/. Anything the pipeline writes may not.
    """
    import json

    from stoic import axis as A

    WRITE_PATH_KEYS = ("gen_pairs", "gen_chunks", "output", "candidates_file")
    offenders = []
    for axis_json in sorted(A.AXES_DIR.glob("*/axis.json")):
        cfg = json.loads(axis_json.read_text())
        specs = {f"paths.{k}": v for k, v in cfg.get("paths", {}).items()}
        specs["calibration.candidates_file"] = cfg.get("calibration", {}).get("candidates_file")
        for where, spec in specs.items():
            if not spec:
                continue
            key = where.rsplit(".", 1)[-1]
            if key in WRITE_PATH_KEYS and spec.get("root") == "reference":
                offenders.append(f"{axis_json.parent.name}: {where} writes into reference/")
    assert not offenders, "reference/ is READ-ONLY:\n" + "\n".join(offenders)


def test_every_axis_loads():
    """A broken axis config is a startup failure for anyone who selects it;
    catch it here rather than three stages into a run."""
    from stoic import axis as A

    for axis_json in sorted(A.AXES_DIR.glob("*/axis.json")):
        A.load_axis(axis_json.parent.name)


def test_write_result_goes_under_results(tmp_path, monkeypatch):
    from stoic import config
    from stoic.results_io import write_result

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    path = write_result("stageX", "check", {"passed": True})
    assert path.is_relative_to(tmp_path / "results")
    assert path.exists()
    import json

    payload = json.load(open(path))
    assert payload["passed"] is True and "timestamp" in payload
