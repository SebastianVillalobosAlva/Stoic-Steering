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
