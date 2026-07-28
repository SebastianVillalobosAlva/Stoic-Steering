"""Behavioral axis as a config object — the thing under test is the *locus*.

This repo compares installing a behavior in weight-space against activation-space.
The behavior itself is the instrument: Stoic corpora build one axis, sycophancy
will build another. An axis is therefore data, not code — `axes/<name>/axis.json`
plus its verbatim prompt files — so adding one is a config directory rather than
an edit to the pipeline.

    axes/stoic/       the philosopher axis: three arms, the v2 dilemma ruler
    axes/_example/    self-contained template, the starting point for sycophancy

Which axis is active is fixed once, at import, from `STOIC_AXIS`. The CLI sets
that variable from `--axis` *before* importing anything under `stoic.` (see
`stoic/__main__.py`) — argparse runs far too late to decide it, and a flag that
silently disagreed with the loaded axis would be worse than no flag.

This module deliberately imports nothing from `stoic.config`: config depends on
the axis, not the reverse, so the path roots live here and config re-exports
them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# --- Roots ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"    # FROZEN. Read-only. Never write here.
GENERATED_DIR = DATA_DIR / "generated"    # everything the pipeline produces
MODELS_DIR = PROJECT_ROOT / "models"
AXES_DIR = PROJECT_ROOT / "axes"

SCHEMA_VERSIONS = (1,)
DEFAULT_AXIS = "stoic"
AXIS_ENV_VAR = "STOIC_AXIS"

# `axis` resolves against the axis's own directory, which is what lets an axis
# be self-contained. `reference` is readable only — the wall CLAUDE.md draws.
READ_ONLY_ROOTS = ("reference",)

# Pass-B generation is still hardcoded to the stoic corpus (deliberate scope
# cut). Any other axis declaring these kinds is a config error, caught at load
# rather than three stages into a run.
STOIC_ONLY_PASS_B = {"corpus": ("gutenberg",), "pairs": ("corpus_contrastive",)}


class AxisError(ValueError):
    """A malformed or impossible axis config. Always names the file."""


@dataclass(frozen=True)
class Arm:
    """One unit the axis is measured on (a philosopher; later, a trait)."""

    key: str        # directory/file stem, e.g. "marcus_aurelius"
    label: str      # short name used in adapter dirs and result keys
    display: str    # human-readable, used in generation prompts
    layer: int      # CAA extraction/injection layer
    coeff: float    # CAA coefficient


@dataclass(frozen=True)
class DilemmaFields:
    """Which keys the forced-choice items actually use.

    The frozen v2 set says `stoic`/`nonstoic`/`stoic_stance`; a sycophancy set
    will say something else. Reading through this map is what lets the same
    ruler score both without rewriting the instrument or touching the frozen
    file.
    """

    id: str
    situation: str
    target: str     # the option the axis steers toward
    foil: str       # the competing option
    stance: str     # the sub-bucket the analysis splits on


@dataclass(frozen=True)
class PairFields:
    """Which keys the contrast pairs use.

    Same reasoning as `DilemmaFields`: the frozen stoic pairs say
    `stoic_text`/`neutral_text`, a trait axis will say something else, and
    `extract_vector` should not care which.
    """

    target: str
    foil: str


@dataclass(frozen=True)
class Axis:
    name: str
    display: str
    directory: Path
    arms: dict[str, Arm]
    fields: DilemmaFields
    pair_fields: PairFields
    pairs_collection_key: str
    dilemmas_file: Path
    dilemmas_collection_key: str
    stances: tuple[str, ...]
    label_tokens: tuple[str, str]
    target_name: str
    judge: dict
    calibration: dict
    criteria: dict
    reference_targets: dict
    pass_b: dict
    results_arm_key: str
    _paths: dict = field(repr=False, default_factory=dict)

    # -- per-arm artifact paths --
    def pairs_file(self, arm: str | Arm) -> Path:
        return self._arm_path("pairs", arm)

    def vector_file(self, arm: str | Arm) -> Path:
        return self._arm_path("vectors", arm)

    def adapter_dir(self, arm: str | Arm) -> Path:
        return self._arm_path("adapters", arm)

    def _arm_path(self, kind: str, arm: str | Arm) -> Path:
        a = self.arms[arm] if isinstance(arm, str) else arm
        root, template = self._paths[kind]
        return _root_dir(root, self.directory) / template.format(key=a.key, label=a.label)

    # -- verbatim text files --
    @property
    def prompts(self) -> list[str]:
        return (self.directory / self._prompts_file).read_text(encoding="utf-8").splitlines()

    @property
    def judge_rubric(self) -> str:
        """Read verbatim — no strip, no normalization. The rubric is an input
        to a judged comparison, so a stray whitespace change is a real change."""
        return (self.directory / self.judge["rubric_file"]).read_text(encoding="utf-8")

    @property
    def _prompts_file(self) -> str:
        return self._paths["__prompts__"]

    @property
    def candidates_file(self) -> Path | None:
        """Resolved dilemmas_v3 candidate set for this axis, if it declares one."""
        spec = self.calibration.get("candidates_file")
        if not spec:
            return None
        return _root_dir(spec["root"], self.directory) / spec["path"]

    def is_read_only(self, path: Path) -> bool:
        return Path(path).is_relative_to(REFERENCE_DIR)


def _root_dir(root: str, axis_dir: Path) -> Path:
    return {
        "reference": REFERENCE_DIR,
        "generated": GENERATED_DIR,
        "models": MODELS_DIR,
        "axis": axis_dir,
    }[root]


def _resolve(spec: dict, axis_dir: Path, where: str, src: Path) -> Path:
    """Resolve a {"root": ..., "path": ...} entry, refusing to escape the repo."""
    if not isinstance(spec, dict) or "root" not in spec:
        raise AxisError(f"{src}: {where} must be an object with a 'root' key, got {spec!r}")
    root = spec["root"]
    if root not in ("reference", "generated", "models", "axis"):
        raise AxisError(f"{src}: {where} has unknown root {root!r} "
                        "(expected reference, generated, models, or axis)")
    rel = spec.get("path") or spec.get("template")
    if not rel:
        raise AxisError(f"{src}: {where} needs a 'path' or 'template'")
    resolved = (_root_dir(root, axis_dir) / rel).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise AxisError(f"{src}: {where} escapes the repo ({resolved})")
    return resolved


def _require(cfg: dict, key: str, src: Path):
    if key not in cfg:
        raise AxisError(f"{src}: missing required key {key!r}")
    return cfg[key]


@lru_cache(maxsize=None)
def load_axis(name: str = DEFAULT_AXIS) -> Axis:
    """Load and validate `axes/<name>/axis.json`.

    Validation is loud on purpose: a silently wrong axis produces numbers that
    look plausible and belong to nothing.
    """
    axis_dir = AXES_DIR / name
    src = axis_dir / "axis.json"
    if not src.exists():
        available = sorted(p.name for p in AXES_DIR.iterdir() if (p / "axis.json").exists()) \
            if AXES_DIR.exists() else []
        raise AxisError(f"no axis {name!r}: {src} not found. Available: {available}")
    cfg = json.loads(src.read_text(encoding="utf-8"))

    version = cfg.get("schema_version")
    if version not in SCHEMA_VERSIONS:
        raise AxisError(f"{src}: schema_version {version!r} not in {SCHEMA_VERSIONS}")
    if cfg.get("name") != name:
        raise AxisError(f"{src}: declares name {cfg.get('name')!r} but lives in axes/{name}/")

    # -- arms --
    arms_cfg = _require(cfg, "arms", src)
    if not arms_cfg:
        raise AxisError(f"{src}: 'arms' is empty — an axis needs at least one arm")
    arms = {}
    for arm_name, a in arms_cfg.items():
        for k in ("key", "label", "layer", "coeff"):
            if k not in a:
                raise AxisError(f"{src}: arm {arm_name!r} missing {k!r}")
        if not isinstance(a["layer"], int) or a["layer"] < 0:
            raise AxisError(f"{src}: arm {arm_name!r} layer must be a non-negative int, "
                            f"got {a['layer']!r}")
        arms[arm_name] = Arm(key=a["key"], label=a["label"],
                             display=a.get("display", a["label"]),
                             layer=a["layer"], coeff=float(a["coeff"]))

    # -- paths (templates kept unresolved; arms fill {key}/{label} later) --
    paths_cfg = _require(cfg, "paths", src)
    paths: dict = {}
    for kind in ("pairs", "vectors", "adapters"):
        spec = _require(paths_cfg, kind, src)
        _resolve({**spec, "path": spec.get("template", "")}, axis_dir, f"paths.{kind}", src)
        paths[kind] = (spec["root"], spec["template"])

    # -- decision instrument --
    di = _require(cfg, "decision_instrument", src)
    dilemmas_file = _resolve(_require(di, "file", src), axis_dir, "decision_instrument.file", src)
    fmap = _require(di, "fields", src)
    for k in ("id", "situation", "target", "foil", "stance"):
        if k not in fmap:
            raise AxisError(f"{src}: decision_instrument.fields missing {k!r}")
    fields = DilemmaFields(**{k: fmap[k] for k in
                              ("id", "situation", "target", "foil", "stance")})
    labels = di.get("label_tokens", [" A", " B"])
    if len(labels) != 2:
        raise AxisError(f"{src}: label_tokens must be exactly two, got {labels!r}")

    # The field map is only meaningful if the items actually carry those keys.
    if dilemmas_file.exists():
        payload = json.loads(dilemmas_file.read_text(encoding="utf-8"))
        items = payload[di.get("collection_key", "dilemmas")] if isinstance(payload, dict) else payload
        if items:
            missing = [v for v in (fields.id, fields.situation, fields.target,
                                   fields.foil, fields.stance) if v not in items[0]]
            if missing:
                raise AxisError(
                    f"{src}: decision_instrument.fields names {missing} but the first item "
                    f"of {dilemmas_file.name} has keys {sorted(items[0])}"
                )

    # -- contrast pairs: field map + optional shape check --
    cp = cfg.get("contrast_pairs", {})
    pf_map = cp.get("fields", {})
    for k in ("target", "foil"):
        if k not in pf_map:
            raise AxisError(f"{src}: contrast_pairs.fields missing {k!r}")
    pair_fields = PairFields(target=pf_map["target"], foil=pf_map["foil"])
    pairs_key = cp.get("collection_key", "pairs")
    sample_pairs = paths["pairs"]
    if arms:
        first = next(iter(arms.values()))
        pp = _root_dir(sample_pairs[0], axis_dir) / sample_pairs[1].format(
            key=first.key, label=first.label)
        if pp.exists():
            payload = json.loads(pp.read_text(encoding="utf-8"))
            entries = payload[pairs_key] if isinstance(payload, dict) else payload
            if entries and (missing := [v for v in (pair_fields.target, pair_fields.foil)
                                        if v not in entries[0]]):
                raise AxisError(
                    f"{src}: contrast_pairs.fields names {missing} but the first pair "
                    f"of {pp.name} has keys {sorted(entries[0])}"
                )

    # -- verbatim text files must exist and be non-empty --
    judge_cfg = _require(cfg, "judge", src)
    prompts_file = _require(cfg, "generation_prompts_file", src)
    for label, fname in (("judge.rubric_file", _require(judge_cfg, "rubric_file", src)),
                         ("generation_prompts_file", prompts_file)):
        p = axis_dir / fname
        if not p.exists():
            raise AxisError(f"{src}: {label} -> {p} does not exist")
        if not p.read_text(encoding="utf-8").strip():
            raise AxisError(f"{src}: {label} -> {p} is empty")
    paths["__prompts__"] = prompts_file

    for k in ("dimensions", "content_dimensions"):
        if not judge_cfg.get(k):
            raise AxisError(f"{src}: judge.{k} is required and must be non-empty")
    unknown = set(judge_cfg["content_dimensions"]) - set(judge_cfg["dimensions"])
    if unknown:
        raise AxisError(f"{src}: judge.content_dimensions {sorted(unknown)} not in judge.dimensions")

    # -- Pass B: the deliberate stoic-only limit --
    pass_b = cfg.get("pass_b", {})
    if name != DEFAULT_AXIS:
        for section, stoic_kinds in STOIC_ONLY_PASS_B.items():
            kind = pass_b.get(section, {}).get("kind")
            if kind in stoic_kinds:
                raise AxisError(
                    f"{src}: pass_b.{section}.kind == {kind!r} is stoic-only. "
                    "stoic/corpus.py and stoic/pairs.py are hardcoded to the stoic "
                    "axis (deliberate Task-2 scope cut; see pass_b._limit). Use kind "
                    "'prepared' and supply paths.pairs directly."
                )

    # -- criteria named arms must exist --
    criteria = cfg.get("criteria", {})
    for k in ("injection_site_arm", "stage4_criterion_arm"):
        arm_name = criteria.get(k)
        if arm_name is not None and arm_name not in arms:
            raise AxisError(f"{src}: criteria.{k} = {arm_name!r} is not an arm ({sorted(arms)})")

    return Axis(
        name=name,
        display=cfg.get("display", name),
        directory=axis_dir,
        arms=arms,
        fields=fields,
        pair_fields=pair_fields,
        pairs_collection_key=pairs_key,
        dilemmas_file=dilemmas_file,
        dilemmas_collection_key=di.get("collection_key", "dilemmas"),
        stances=tuple(di.get("stances", ())),
        label_tokens=(labels[0], labels[1]),
        target_name=di.get("target_name", name),
        judge=judge_cfg,
        calibration=cfg.get("calibration", {}),
        criteria=criteria,
        reference_targets=cfg.get("reference_targets", {}),
        pass_b=pass_b,
        results_arm_key=cfg.get("results", {}).get("arm_key", "per_arm"),
        _paths=paths,
    )


def active_axis_name() -> str:
    return os.environ.get(AXIS_ENV_VAR) or DEFAULT_AXIS


# Bound once, at import. The CLI sets STOIC_AXIS before importing this module.
ACTIVE = load_axis(active_axis_name())
