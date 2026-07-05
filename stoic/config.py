"""Paths and canonical configuration — the single source of truth.

The reference wall is enforced here by construction: `REFERENCE_DIR` is only ever
read from, `GENERATED_DIR` is the only place the pipeline writes data artifacts.
Nothing in this package should build a path into `reference/` for writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

# --- Roots ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# FROZEN. Read-only. Never write here.
REFERENCE_DIR = DATA_DIR / "reference"
# Everything the new pipeline produces goes here.
GENERATED_DIR = DATA_DIR / "generated"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Reference sub-paths (Pass A inputs) ---------------------------------
REF_PROCESSED_DIR = REFERENCE_DIR / "processed"      # {author}/neutral_pairs.json
REF_CONFIG_DIR = REFERENCE_DIR / "config"            # dilemmas_v2.json, sources.json
REF_VECTORS_DIR = REFERENCE_DIR / "steering_vectors"  # {author}_steering_3B.pt
DILEMMAS_V2 = REF_CONFIG_DIR / "dilemmas_v2.json"

# --- Model (Llama-3.2-3B only) -------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B"
DTYPE = torch.float16
DEVICE = "cpu"
NUM_LAYERS = 28
HIDDEN_DIM = 3072

# --- Canonical decoding (defined ONCE, used everywhere) ------------------
# Greedy + repetition controls. Deterministic: do_sample=False.
GEN_KWARGS = dict(
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.3,
    no_repeat_ngram_size=3,
)


@dataclass(frozen=True)
class Author:
    """A philosopher and the CAA config that steers toward them."""

    key: str          # directory/name stem, e.g. "marcus_aurelius"
    label: str        # short label used in dilemma-result configs, e.g. "marcus"
    layer: int        # clean best CAA layer
    coeff: float = 0.11

    @property
    def pairs_file(self) -> Path:
        return REF_PROCESSED_DIR / self.key / "neutral_pairs.json"

    @property
    def vector_file(self) -> Path:
        return REF_VECTORS_DIR / f"{self.key}_steering_3B.pt"

    @property
    def adapter_dir(self) -> Path:
        return MODELS_DIR / f"lora_{self.label}_clean"


# CAA clean best layers / coeff (ground truth): Marcus L26, Seneca L4, Epictetus L8.
AUTHORS: dict[str, Author] = {
    "marcus": Author("marcus_aurelius", "marcus", layer=26),
    "seneca": Author("seneca", "seneca", layer=4),
    "epictetus": Author("epictetus", "epictetus", layer=8),
}

# The forced-choice ruler's known baseline (v2 set, both label orders averaged).
DILEMMA_BASELINE = 0.542  # exact reference value: 0.541601902275579

# Exp 9 content effect targets (clean pairs, Gemini judge, coeff 0.11, n_seeds=5).
# {author: (content_mean, content_std)}. Judge is nondeterministic → reproduce
# the pattern (all positive, error bars overlapping), not the decimals.
EXP9_CONTENT: dict[str, tuple[float, float]] = {
    "marcus": (0.408, 0.136),
    "seneca": (0.583, 0.121),
    "epictetus": (0.767, 0.076),
}


# --- Prompts for qualitative steering / seed eval (Stage 3+, unused in Pass A 0-2) ---
DEFAULT_PROMPTS: list[str] = [
    "When facing difficulty, one should",
    "The nature of virtue is",
    "To live well means",
    "What is not in our power",
    "We suffer not from events but from",
    "The wise person is one who",
    "True freedom comes from",
    "The obstacle in the path becomes",
    "Suffering teaches us that",
    "We should remember that all things",
    "Each day should be lived as if",
    "Our duty to others requires",
]


@dataclass
class Config:
    """Runtime knobs. Defaults are the canonical values; override sparingly."""

    model_name: str = MODEL_NAME
    dtype: torch.dtype = DTYPE
    device: str = DEVICE
    authors: dict[str, Author] = field(default_factory=lambda: dict(AUTHORS))
    gen_kwargs: dict = field(default_factory=lambda: dict(GEN_KWARGS))


def results_dir(stage: str) -> Path:
    """Return (and create) a results subdir for a stage's checkpoint JSONs."""
    d = RESULTS_DIR / stage
    d.mkdir(parents=True, exist_ok=True)
    return d
