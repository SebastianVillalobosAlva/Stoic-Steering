"""Paths and canonical configuration — the single source of truth.

Two kinds of setting live here, and the difference matters:

- **Setup constants** (model, dtype, decoding) are properties of the *experiment*
  and are identical on every behavioral axis. They are defined here.
- **Instrument constants** (which arms, which layers, which dilemma set, which
  judge rubric, which thresholds) are properties of the *axis under test*. They
  are read from `stoic.axis.ACTIVE` and merely re-exported here, so that older
  call sites and `scripts/exp12_*.py` keep the surface they were written
  against.

The reference wall is enforced by construction: `REFERENCE_DIR` is only ever
read from, `GENERATED_DIR` is the only place the pipeline writes data artifacts.
Path roots are defined in `stoic/axis.py` (which must not import this module)
and re-exported below, so there is exactly one definition of each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from stoic.axis import ACTIVE, Arm, Axis
from stoic.axis import (  # roots: defined in axis.py, re-exported here
    DATA_DIR,
    GENERATED_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    REFERENCE_DIR,
)

RESULTS_DIR = PROJECT_ROOT / "results"

# --- Reference sub-paths (Pass A inputs) ---------------------------------
REF_PROCESSED_DIR = REFERENCE_DIR / "processed"      # {author}/neutral_pairs.json
REF_CONFIG_DIR = REFERENCE_DIR / "config"            # dilemmas_v2.json, sources.json
REF_VECTORS_DIR = REFERENCE_DIR / "steering_vectors"  # {author}_steering_3B.pt
REF_CHUNKED_DIR = REFERENCE_DIR / "chunked"          # {author}/{work}.json (frozen chunks)
# The active axis's forced-choice ruler. On the stoic axis this is dilemmas_v2.json.
DILEMMAS_V2 = ACTIVE.dilemmas_file
# Corpus-acquisition source manifest (Gutenberg URLs + slicing boundaries).
# Read-only input; provenance is also mirrored in docs/corpus-sources.md.
# Stoic-only: corpus acquisition is not axis-generalized (see axis.json pass_b).
SOURCES_JSON = REF_CONFIG_DIR / "sources.json"

# --- Generated sub-paths (corpus/pairs pipeline output) ------------------
GEN_RAW_DIR = GENERATED_DIR / "raw"             # {author}/{work}.txt  (downloaded)
GEN_PROCESSED_DIR = GENERATED_DIR / "processed"  # {author}/{work}.txt  (sliced clean)
GEN_CHUNKED_DIR = GENERATED_DIR / "chunked"      # {author}/{work}.json (paragraph chunks)

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
    """An axis arm bound to its axis, exposing the artifact paths.

    `Arm` deliberately knows nothing about paths — the `Axis` resolves those,
    since where an arm's pairs/vector/adapter live is a property of the axis
    config, not of the arm. This binds the two back together so the long-standing
    `author.vector_file` / `author.adapter_dir` surface keeps working, which is
    what `scripts/exp12_*.py` is written against.
    """

    arm: Arm
    axis: Axis = field(default=ACTIVE, repr=False)

    @property
    def key(self) -> str:
        return self.arm.key

    @property
    def label(self) -> str:
        return self.arm.label

    @property
    def display(self) -> str:
        return self.arm.display

    @property
    def layer(self) -> int:
        return self.arm.layer

    @property
    def coeff(self) -> float:
        return self.arm.coeff

    @property
    def pairs_file(self) -> Path:
        return self.axis.pairs_file(self.arm)

    @property
    def vector_file(self) -> Path:
        return self.axis.vector_file(self.arm)

    @property
    def adapter_dir(self) -> Path:
        return self.axis.adapter_dir(self.arm)


# The arms of the active axis. On the stoic axis this is the CAA clean best
# configuration (ground truth): Marcus L26, Seneca L4, Epictetus L8, coeff 0.11.
AUTHORS: dict[str, Author] = {
    name: Author(arm, ACTIVE) for name, arm in ACTIVE.arms.items()
}

# The forced-choice ruler's known baseline (v2 set, both label orders averaged).
DILEMMA_BASELINE = ACTIVE.criteria.get("decision_baseline")  # stoic: 0.542

# Exp 9 content effect targets (clean pairs, Gemini judge, coeff 0.11, n_seeds=5).
# {author: (content_mean, content_std)}. Judge is nondeterministic → reproduce
# the pattern (all positive, error bars overlapping), not the decimals.
EXP9_CONTENT: dict[str, tuple[float, float]] = {
    k: tuple(v) for k, v in ACTIVE.reference_targets.get("exp9_content", {}).items()
}

# Exp 3b stylistic-authenticity deltas (the "robust" style claim under test).
# CAVEAT: Exp 3b ran the superseded all-L8 configs (coeffs 0.08/0.15/0.11) with
# pre-Exp-8 vectors AND the asymmetric decoding. The style re-test runs the
# canonical clean configs at matched decoding; Exp 3b is the historical
# reference, not an exact-config comparison.
EXP3B_STYLE: dict[str, float] = dict(ACTIVE.reference_targets.get("exp3b_style", {}))


# --- Prompts for qualitative steering / seed eval (Stage 3+, unused in Pass A 0-2) ---
# Read once at import from the axis's prompts file, in file order.
DEFAULT_PROMPTS: list[str] = list(ACTIVE.prompts)


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


# --- Contrastive-pair generation prompt (Pass B) -------------------------
# STOIC-ONLY. Pass-B generation (corpus.py, pairs.py) is not axis-generalized;
# a non-stoic axis supplies prepared pairs instead (see axis.json pass_b).
# The exact prompt that produced the frozen neutral_pairs.json sets. Given a
# Stoic passage, Claude argues the SAME situation from a competing worldview so
# the pair isolates reasoning, not topic. Kept verbatim for provenance.
NEUTRAL_PAIR_PROMPT = """Below is a philosophical passage from {author_name}. Your job is to give advice about the SAME situation, but reasoning from a worldview that genuinely DISAGREES with Stoicism — not Stoicism in plainer words.

Pick a competing framework and argue from it, e.g.:
- Ambition/achievement: pursue status, wealth, and winning; external success IS what matters
- Hedonism: maximize pleasure and comfort; avoid discomfort rather than accept it
- Assertiveness/self-advocacy: change your circumstances, push back, demand more
- Emotional expression: feel and express anger/desire fully rather than governing them

Hard requirements:
- Reach a recommendation a Stoic would REJECT. The conclusion itself must differ, not just the wording.
- FORBIDDEN (these are Stoic ideas — do not endorse any of them, even casually): accepting what you can't control, focusing on what's "up to you", indifference to externals (reputation, money, body, outcomes), virtue/character as the main good, "this won't matter in the long run", inner tranquility over external change, others' opinions don't matter.
- Do NOT use a calm, detached, or "wise" self-help tone. Write as someone who actively wants the external thing — the promotion, the win, the pleasure, the apology owed to them.

FAILURE CONDITION: If your rewrite could be summarized the same way as the original passage, you have failed. The original and your rewrite must give OPPOSITE life advice, not the same advice in different words.
- Output ONLY the advice itself. No headers, no preamble, no labeling which framework you are using, no meta-commentary. Start directly with the advice and write it as continuous prose.

Passage:
{stoic_text}"""
