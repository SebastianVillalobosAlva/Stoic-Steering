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
REF_CHUNKED_DIR = REFERENCE_DIR / "chunked"          # {author}/{work}.json (frozen chunks)
DILEMMAS_V2 = REF_CONFIG_DIR / "dilemmas_v2.json"
# Corpus-acquisition source manifest (Gutenberg URLs + slicing boundaries).
# Read-only input; provenance is also mirrored in docs/corpus-sources.md.
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

# Exp 3b stylistic-authenticity deltas (the "robust" style claim under test).
# CAVEAT: Exp 3b ran the superseded all-L8 configs (coeffs 0.08/0.15/0.11) with
# pre-Exp-8 vectors AND the asymmetric decoding. The style re-test runs the
# canonical clean configs at matched decoding; Exp 3b is the historical
# reference, not an exact-config comparison.
EXP3B_STYLE: dict[str, float] = {
    "marcus": 1.00,
    "seneca": 1.42,
    "epictetus": 1.58,
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


# --- Contrastive-pair generation prompt (Pass B) -------------------------
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
