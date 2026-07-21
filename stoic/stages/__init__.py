"""Pass A / Pass B stage orchestration, one module per concern.

verify.py    Stages 0-2: determinism, the 0.542 baseline, vector fidelity + CAA null
content.py   Stage 3 + style validation: Gemini-judged effects under matched decoding
adapters.py  Stage 4: LoRA decision shift (judge-free, frozen adapters)
passb.py     Pass B: corpus acquisition + contrastive-pair generation
calibrate.py dilemmas_v3 calibration gate (base-model per-cell P(stoic) ~= 0.5)
"""

from stoic.stages.adapters import stage4
from stoic.stages.calibrate import calibrate_stage
from stoic.stages.content import stage3, style_check
from stoic.stages.passb import corpus_stage, pairs_stage
from stoic.stages.verify import stage0, stage1, stage2

__all__ = [
    "stage0",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "style_check",
    "corpus_stage",
    "pairs_stage",
    "calibrate_stage",
]
