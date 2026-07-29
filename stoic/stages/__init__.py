"""Stage orchestration — the authoritative stage → module → command map.

The CLI exposes stages by number; the modules are named for what they do. That
mapping used to live only in people's heads, so it lives here:

    command      module        what it establishes                     
    -----------  ------------  --------------------------------------- 
    stage0       verify.py     deterministic decoding                  
    stage1       verify.py     the ruler's base P(target)              
    stage2       verify.py     vector fidelity + the CAA decision null 
    stage3       content.py    judge-scored content effect             
    style        content.py    the Exp 3b style claim, re-tested       
    stage4       adapters.py   LoRA decision shift, judge-free         
    calibrate    calibrate.py  the dilemmas_v3 per-cell gate           
    corpus       passb.py      Pass B corpus acquisition               
    pairs        passb.py      Pass B contrastive-pair generation      

Two files in the tree are called `calibrate`, and the split is deliberate:
this package's `calibrate.py` is the *stage* — loads candidates, scores them on
the base model, applies the gate, writes a checkpoint. `stoic/calibration.py`
is the *logic* — structural validation and the report maths, no model, no I/O
beyond reading the candidate file. The stage imports the logic; never the
reverse.

No stage decides which axis it runs against: each reads `ACTIVE`, and its
thresholds come from that axis's `criteria` block. Every stage writes exactly
one JSON under `results/<stage>/` through `results_io.write_result`.
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
