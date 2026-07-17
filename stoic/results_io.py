"""Checkpoint JSON writing — one place, always under results/<stage>/."""

from __future__ import annotations

import json
import time
from pathlib import Path

from stoic import config


def write_result(stage: str, name: str, payload: dict) -> Path:
    payload = {"timestamp": time.strftime("%Y%m%d_%H%M%S"), **payload}
    path = config.results_dir(stage) / f"{name}_{payload['timestamp']}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ↳ wrote {path.relative_to(config.PROJECT_ROOT)}")
    return path
