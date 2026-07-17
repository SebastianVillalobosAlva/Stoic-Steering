"""API-key lookup: environment first, then KEY=VALUE lines in project-root .env.

Keys never enter the package as constants; stages that cost $ fail fast with a
message naming exactly which key is missing.
"""

from __future__ import annotations

import os

from stoic import config


def _lookup(names: tuple[str, ...]) -> str | None:
    for k in names:
        if os.environ.get(k):
            return os.environ[k]
    env = config.PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            for k in names:
                if line.startswith(k) and "=" in line:
                    return line.split("=", 1)[1].strip().strip("'\"")
    return None


def gemini_key() -> str:
    key = _lookup(("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    if key is None:
        raise SystemExit(
            "No Gemini key found. Set GEMINI_API_KEY in the environment or in "
            f"{config.PROJECT_ROOT / '.env'} (GEMINI_API_KEY=...). "
            "Stage 3 / style call the Gemini judge API ($)."
        )
    return key


def anthropic_key() -> str:
    key = _lookup(("ANTHROPIC_API_KEY",))
    if key is None:
        raise SystemExit(
            "No ANTHROPIC_API_KEY (env or .env). Pair generation calls the Claude API ($)."
        )
    return key
