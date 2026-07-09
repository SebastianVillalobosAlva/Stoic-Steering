"""Contrastive-pair generation (Pass B): for each Stoic chunk, ask Claude to
argue the SAME situation from a worldview that rejects Stoicism, producing a
(stoic_text, neutral_text) pair that isolates reasoning from topic.

Ported unchanged in substance from the original pipeline: same length +
non-philosophical filter, same seeded sampling (seed=613), same contrastive
prompt (`config.NEUTRAL_PAIR_PROMPT`), same Claude model. Writes only under
`data/generated/`. Costs Anthropic API credits — this is Pass B, not a Pass-A
checkpoint; the frozen reference pairs are the Pass-A inputs.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from stoic import config

PAIR_MODEL = "claude-sonnet-4-20250514"

_BIBLIO_MARKERS = [
    "pp.", "Vol.", "ISBN", "Published", "published", "Editor", "Reprinted",
    "translation", "emendation", "corrupt", "Casaubon",
]
_BIOGRAPHICAL_MARKERS = [
    "was born", "his death", "his life", "his daily life", "we meet with",
    "translator", "preface", "biography", "the author", "his reign", "Faustina",
    "Commodus", "Hadrian", "A.D.", "B.C.", "born in", "died in", "rhetorician",
    "bestseller", "vernacular", "founder of", "Cyprus", "present century",
]


def is_non_philosophical(text: str, author_name: str) -> bool:
    """Flag citations / biographical-editorial prose ABOUT the author (not
    philosophy BY the author). Religious vocabulary is deliberately NOT filtered
    — 'God/nature/Providence' are core Stoic terms, not contamination."""
    has_citation = bool(re.search(r",\s+\d{4}\b", text))  # "New York, 1955"
    biblio = sum(1 for m in _BIBLIO_MARKERS if m in text) + (1 if has_citation else 0)
    biographical = sum(1 for m in _BIOGRAPHICAL_MARKERS if m in text)
    names_self = author_name.split()[-1] in text
    return biblio >= 2 or biographical >= 2 or (names_self and biographical >= 1)


def filter_chunks(chunks: list[dict], author_name: str, min_chars=300, max_chars=1000) -> list[dict]:
    """Keep philosophical chunks within the length band."""
    return [
        c for c in chunks
        if min_chars <= len(c["text"]) <= max_chars
        and not is_non_philosophical(c["text"], author_name)
    ]


def generate_neutral_text(client, stoic_text: str, author_name: str,
                          model: str = PAIR_MODEL, max_tokens: int = 1000) -> str:
    prompt = config.NEUTRAL_PAIR_PROMPT.format(author_name=author_name, stoic_text=stoic_text)
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def create_pairs(chunks_file: str | Path, author_name: str, author_folder: str,
                 api_key: str, *, num_pairs=63, min_chars=300, max_chars=1000,
                 seed=613, delay=0.5) -> Path:
    """Generate up to num_pairs contrastive pairs for one author; write
    data/generated/processed/{author}/neutral_pairs.json."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    with open(chunks_file) as f:
        chunks = json.load(f)["chunks"]

    filtered = filter_chunks(chunks, author_name, min_chars, max_chars)
    print(f"  {len(filtered)} chunks after filtering (length {min_chars}-{max_chars})")
    if len(filtered) > num_pairs:
        random.seed(seed)
        to_process = random.sample(filtered, num_pairs)
    else:
        to_process = filtered
        if len(filtered) < num_pairs:
            print(f"  ⚠ only {len(filtered)} chunks — generating {len(filtered)} (< {num_pairs})")

    pairs = []
    for i, chunk in enumerate(to_process, 1):
        print(f"  [{i}/{len(to_process)}] chunk {chunk['id']}")
        try:
            neutral = generate_neutral_text(client, chunk["text"], author_name)
            pairs.append({"id": chunk["id"], "stoic_text": chunk["text"], "neutral_text": neutral})
        except Exception as e:
            print(f"    ✗ failed: {e}")
        time.sleep(delay)

    out = config.GEN_PROCESSED_DIR / author_folder / "neutral_pairs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)
    print(f"  ✓ {len(pairs)} pairs -> {out.relative_to(config.PROJECT_ROOT)}")
    return out


# Per-author generation config (equal N to control for data QUANTITY, so any
# cross-philosopher difference reflects pair QUALITY). Epictetus uses a lower
# min_chars — the Enchiridion is short and aphoristic.
PAIR_AUTHORS = {
    "marcus_aurelius": {"display": "Marcus Aurelius", "min_chars": 300, "max_chars": 1000},
    "seneca": {"display": "Seneca", "min_chars": 300, "max_chars": 1000},
    "epictetus": {"display": "Epictetus", "min_chars": 150, "max_chars": 1000},
}
N_PAIRS = 63
