"""Corpus acquisition (Pass B): download raw texts, slice to the work proper,
chunk into paragraphs.

Ported from the original pipeline unchanged in substance — same Gutenberg
sources, same license-stripping, same content_start/content_end slicing, same
paragraph chunking — so the chunks this produces carry the same provenance as
the frozen reference corpus. Source manifest: `data/reference/config/sources.json`
(read-only); provenance is also written out in `docs/corpus-sources.md`.

Reads sources from config, writes only under `data/generated/`. Never touches
`data/reference/`.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from stoic import config

# Gutenberg wrapper markers — the license text before/after the work body.
_GUT_START = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.DOTALL)
_GUT_END = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.DOTALL)


def load_sources(path: str | Path = config.SOURCES_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def download(cfg: dict) -> Path:
    """Fetch one source's raw text into data/generated/raw/{author}/{file}."""
    out = config.GEN_RAW_DIR / cfg["author_folder"] / cfg["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {cfg['url']}")
    urllib.request.urlretrieve(cfg["url"], out)
    print(f"  ✓ raw -> {out.relative_to(config.PROJECT_ROOT)}")
    return out


def _find_boundaries(text: str, content_start: str | None, content_end: str | None):
    """Three-stage slice: (1) strip the Gutenberg license wrapper; then, inside
    the body, (2) skip front-matter up to content_start and (3) cut back-matter
    from content_end. Stages 2/3 apply only if the regex is given AND matches."""
    s, e = _GUT_START.search(text), _GUT_END.search(text)
    if not (s and e):
        return None, None
    inner_start, inner_end = s.end(), e.start()
    body = text[inner_start:inner_end]

    start_offset = 0
    if content_start:
        m = re.search(content_start, body, re.IGNORECASE)
        if m:
            start_offset = m.start()
        else:
            print(f"  ⚠ content_start /{content_start}/ not matched — keeping from body start")

    end_offset = len(body)
    if content_end:
        m = re.search(content_end, body[start_offset:], re.IGNORECASE)
        if m:
            end_offset = start_offset + m.start()
        else:
            print(f"  ⚠ content_end /{content_end}/ not matched — keeping to body end")

    return inner_start + start_offset, inner_start + end_offset


def clean(cfg: dict, raw_path: Path) -> Path:
    """Strip the Gutenberg wrapper and narrow to the work proper."""
    text = Path(raw_path).read_text(encoding="utf-8")
    start, end = _find_boundaries(text, cfg.get("content_start"), cfg.get("content_end"))
    if start is not None and end is not None:
        clean_text = text[start:end]
    else:
        print("  ⚠ Gutenberg license markers not found — keeping full text")
        clean_text = text

    pct = 100 * len(clean_text) / max(len(text), 1)
    print(f"  sliced to {len(clean_text):,} chars ({pct:.0f}% of raw)")
    if pct < 20:
        print("  ⚠⚠ kept <20% of file — a boundary marker may have mismatched")

    out = config.GEN_PROCESSED_DIR / cfg["author_folder"] / cfg["filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(clean_text, encoding="utf-8")
    print(f"  ✓ clean -> {out.relative_to(config.PROJECT_ROOT)}")
    return out


def chunk_paragraphs(text: str) -> list[str]:
    """Paragraph split: drop [123] footnote markers, split on blank lines,
    strip, keep non-empty. This is the chunk unit for pair generation."""
    text = re.sub(r"\[\d+\]", "", text)
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk(cfg: dict, clean_path: Path) -> Path:
    """Chunk a cleaned text into paragraphs; write the {author}/{work}.json
    manifest (same schema as the frozen reference chunks)."""
    author = cfg["author_folder"]
    stem = Path(clean_path).name.replace(".txt", "")
    paras = chunk_paragraphs(Path(clean_path).read_text(encoding="utf-8"))
    payload = {
        "source_file": str(clean_path),
        "author": author,
        "total_chunks": len(paras),
        "chunks": [{"id": i, "text": p} for i, p in enumerate(paras, 1)],
    }
    out = config.GEN_CHUNKED_DIR / author / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {len(paras)} chunks -> {out.relative_to(config.PROJECT_ROOT)}")
    return out


def build(sources_path: str | Path = config.SOURCES_JSON) -> dict[str, Path]:
    """Full corpus acquisition for every source: download -> clean -> chunk.
    Returns {source_key: chunk_json_path}. Writes only under data/generated/."""
    sources = load_sources(sources_path)
    out = {}
    for key, cfg in sources.items():
        print(f"\n[{key}] {cfg['author']}")
        raw = download(cfg)
        clean_path = clean(cfg, raw)
        out[key] = chunk(cfg, clean_path)
    return out
