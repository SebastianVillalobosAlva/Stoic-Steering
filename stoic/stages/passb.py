"""Pass B — regenerate the pipeline's inputs from raw text.

Corpus acquisition ($0, network, no model) and contrastive-pair generation
(Claude API, $). Both write ONLY under data/generated/; the frozen reference
set is compared against, never touched.
"""

from __future__ import annotations

from pathlib import Path

from stoic import config
from stoic.secrets import anthropic_key


def corpus_stage() -> dict:
    """Download -> slice -> chunk into data/generated/, then compare chunk
    counts to the frozen reference corpus (read-only). Reports; never forces."""
    import json as _json

    from stoic import corpus

    print("\n=== Corpus acquisition (Pass B) ===")
    produced = corpus.build()

    print("\nChunk-count comparison vs frozen reference (data/reference/chunked/):")
    ok = True
    for key, path in produced.items():
        author = _json.load(open(path))["author"]
        new_n = _json.load(open(path))["total_chunks"]
        ref_path = config.REF_CHUNKED_DIR / author / Path(path).name
        if ref_path.exists():
            ref_n = _json.load(open(ref_path))["total_chunks"]
            match = new_n == ref_n
            ok = ok and match
            print(f"  {author:16s} {Path(path).name:18s} generated {new_n:>4d}  "
                  f"reference {ref_n:>4d}  {'match ✓' if match else 'DRIFT ✗'}")
        else:
            print(f"  {author:16s} {Path(path).name:18s} generated {new_n:>4d}  (no reference)")
    print(f"\nCorpus: {'all counts match reference ✓' if ok else 'counts drifted (reported, not forced)'}")
    print("(reference is frozen and untouched; generated output is under data/generated/)")
    return {"produced": {k: str(v) for k, v in produced.items()}, "counts_match": ok}


def pairs_stage(num_pairs: int) -> None:
    """Generate contrastive pairs from the freshly-acquired chunks (Pass B,
    COSTS $). Requires `stoic corpus` to have run first."""
    from stoic import pairs

    print("\n=== Contrastive pair generation (Pass B, Claude API) ===")
    key = anthropic_key()
    for folder, cfg in pairs.PAIR_AUTHORS.items():
        chunk_files = sorted((config.GEN_CHUNKED_DIR / folder).glob("*.json"))
        if not chunk_files:
            print(f"[{folder}] no generated chunks — run `python -m stoic corpus` first; skipping")
            continue
        print(f"\n[{folder}] {cfg['display']}")
        pairs.create_pairs(
            chunk_files[0], cfg["display"], folder, key,
            num_pairs=num_pairs, min_chars=cfg["min_chars"], max_chars=cfg["max_chars"],
        )
