"""Fetch the frozen binaries (steering vectors, LoRA adapters) into place and
verify them against data/MANIFEST.sha256.

These files are too large to track in git but are load-bearing for Pass A
Stages 2 and 4: Stage 2 checks freshly-extracted vectors against the frozen
`.pt` files (cosine >= 0.99), and Stage 4 merges the frozen adapters onto a
fresh base. A checkout without them can run Stages 0-1 and the unit tests, but
not 2 or 4.

Usage:
    python scripts/fetch_artifacts.py            # download missing, verify all
    python scripts/fetch_artifacts.py --check    # verify only, download nothing
    python scripts/fetch_artifacts.py --force    # re-download even if present
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "MANIFEST.sha256"
HF_REPO = "seb-vil/llama-3.2-3b-stoic-steering"

# Manifest paths are repo-relative; map each to its path within the HF repo.
# steering vectors: data/reference/steering_vectors/X.pt  -> steering_vectors/X.pt
# adapters:         models/lora_A_clean/F                 -> lora_A_clean/F
def _hf_path(repo_rel: str) -> str:
    p = Path(repo_rel)
    if repo_rel.startswith("data/reference/steering_vectors/"):
        return f"steering_vectors/{p.name}"
    if repo_rel.startswith("models/"):
        return f"{p.parent.name}/{p.name}"
    raise ValueError(f"Don't know how to map {repo_rel!r} to an HF path")


def _parse_manifest() -> list[tuple[str, str]]:
    entries = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        entries.append((digest, rel))
    return entries


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    entries = _parse_manifest()
    # adapter_config.json files are tiny and tracked in git; only fetch what's missing.
    to_fetch = [(d, r) for d, r in entries if not (ROOT / r).exists() or args.force]

    if to_fetch and not args.check:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("huggingface_hub not installed. `pip install huggingface-hub` "
                  "(already a core dep of this project).", file=sys.stderr)
            return 2
        for _digest, rel in to_fetch:
            dest = ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  fetching {rel} ...")
            got = hf_hub_download(repo_id=HF_REPO, filename=_hf_path(rel), repo_type="model")
            dest.write_bytes(Path(got).read_bytes())
    elif to_fetch and args.check:
        for _d, rel in to_fetch:
            print(f"  MISSING {rel}")

    print("\nVerifying against data/MANIFEST.sha256:")
    ok = True
    for digest, rel in entries:
        path = ROOT / rel
        if not path.exists():
            print(f"  ✗ MISSING   {rel}")
            ok = False
            continue
        actual = _sha256(path)
        match = actual == digest
        ok = ok and match
        print(f"  {'✓' if match else '✗ MISMATCH'} {rel}")

    print(f"\n{'All artifacts present and verified ✓' if ok else 'Artifacts missing or corrupt ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
