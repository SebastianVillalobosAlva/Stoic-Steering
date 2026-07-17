"""Frozen-fixture integrity: the counts CLAUDE.md freezes, and the binary manifest."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from stoic import config
from stoic.steering import load_pairs

ROOT = Path(__file__).resolve().parent.parent


def test_neutral_pairs_53_each_and_loadable():
    for author in config.AUTHORS.values():
        pairs = load_pairs(author.pairs_file)
        assert len(pairs) == 53, author.key
        for p in pairs:
            assert p["stoic_text"].strip() and p["neutral_text"].strip()


def test_dilemmas_v2_shape():
    with open(config.DILEMMAS_V2) as f:
        dilemmas = json.load(f)["dilemmas"]
    assert len(dilemmas) == 40
    ids = [d["id"] for d in dilemmas]
    assert len(set(ids)) == 40
    for d in dilemmas:
        for field in ("situation", "stoic", "nonstoic", "stoic_stance"):
            assert d[field], d["id"]
    stances = Counter(d["stoic_stance"] for d in dilemmas)
    # the stance buckets the Stage 4 analysis reports (accepting n=22, active n=18)
    assert stances == {"accepting": 22, "active": 18}


def test_binary_manifest_matches_disk():
    """data/MANIFEST.sha256 pins the untracked frozen binaries (steering
    vectors, adapters). Where the files exist locally, they must match."""
    manifest = ROOT / "data" / "MANIFEST.sha256"
    assert manifest.exists(), "checksum manifest missing"
    checked = 0
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        path = ROOT / rel
        if not path.exists():
            continue  # fresh clone without the binaries: nothing to verify yet
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f"checksum mismatch: {rel}"
        checked += 1
    if checked == 0:
        pytest.skip("no frozen binaries present locally (fresh clone)")
