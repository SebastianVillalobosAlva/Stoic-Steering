"""Build the matched-length non-philosophical control LoRA training set.

Why this adapter exists: without it, "LoRA moves decisions" cannot be
distinguished from "any fine-tuning on formal prose moves decisions". Method and
training objective are already confounded — CAA is contrastive, LoRA is
continued pretraining — so the control is the only thing that separates the axis
from the procedure. See docs/pre-registration-v3.md §4.

What "matched" means here, made explicit:

- **Same count.** 53 examples, exactly as each philosopher adapter had.
- **Same length distribution**, not merely the same mean. Each target length is
  filled by the closest unused candidate chunk, so the achieved distribution
  tracks the target's shape rather than only its average.
- **Same register.** The source should be formal translated prose of comparable
  era, because the comparison is about *content*. Matching the register is what
  makes a null result on this adapter mean "not the Stoicism" rather than "not
  the prose style".
- **Different content.** No philosophy, no ethical instruction.

Deliberately NOT automated: which source text to use is a methodological choice,
so this script takes it as an argument. Candidates worth considering are listed
in CANDIDATE_SOURCES below.

    # inspect the target profile without writing anything
    python scripts/prep_control_lora.py --profile-only

    # build from a local text
    python scripts/prep_control_lora.py --source path/to/text.txt

    # build from a Gutenberg URL (downloads to data/generated/raw/control/)
    python scripts/prep_control_lora.py --url https://www.gutenberg.org/...

Output: data/generated/lora_training/control_{nonphil,neutral}_train.jsonl — the same one
{"text": ...} per line format `stoic.lora.prep_jsonl` produces, so the existing
`stoic.lora.train()` recipe consumes it unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Vetted candidates. Each is public domain, formal translated prose, and carries
# no ethical or philosophical instruction. Pick one deliberately.
CANDIDATE_SOURCES = {
    "frontinus": {
        "title": "Frontinus, De Aquaeductu Urbis Romae (On the Water Supply of Rome)",
        "why": "Roman, same translation register as the philosophers, and entirely "
               "administrative-technical: discharge measurements, pipe calibres, "
               "maintenance law. The tightest register match with zero philosophy.",
        "url": "https://www.gutenberg.org/cache/epub/46345/pg46345.txt",
    },
    "vitruvius": {
        "title": "Vitruvius, De Architectura (The Ten Books on Architecture)",
        "why": "Roman technical prose, long and easily large enough. Caveat: the "
               "opening books moralise about the architect's education, so the "
               "front matter would need slicing off.",
        "url": "https://www.gutenberg.org/cache/epub/20239/pg20239.txt",
    },
    "beeton": {
        "title": "Mrs Beeton, Book of Household Management",
        "why": "Utterly non-philosophical formal Victorian prose. Register is "
               "19th-century English rather than translated antiquity, which is a "
               "looser match but an unambiguous content contrast.",
        "url": "https://www.gutenberg.org/cache/epub/10136/pg10136.txt",
    },
}

N_EXAMPLES = 53
SEED = 613  # same seed the frozen pair sampling used


def target_profile(match_arm: str | None = None) -> list[int]:
    """The length distribution to match, taken from the frozen pairs.

    Default is the pooled distribution across all three arms, sub-sampled to 53
    lengths by quantile — the arms differ (Seneca 36.3k chars, Epictetus 23.3k),
    so a single control matched to the pool sits between them rather than
    favouring one. `--match-arm seneca` matches the strongest decision-mover
    instead, which is the more demanding comparison.
    """
    from stoic import config
    from stoic.steering import load_pairs

    if match_arm:
        arm = config.ARMS[match_arm]
        return sorted(len(p["stoic_text"]) for p in load_pairs(arm.pairs_file))

    pooled = sorted(
        len(p["stoic_text"])
        for a in config.ARMS.values()
        for p in load_pairs(a.pairs_file)
    )
    # Quantile sub-sample to N_EXAMPLES, preserving the shape of the pool.
    step = (len(pooled) - 1) / (N_EXAMPLES - 1)
    return [pooled[round(i * step)] for i in range(N_EXAMPLES)]


def neutral_examples(n: int = N_EXAMPLES) -> tuple[list[str], dict]:
    """The anti-Stoic halves of the frozen contrast pairs.

    Matched to the philosopher adapters by construction on topic and generation
    process — each neutral_text argues the SAME situation from a worldview that
    rejects Stoicism. That makes it a tighter control than any external corpus
    on everything except the one axis it is meant to isolate.

    It CANNOT be length-matched: the neutral texts run 2.2-3.1x longer than the
    stoic ones (pooled mean 1436 vs 557 chars) with barely overlapping ranges,
    because the generator wrote fuller counterarguments than its sources.
    Truncating to length would cut the conclusion, which is where the
    Stoic-rejecting recommendation sits, so it would remove the very contrast
    the control exists to provide.

    Resolution: keep the example count and the recipe, accept ~2.6x the training
    tokens, and document it. A control with MORE training signal that still
    fails to move decisions strengthens the content conclusion; one that moves
    is ambiguous between content and token count, and is then followed by a
    token-matched variant. Selection is stratified by author and deterministic.
    """
    import statistics as st

    from stoic import config
    from stoic.steering import load_pairs

    per_author = n // len(config.ARMS)
    extra = n - per_author * len(config.ARMS)
    chosen, meta = [], {}
    for i, (name, arm) in enumerate(sorted(config.ARMS.items())):
        take = per_author + (1 if i < extra else 0)
        texts = sorted((p["neutral_text"] for p in load_pairs(arm.pairs_file)), key=len)
        step = (len(texts) - 1) / (take - 1) if take > 1 else 0
        picked = [texts[round(k * step)] for k in range(take)]
        chosen += picked
        meta[name] = {"n": take, "mean_chars": round(st.mean([len(t) for t in picked]))}
    return chosen, meta


def load_source(source: Path | None, url: str | None) -> str:
    if url:
        out = ROOT / "data" / "generated" / "raw" / "control"
        out.mkdir(parents=True, exist_ok=True)
        dest = out / (url.rstrip("/").split("/")[-1] or "control.txt")
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest.relative_to(ROOT)}")
        return dest.read_text(encoding="utf-8", errors="replace")
    return Path(source).read_text(encoding="utf-8", errors="replace")


def candidate_chunks(text: str, lo: int, hi: int) -> list[str]:
    """Paragraph-chunk the source and keep those inside the target length band.

    Reuses `stoic.corpus.chunk_paragraphs` — the same paragraph unit the
    philosopher corpora used, so the chunking is not a second uncontrolled
    difference between the arms and the control.
    """
    from stoic.corpus import _GUT_END, _GUT_START, chunk_paragraphs

    s, e = _GUT_START.search(text), _GUT_END.search(text)
    if s and e:
        text = text[s.end():e.start()]
        print("  stripped Gutenberg licence wrapper")
    chunks = chunk_paragraphs(text)
    kept = [c for c in chunks if lo <= len(c) <= hi]
    print(f"  {len(chunks)} paragraphs, {len(kept)} inside [{lo}, {hi}] chars")
    return kept


def match_lengths(candidates: list[str], targets: list[int]) -> list[str]:
    """Fill each target length with the closest unused candidate.

    Closest-fit rather than random sampling: random sampling matches the mean
    only in expectation, and with n=53 the realised distribution can drift far
    enough that 'matched-length' stops being true.
    """
    pool = sorted(candidates, key=len)
    chosen, used = [], set()
    for t in targets:
        best, best_gap = None, None
        for i, c in enumerate(pool):
            if i in used:
                continue
            gap = abs(len(c) - t)
            if best_gap is None or gap < best_gap:
                best, best_gap = i, gap
        if best is None:
            raise SystemExit(f"ran out of candidate chunks at target length {t}")
        used.add(best)
        chosen.append(pool[best])
    return chosen


def report(achieved: list[str], targets: list[int]) -> dict:
    a = [len(c) for c in achieved]
    rows = [("n", len(a), len(targets)),
            ("mean chars", round(st.mean(a)), round(st.mean(targets))),
            ("median chars", round(st.median(a)), round(st.median(targets))),
            ("min chars", min(a), min(targets)),
            ("max chars", max(a), max(targets)),
            ("total chars", sum(a), sum(targets))]
    print(f"\n{'':14s} {'achieved':>10s} {'target':>10s}   drift")
    for label, got, want in rows:
        drift = "" if got == want else f"{(got - want) / want:+.1%}" if want else ""
        print(f"{label:14s} {got:>10} {want:>10}   {drift}")
    return {k: {"achieved": g, "target": w} for k, g, w in rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=["nonphilosophical", "neutral"],
                   default="nonphilosophical",
                   help="nonphilosophical: external corpus, isolates content-vs-fine-tuning. "
                        "neutral: the frozen pairs' anti-Stoic halves, isolates "
                        "Stoic-content-vs-any-normative-content.")
    p.add_argument("--source", help="local text file")
    p.add_argument("--url", help="Gutenberg URL to download")
    p.add_argument("--match-arm", choices=["marcus", "seneca", "epictetus"],
                   help="match one arm's length distribution instead of the pool")
    p.add_argument("--profile-only", action="store_true",
                   help="print the target profile and the candidate sources, write nothing")
    args = p.parse_args()

    if args.kind == "neutral":
        chosen, per_author = neutral_examples()
        import statistics as st
        L = [len(c) for c in chosen]
        tgt = target_profile(args.match_arm)
        print(f"neutral control: {len(chosen)} examples, stratified by author {per_author}")
        print(f"  mean {st.mean(L):.0f} chars vs philosopher target {st.mean(tgt):.0f} "
              f"({st.mean(L)/st.mean(tgt):.2f}x)  total {sum(L)} vs {sum(tgt)}")
        print("  NOT length-matched, by design — see neutral_examples() for why")
        out_dir = ROOT / "data" / "generated" / "lora_training"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "control_neutral_train.jsonl"
        with open(out, "w") as f:
            for c in chosen:
                f.write(json.dumps({"text": c}) + "\n")
        (out_dir / "control_neutral_train.meta.json").write_text(json.dumps({
            "kind": "neutral (anti-Stoic halves of the frozen contrast pairs)",
            "isolates": "Stoic content vs any strong normative content",
            "source": "data/reference/processed/{arm}/neutral_pairs.json (frozen, read-only)",
            "n_examples": len(chosen), "per_author": per_author,
            "mean_chars": round(st.mean(L)), "total_chars": sum(L),
            "philosopher_total_chars": sum(tgt),
            "token_ratio_vs_philosophers": round(sum(L) / sum(tgt), 2),
            "length_matched": False,
            "why_not": "neutral texts run 2.2-3.1x longer with barely overlapping ranges; "
                       "truncating to length would cut the Stoic-rejecting recommendation, "
                       "which is the contrast being tested. Example count and recipe are "
                       "held constant instead; the token excess is documented and biases "
                       "toward the control moving, so a null is the strong result.",
            "selection": "stratified by author, quantile-spaced by length, deterministic",
        }, indent=2) + "\n")
        print(f"\n\u21b3 wrote {out.relative_to(ROOT)} ({len(chosen)} examples)")
        print("\nNext: train on Colab, canonical recipe unchanged —")
        print("  train('control_neutral_train.jsonl', 'models/lora_control_neutral', epochs=3)")
        return 0

    targets = target_profile(args.match_arm)
    which = args.match_arm or "pooled across all three arms"
    print(f"target profile ({which}): n={len(targets)}, mean={st.mean(targets):.0f}, "
          f"median={st.median(targets):.0f}, range {min(targets)}-{max(targets)}, "
          f"total={sum(targets)}")

    if args.profile_only or not (args.source or args.url):
        print("\ncandidate sources (pick deliberately — this is a methodological choice):")
        for key, c in CANDIDATE_SOURCES.items():
            print(f"\n  {key}: {c['title']}")
            print(f"    {c['why']}")
            print(f"    {c['url']}")
        if not args.profile_only:
            print("\nNothing written. Re-run with --source or --url.")
        return 0

    text = load_source(args.source, args.url)
    cands = candidate_chunks(text, min(targets), max(targets))
    if len(cands) < len(targets):
        raise SystemExit(f"only {len(cands)} usable chunks, need {len(targets)}")
    chosen = match_lengths(cands, targets)
    stats = report(chosen, targets)

    out_dir = ROOT / "data" / "generated" / "lora_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "control_nonphil_train.jsonl"
    with open(out, "w") as f:
        for c in chosen:
            f.write(json.dumps({"text": c}) + "\n")
    (out_dir / "control_nonphil_train.meta.json").write_text(json.dumps({
        "built_on_utc_note": "stamp this from the shell; scripts avoid wall-clock",
        "source": args.url or str(args.source),
        "matched_to": which,
        "n_examples": len(chosen),
        "length_profile": stats,
        "chunking": "stoic.corpus.chunk_paragraphs (same unit as the philosopher corpora)",
        "selection": "closest-unused-fit per target length (deterministic, no sampling)",
        "purpose": "matched-length non-philosophical control for the v3 behavioral eval; "
                   "see docs/pre-registration-v3.md section 4",
    }, indent=2) + "\n")
    print(f"\n↳ wrote {out.relative_to(ROOT)} ({len(chosen)} examples)")
    print(f"↳ wrote {(out_dir / 'control_nonphil_train.meta.json').relative_to(ROOT)}")
    print("\nNext: train on Colab with the canonical recipe —")
    print("  from stoic.lora import train")
    print("  train('control_nonphil_train.jsonl', 'models/lora_control_nonphil', epochs=3)")
    print("  (r=8, alpha=32, q_proj+v_proj — stoic.lora.LORA_CONFIG, unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
