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
# IDs verified against gutenberg.org/ebooks/search on 2026-07-29. Frontinus's
# De Aquaeductu was the first recommendation and is NOT on Gutenberg in English
# (only Les strategemes, in French, #16237) — it was removed rather than left as
# an unavailable suggestion.
CANDIDATE_SOURCES = {
    "vitruvius": {
        "title": "Vitruvius, De Architectura (The Ten Books on Architecture)",
        "why": "VERIFIED #20239. Roman technical prose, translated in the same "
               "tradition as the philosophers, and long enough. Caveat: Book 1 "
               "moralises about the architect's education and character — use "
               "--skip-until to cut it, and read the content warnings.",
        "url": "https://www.gutenberg.org/cache/epub/20239/pg20239.txt",
    },
    "beeton": {
        "title": "Mrs Beeton, Book of Household Management",
        "why": "VERIFIED #10136. Utterly non-philosophical and plentiful. Register "
               "is Victorian domestic English rather than translated antiquity — a "
               "looser register match, but the cleanest content contrast.",
        "url": "https://www.gutenberg.org/cache/epub/10136/pg10136.txt",
    },
}

SKIP_UNTIL = None  # set by --skip-until
STOP_AT = None     # set by --stop-at
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


# Vocabulary that has no business in a non-philosophical control, plus markers of
# first-person narrative fiction. Screened AFTER selection, because what matters
# is the 53 chunks that reach training, not the source as a whole.
# Whole-word only. Three words were removed after they produced pure false
# positives on genuine technical prose:
#   "sin"    matched since / sinew / sinks (the regex lacked a trailing \b)
#   "ought"  is how a manual gives instructions ("the wall ought to be stronger")
#   "honour" appeared as a proper noun ("temple of Honour and Valour")
# What is left is vocabulary that is hard to use non-normatively.
NORMATIVE_MARKERS = ("virtue", "vice", "moral", "morality", "gratitude", "the soul",
                     "wisdom", "philosophy", "philosophical", "conscience",
                     "righteous", "wickedness", "piety")
# Arguing FROM nature TO what is right is structurally the Stoic "live according
# to nature" move, and it appears in the judge rubric. It slipped past the first
# screen inside a passage about column proportions, so it is matched explicitly.
NATURE_NORM_MARKERS = ("imitate nature", "nature requires", "right arrangement",
                       "according to nature", "as nature", "nature herself",
                       "primordial substance", "these elements")
# First-person PAST narrative. Technical prose uses "I have stated" and "we ought";
# it does not narrate "I was" / "I had" at length.
NARRATIVE_MARKERS = (" i said", " i was", " i had", " he cried", " she replied")


def identify(text: str) -> dict:
    """Pull Title/Author out of the Gutenberg header and print them loudly.

    Added after a mislabelled URL silently produced a control built from a
    Victorian novel whose chunks contained first-person moral deliberation,
    while the length report showed a perfect +0.0% match. Quantity checks cannot
    catch a wrong text; only looking at it can.
    """
    import re

    got = {}
    for field in ("Title", "Author", "Translator"):
        m = re.search(rf"^{field}:\s*(.+)$", text[:4000], re.M)
        if m:
            got[field] = m.group(1).strip()
    print("\n  SOURCE IDENTITY (check this is what you intended):")
    for k, v in got.items() or [("Title", "<not found in header>")]:
        print(f"    {k}: {v}")
    return got


def looks_like_prose(t: str) -> bool:
    """Is this continuous prose at all?

    Filtered BEFORE selection, not warned about afterwards. Three separate
    non-prose forms reached the training set on earlier attempts, each slipping
    past a screen written for the previous one: index rows ("Apollo, 69, 102"),
    figure captions with layout whitespace ("Bertano (1558).    Swellings of the
    die..."), and plate lists. Screening after the fact was always one step
    behind the data, so the requirement is now positive: it must look like prose
    to be eligible.
    """
    import re

    if re.search(r"\s{3,}", t):          # layout whitespace -> caption/table
        return False
    if not re.search(r"[.!?](\s|$)", t):  # no sentence ending -> heading/label
        return False
    if len(t.split()) < 15:               # too short to be a paragraph
        return False
    if len(re.findall(r"\d", t)) / max(len(t), 1) > 0.06:  # page/plate numbers
        return False
    return True


def is_index_like(t: str) -> bool:
    """Index/table rows masquerading as paragraphs.

    Added after three chunks of the book's INDEX reached the training set
    ("Apollo, 69, 102, 103, 196; Panionion, 103, 255; ..."). The screen had only
    ever asked whether the text was philosophy, never whether it was prose.
    """
    import re

    digit_density = len(re.findall(r"\d", t)) / max(len(t), 1)
    semicolons = t.count(";")
    sentences = len(re.findall(r"[.!?]\s", t))
    return digit_density > 0.04 and semicolons >= 3 and sentences < 3


def screen(chunks: list[str]) -> list[str]:
    """Flag normative content, narrative, or non-prose in the SELECTED chunks."""
    import re

    body = " ".join(chunks)
    low = body.lower()
    n_chars = max(len(body), 1)
    hits = {w: len(re.findall(r"\b" + re.escape(w) + r"\b", low))
            for w in NORMATIVE_MARKERS}
    hits = {w: n for w, n in hits.items() if n}
    # Dialogue = quoted spans of several words. Short quoted spans are glossed
    # technical terms ("xystus", "prothyrum") and are expected in a translation.
    dialogue = [q for q in re.findall(r'"([^"]{1,200})"', body)
                if len(q.split()) >= 4]
    narrative = {m.strip(): low.count(m) for m in NARRATIVE_MARKERS}
    narrative = {m: n for m, n in narrative.items() if n}
    nature = {m: low.count(m) for m in NATURE_NORM_MARKERS}
    nature = {m: n for m, n in nature.items() if n}

    warnings = []
    index_like = [i for i, c in enumerate(chunks, 1) if is_index_like(c)]
    if index_like:
        warnings.append(f"{len(index_like)} chunk(s) look like index/table rows, not "
                        f"prose: {index_like[:8]} — use --stop-at to cut the back matter")
    if sum(hits.values()) > 5:
        warnings.append(f"normative vocabulary: {hits}")
    if len(dialogue) > 8:
        warnings.append(f"{len(dialogue)} quoted passages of 4+ words — likely dialogue, "
                        f"e.g. {dialogue[0][:60]!r}")
    if sum(narrative.values()) > 8:
        warnings.append(f"first-person past narrative: {narrative}")
    if nature:
        warnings.append(f"nature-as-normative-standard or element cosmology: {nature} "
                        "— structurally the Stoic 'according to nature' move")
    print(f"\n  CONTENT SCREEN over {n_chars} selected chars:")
    if warnings:
        for w in warnings:
            print(f"    ⚠ {w}")
    else:
        print("    clean — no normative vocabulary or narrative markers above threshold")
    return warnings


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
    if SKIP_UNTIL:
        import re
        # Gutenberg texts list every chapter title TWICE — once in the table of
        # contents, once at the section itself — and the TOC comes first. Taking
        # the FIRST match therefore skips only the front matter and silently
        # keeps the whole work, which is what happened on the first attempt here:
        # a "skip to Book VIII" dropped 2% and drew chunks from Book I onward.
        # The last match is the body occurrence.
        hits = list(re.finditer(SKIP_UNTIL, text, re.M))
        if not hits:
            raise SystemExit(f"--skip-until /{SKIP_UNTIL}/ did not match; refusing to "
                             "silently use the whole text")
        m = hits[-1]
        kept = 100 * (len(text) - m.start()) / len(text)
        print(f"  --skip-until found {len(hits)} match(es); using the LAST "
              f"(char {m.start()}, keeps {kept:.0f}% of the body)")
        if len(hits) > 1:
            print(f"    earlier matches at {[h.start() for h in hits[:-1]]} "
                  "— almost certainly the table of contents")
        if kept > 90:
            print("    ⚠ over 90% of the body kept — the anchor may not be where "
                  "you think it is; check the identity and screen output below")
        text = text[m.start():]
    if STOP_AT:
        import re
        m = re.search(STOP_AT, text, re.M)
        if not m:
            raise SystemExit(f"--stop-at /{STOP_AT}/ did not match after the skip point; "
                             "refusing to silently keep the back matter")
        print(f"  --stop-at matched at char {m.start()}; dropping the "
              f"{100 * (len(text) - m.start()) / len(text):.0f}% after it (back matter)")
        text = text[:m.start()]
    chunks = chunk_paragraphs(text)
    in_band = [c for c in chunks if lo <= len(c) <= hi]
    kept = [c for c in in_band if looks_like_prose(c)]
    print(f"  {len(chunks)} paragraphs, {len(in_band)} inside [{lo}, {hi}] chars, "
          f"{len(kept)} of those are prose ({len(in_band) - len(kept)} captions/index/"
          "headings dropped)")
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
    p.add_argument("--stop-at", metavar="REGEX",
                   help="drop everything from this regex onward (e.g. an INDEX). "
                        "Fails loudly if it does not match.")
    p.add_argument("--skip-until", metavar="REGEX",
                   help="drop everything before this regex (e.g. cut a moralising "
                        "Book 1). Fails loudly if it does not match.")
    p.add_argument("--force", action="store_true",
                   help="write even if the content screen raises warnings")
    p.add_argument("--source", help="local text file")
    p.add_argument("--url", help="Gutenberg URL to download")
    p.add_argument("--match-arm", choices=["marcus", "seneca", "epictetus"],
                   help="match one arm's length distribution instead of the pool")
    p.add_argument("--profile-only", action="store_true",
                   help="print the target profile and the candidate sources, write nothing")
    args = p.parse_args()

    if args.kind == "neutral":
        chosen, per_author = neutral_examples()
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

    global SKIP_UNTIL, STOP_AT
    SKIP_UNTIL, STOP_AT = args.skip_until, args.stop_at
    text = load_source(args.source, args.url)
    identify(text)
    cands = candidate_chunks(text, min(targets), max(targets))
    if len(cands) < len(targets):
        raise SystemExit(f"only {len(cands)} usable chunks, need {len(targets)}")
    chosen = match_lengths(cands, targets)
    stats = report(chosen, targets)
    warnings = screen(chosen)
    if warnings and not args.force:
        print("\nREFUSING to write: the selected chunks look unsuitable for a "
              "non-philosophical control.\nInspect the source, use --skip-until to cut "
              "the offending section, or pass --force if you have read the chunks and "
              "disagree.")
        return 1

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
