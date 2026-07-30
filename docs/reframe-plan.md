# Reframe audit — philosophy-led → locus-led

**What this is.** A map, not a rewrite. Every place the repo leads with Stoic philosophy where the real claim is **intervention locus** — *does a behavior installed in weight-space recruit the same circuit as one installed in activation-space, and does the locus predict how the behavior fails?* Stoicism is the **instrument** that builds a behavioral axis, not the subject.

Sebastian writes all external-facing prose. This file gives location, current framing, and a one-line beat for the locus reframe — **no paragraphs to ship.**

**Target voice:** `CLAUDE.md` "What this repo is about" is already reframed to locus; it is the reference, not a violation. `docs/measurement-artifact.md` is the evals-hygiene / artifact story (story beat 1), locus-neutral — no change.

------------------------------------------------------------------------

## README.md — external-facing, top priority

| Location | Current framing (philosophy-led) | Locus reframe (beat) | Priority |
|------------------|------------------|------------------|------------------|
| L1 `# Stoic Steering` | Title names the instrument. | Lead the title with the locus claim (weight-space vs activation-space install; same behavior, different circuit). | **High** |
| L3–5 tagline | "Steering Stoic philosophical reasoning into Llama-3.2-3B via CAA and LoRA…" — Stoicism as subject. | Stoicism is the instrument; the subject is same-behavior/different-locus and whether locus predicts failure. | **High** |
| L7–14 "Core finding" | CAA vs LoRA (locus-aligned) but framed around "Stoic register." | Keep the CAA/LoRA spine; recast the finding as a **locus dissociation demonstrated on a Stoic axis**. | Med |
| L18–81 "Key findings" | Bullets already CAA-vs-LoRA; heavy Stoic-idiom ("stoic-content circuit"). | Mostly aligned — keep instrument language but frame each bullet as evidence about locus, not about Stoicism. | Med |
| L85–98 "Method" | "Two interventions compared" spine is already locus. | Tighten the lead sentence to locus; the three-depths structure stays. | Low |
| L100–101 "Philosophers studied … three Stoic traditions done rigorously" | Framed as philosophical depth. | One behavioral axis, three corpora as instrument. This is where the **n=1-in-behavior** caveat belongs; ties to the sycophancy second-axis plan. | Med |
| Repo-name / quickstart refs (L133–166) | Names the instrument throughout. | Low priority — see rename dependency chain below. | Low |

### Rename dependency chain (in-repo; size, not a recommendation)

Three names already diverge: local dir `llm-stoic`, GitHub `Stoic-Steering`, HF `seb-vil/llama-3.2-3b-stoic-steering`.

| Reference | File / location | Breaks on rename? |
|------------------------|------------------------|------------------------|
| `HF_REPO` constant | `scripts/fetch_artifacts.py:25` | **Yes** — fetch fails if the HF repo is renamed. |
| HF path in fetch/clone instructions | `README.md:161` | **Yes** — must move in lockstep with `fetch_artifacts.py`. |
| Description string "Stoic-steering pipeline" | `pyproject.toml:8` | No — cosmetic string, no import breakage. |
| Frozen-binary manifest | `data/MANIFEST.sha256` | **No — name-independent.** Paths are repo-relative (`data/reference/…`, `models/…`); the HF mapping lives in `_hf_path()` in `fetch_artifacts.py`, not the manifest. |
| HF repo path itself | external | Renaming the HF repo forces the two in-repo refs above to change together. |
| ModelLens cross-links | separate repo (see Off-repo) | GitHub redirects on rename → link cost bounded; framing cost is not. |

Bounded: a GitHub rename redirects old links; the only hard breaks are the two HF-path references, which move together.

------------------------------------------------------------------------

## results/README.md — internal record, medium/low priority

| Location | Current framing | Locus reframe (beat) | Priority |
|------------------|------------------|------------------|------------------|
| L1 title "Verification record" + Exp 12 headers (L142+) | Already lead with circuit / method split. | Mostly clean — no lead change needed. | Low |
| Throughout | "stoic-content circuit", "Stoic option" instrument phrasing. | Keep as instrument language; frame as *the axis under test*, not the topic. | Low |

Internal artifact — Sebastian's call whether to touch at all.

------------------------------------------------------------------------

## REPRODUCTION_PLAN.md — internal, low priority

| Location | Current framing | Locus reframe (beat) | Priority |
|------------------|------------------|------------------|------------------|
| L1–6 title / purpose "Stoic-steering codebase" | Instrument-named. | Doc can stay instrument-named; ensure any **external pointer** to it carries the locus framing. | Low |

------------------------------------------------------------------------

## docs/measurement-artifact.md — no change needed

The decoding-asymmetry artifact writeup is the evals-hygiene story (story beat 1) and is locus-neutral. Listed here so coverage is explicit, not silent.

------------------------------------------------------------------------

## Off-repo surfaces — manual checklist (no fetching, no editing)

Outside this working tree. Check each by hand; do not action from this task.

| Surface | What to check | Why |
|------------------------|------------------------|------------------------|
| **HuggingFace model card** `seb-vil/llama-3.2-3b-stoic-steering` | Card leads with the locus claim, not "Stoic steering." | Public, high-traffic external surface. |
| **GitHub About field + repo topics** | One-liner and topics lead with intervention-locus. | First thing a reviewer reads. |
| **Portfolio blurb** (sebastianvillalobosalva.github.io) | Project blurb matches the locus framing. | External-facing; consistency with README. |
| **ModelLens README** (separate repo, accessible at `../modellens` — lines verified, not guessed) | `README.md` **L13, L18, L24, L28, L39** describe the companion project as "steers Llama toward Stoic philosophy" and link `github.com/SebastianVillalobosAlva/Stoic-Steering`. | **New-contradiction risk:** if Stoic-Steering reframes to locus and ModelLens still calls it "the Stoic project," the two repos now disagree — *worse* than the current consistent-but-wrong framing. Edit belongs in the ModelLens repo. |

------------------------------------------------------------------------

## Coverage note

Audited: `README.md`, `results/README.md`, `REPRODUCTION_PLAN.md`, `docs/measurement-artifact.md` (in-repo); HF card, GitHub About/topics, portfolio blurb, ModelLens README (off-repo checklist). `CLAUDE.md` is the target locus voice, used as reference, not audited as a violation. No prose rewritten; no results JSON touched; Task 2 not started.