# Claim-revision map — after the v4 powered eval

**What v4 established (2026-07-31).** The v2 "LoRA reaches decisions" result is
lexical echo, not reasoning. On a calibrated, powered set (100 items, 25/cell)
Seneca's decision effect is **register-dependent**: idiom-worded items move
+0.099 log-odds, plain-worded items move −0.083; the off-topic/plain reasoning
cell is bounded-null (mean −0.07, 95% CI excludes an idiom-sized +0.15). Marcus
and Epictetus are null on the axis.

**This is a map, not a rewrite.** Sebastian owns the prose. Below: every
external-facing claim that v4 touches, its location, why it changes, and
proposed narrowed language to adapt. Nothing here has been shipped as his.

**The framing to carry (a recommendation, not a decision).** This is the
story-beat-1 shape, not a retraction: *I built the instrument that could
falsify my own headline claim, pre-registered the test, and it did.* The
decision claim narrows; the credibility of the method goes **up**. Lead with the
confound-caught, not with the loss.

---

## The numbers, once, for reuse

| arm | overall Δlog-odds (t) | phrasing idiom/plain (p) | reasoning cell mean, 95% CI (p) | exceeds both controls |
|---|---|---|---|---|
| **Seneca** | **+0.008 (0.21)** — was +0.135 (2.34) on n=40 | **+0.099 / −0.083 (0.015)** | −0.070, [−0.196, +0.056] (0.282) | phrasing ✓, reasoning ✗ |
| Marcus | −0.041 (−1.85) | −0.002 / −0.080 (0.078) | −0.102, [−0.179, −0.025] (0.015) | none |
| Epictetus | −0.010 (−1.47) | (ns) | −0.025 (0.025) | none; R6 confounded r=−0.997 |
| control_nonphil | −0.036 (−1.44) | +0.012 / −0.085 (0.055) | −0.094 (0.038) | — |
| control_neutral | −0.205 (−2.47) | (ns) | −0.402 (0.021) | — |

Base integrity: v2 ruler 0.541609 → 0.541609 (torch 2.8 drift value; the published fixture is 0.541602, see docs/decisions.md), drift 0. Seneca R6 clean
(r=−0.32, p=0.68). Reading, per the pre-registered §4: **ECHO**.

Two honesty notes to keep with any restatement:
1. The non-philosophical control shows a *weak, same-direction, non-significant*
   idiom>plain trend (+0.096, p=0.055). Any fine-tuning on formal prose slightly
   favours idiom-phrased items; Seneca's effect is significantly larger and is
   the only significant one — but it sits on top of that baseline.
2. Marcus's significant negative reasoning cell is **not** an anti-Stoic finding;
   it is matched by the inert control and is generic fine-tuning drift.

---

## Site-by-site

### README.md

| # | Line(s) | Current claim | Why it changes | Proposed direction |
|---|---|---|---|---|
| R1 | 3 (tagline) | "Only LoRA changes the circuit, **and reaches decisions**." | "reaches decisions" is the narrowed claim. | "…and moves decisions **through register, not reasoning** (n=100 powered test)." Applied 2026-10-16 in commit 160bc35. |
| R2 | 5 (core finding) | "LoRA moves the judge-free decision instrument…" | True on v2; on the powered set the movement is phrasing-driven. | Keep the CAA/LoRA split; add that the decision movement is register-dependent, one clause. Applied 2026-10-16 in commit 160bc35. |
| R3 | 17–24 ("LoRA reaches decisions" table) | v2 numbers presented as the decision result. | v2 numbers are correct **as v2**; they do not survive at n=100. | Keep the table, relabel it "v2 (n=40)", add the v4 row / a one-line "superseded — see v4" note under it. Receipts rule: do not delete. |
| R4 | 26 (caveat) | "The non-philosophical control adapter in v3 is what separates them." | It now exists and separated them — and it is not inert. | Update to past tense with the result: control moved −0.036, Marcus's cell matched it. Applied 2026-10-16 in commit 48471e5. |
| R5 | 30 ("LoRA does not install uniform Stoic reasoning") | "…a possible Senecan-idiom lexical-echo confound in the decision instrument is also open." | The confound is no longer *open* — it was tested and **confirmed as the dominant effect**. | This sentence becomes the headline of the paragraph, not its final caveat. |
| R6 | 7, 28, 32–34 (figures) | Three-depths + LoRA-shift figures reflect v2. | The decision panel is now echo. | New figure (task): v4 idiom-vs-plain by arm with controls. The three-depths figure needs a v4 footnote or a decision-panel redraw. Applied 2026-10-16 in commit 48471e5. |

### results/README.md

| # | What | Change |
|---|---|---|
| S1 | The LoRA decision section (v2 numbers, 25/40 sign test, per-stance) | Stays as the v2 record. Add a v4 subsection: the powered result, the echo reading, the bounded-null CI, base integrity. Cross-link `results/v4_behavioral/`. |
| S2 | Any "reaches decisions" summary line | Narrow to register-not-reasoning, pointer to v4. |

### docs/measurement-artifact.md

| # | What | Change |
|---|---|---|
| M1 | The decoding-asymmetry artifact write-up | **Unaffected as an artifact story**, but v4 is a second, cleaner instance of the same discipline (caught my own confound). Optionally add a one-line "see also v4 echo result" — the two together are the strongest version of story beat 1. |

### Off-repo surfaces (checklist, not files here)

| # | Surface | Change |
|---|---|---|
| O1 | HF model card (`seb-vil/llama-3.2-3b-stoic-steering`) | If it says "reaches decisions", narrow it. |
| O2 | GitHub About / repo description | Same. |
| O3 | Portfolio / application blurb, CV bullets | The strongest phrasing is now the confound-caught, not the decision-mover. "Found and confirmed a lexical-echo confound in my own decision result via a pre-registered powered test." |
| O4 | Any draft write-up / abstract | Beat 3 rewrites from "only weight adaptation reaches decisions" to "weight adaptation moves decisions through register; a pre-registered powered test rules out reasoning on off-topic items." |

---

## What does NOT change (guard against over-correcting)

- **CAA is still null everywhere.** v4 does not touch the CAA result; the
  activation-space arm moved nothing at style, content, decision, or circuit.
- **The locus thesis is intact.** "Weight-space and activation-space install
  behavior differently" stands — if anything sharpened: weight-space reaches the
  *decision instrument* (via register), activation-space reaches nothing.
- **The measurement-artifact story stands** and is reinforced.
- **The v2 numbers are real** as a v2 result; they are receipts, not errors.
- **Epictetus** remains an underpowered-corpus null, now with a quantified
  inertness (53/100 items tie) and a baseline confound flag.

---

## Suggested order of operations

1. results/README.md — the primary numeric record; add the v4 subsection first.
2. README.md — apply R1–R5 as narrowings, R3 via the receipts pointer.
3. Figure 1 refresh / new v4 decision panel (separate task).
4. Off-repo surfaces last, once the in-repo language is settled.

Log the retirement through `/experiment-claim-tracker` so the v2 decision claim
is superseded-with-pointer rather than edited away.
