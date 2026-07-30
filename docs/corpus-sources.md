# Corpus sources & provenance

The raw philosopher texts are public-domain Project Gutenberg editions. The
acquisition code (`stoic/corpus.py`) reads this manifest from
`data/reference/config/sources.json` and reproduces the exact editions, slicing
boundaries, and paragraph chunking that produced the frozen reference corpus in
`data/reference/chunked/`. This file records that provenance in-repo; do not
change the IDs or boundaries without re-freezing downstream artifacts.

| Author | Work | Gutenberg ID | URL | Encoding |
|---|---|---|---|---|
| Marcus Aurelius | Meditations | **2680** | `https://www.gutenberg.org/files/2680/2680-0.txt` | UTF-8 |
| Seneca | Moral letters (Of Benefits) | **56075** | `https://www.gutenberg.org/cache/epub/56075/pg56075.txt` | UTF-8 |
| Epictetus | Enchiridion | **45109** | `https://www.gutenberg.org/cache/epub/45109/pg45109.txt` | UTF-8 |

## Slicing boundaries (verbatim from the manifest)

Each text is narrowed in three stages: strip the Gutenberg license wrapper
(`*** START/END OF THE PROJECT GUTENBERG EBOOK … ***`), then skip front-matter up
to `content_start`, then cut back-matter from `content_end` (both case-insensitive
regex, applied inside the license-stripped body).

| Work | `content_start` | `content_end` |
|---|---|---|
| meditations | `MARCUS AURELIUS ANTONINUS THE ROMAN EMPEROR\|HIS FIRST BOOK` | `\n\s*APPENDIX\|CORRESPONDENCE OF M\.` |
| moral_letters | `SENECA OF BENEFITS\.` | `\n\s*Transcriber's note:` |
| enchiridion | `There are things which are within our power` | `\n\s*Footnotes` |

## Chunking

Paragraph split: remove `[123]`-style footnote markers, split on blank lines,
strip, drop empties. Each surviving paragraph is one chunk. Frozen reference
chunk counts (the regression target for a re-run):

| Author | Work | chunks |
|---|---|---|
| Marcus Aurelius | meditations | 437 |
| Seneca | moral_letters | 540 |
| Epictetus | enchiridion | 123 |

## Reproducing

```bash
python -m stoic corpus     # download -> slice -> chunk into data/generated/,
                           # then report chunk counts vs the frozen reference
```

Downloaded texts and chunks land under `data/generated/` (gitignored). A re-run
should match the counts above; Project Gutenberg occasionally re-edits a file, so
a small drift is possible and is reported rather than forced.

---

## Control-adapter corpora (v3 behavioral eval)

Recorded here because `data/generated/lora_training/*.meta.json` is gitignored —
without this section the controls' provenance would live only in an untracked
file and in commit messages. Purpose and reading of these adapters:
[docs/pre-registration-v3.md](pre-registration-v3.md) §4.

| Control | Source | Gutenberg ID | Slice | n | mean chars |
|---|---|---|---|---:|---:|
| `lora_control_nonphil` | Vitruvius, *The Ten Books on Architecture*, **Book X only** | **20239** | `--skip-until "HOISTING MACHINES"` (last match), `--stop-at "^INDEX"` | 53 | 565 |
| `lora_control_neutral` | `neutral_text` halves of the frozen `neutral_pairs.json` | — | stratified 18/18/17 by arm, quantile-spaced by length | 53 | 1432 |

Build commands (`scripts/prep_control_lora.py`):

```
python scripts/prep_control_lora.py --kind nonphilosophical \
  --url https://www.gutenberg.org/cache/epub/20239/pg20239.txt \
  --skip-until "HOISTING MACHINES" --stop-at "^INDEX"
python scripts/prep_control_lora.py --kind neutral
```

### Why Book X specifically, and what was rejected

Four candidate sources or slices were rejected before this one. Each rejection
was a real content fault, and each was caught only by reading the selected
chunks — never by the length report, which showed a good match every time.

| Rejected | Why |
|---|---|
| Frontinus, *De Aquaeductu* | Recommended first, then found **not to exist on Gutenberg in English** (only *Les stratagèmes*, French, #16237). |
| Gutenberg **#46345** | The URL given for Frontinus is actually *Sir Quixote of the Moors*, a John Buchan novel. Its chunks contained first-person moral deliberation ("bound in common honor to requite their kindness"). |
| Vitruvius, whole work | `--skip-until` took the *first* regex match, which is the table of contents, so the slice kept the entire book. Chunks included Book II's element cosmology and a Book III passage arguing "we **ought** to imitate nature… it is the **right arrangement**" — structurally the Stoic *according to nature* move, which also appears in the judge rubric. |
| Vitruvius, Book X with back matter | Swept in the book's INDEX ("Apollo, 69, 102, 103, 196; …") and figure captions with layout whitespace ("Bertano (1558).    Swellings of the die…"). |

Book X (machines, hoists, ballistae, water screws, water organs, siege
engines) has no cosmology preface and no nature-as-standard arguments. Its
historical anecdotes — Diognetus at Rhodes, the Ephesus quarry — are narrative,
not moral instruction.

### Guards the script now enforces

Added in response to the failures above; a length match alone certifies nothing.

- **Identity** — prints `Title`/`Author` from the Gutenberg header before anything else.
- **Prose filter, applied before selection** — rejects layout whitespace, chunks
  with no sentence ending, fewer than 15 words, or >6% digits. Positive
  requirement rather than a blacklist, because three successive non-prose forms
  each slipped past a screen written for the previous one.
- **Content screen, applied after selection** — normative vocabulary, dialogue
  (quoted spans of 4+ words), first-person past narrative, index-like rows, and
  nature-as-normative-standard phrasing. Refuses to write on failure; `--force`
  overrides only after the chunks have been read.
- **Anchors fail loudly** — `--skip-until` uses the *last* match and reports how
  many it found; `--stop-at` errors rather than silently keeping back matter.
