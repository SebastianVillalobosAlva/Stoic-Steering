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
