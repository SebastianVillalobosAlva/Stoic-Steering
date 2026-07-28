"""CLI: python -m stoic [--axis NAME] <command>

Pass A, Stage 0-2 checkpoints (all $0, local CPU):

    python -m stoic stage0     # deterministic decoding
    python -m stoic stage1     # base P(stoic) == 0.542  (load-bearing)
    python -m stoic stage2     # vector cosine >=0.99 + steered dilemmas flat
    python -m stoic all        # run 0,1,2 in one model load

Each command writes one JSON checkpoint under results/<stage>/. Stage logic
lives in stoic/stages/; this module only parses arguments and dispatches.

--- Why the axis is bound before the imports below ---

`--axis` selects which behavioral axis the whole package runs against, and the
axis is resolved once at import time (`stoic.axis.ACTIVE`). argparse runs far
too late for that: by the time it parses `--axis`, `from stoic import config`
has already executed and the axis is fixed. The flag would have looked like it
worked while quietly doing nothing — and only the env-var path would have been
real.

So the flag is read straight off `sys.argv` *above* every `stoic` import, and
exported as STOIC_AXIS. argparse still declares `--axis` so `--help` documents
it, with its default read back from the environment, which is why `args.axis`
and `ACTIVE.name` agree on both paths.
"""

from __future__ import annotations

import argparse
import os
import sys

# Duplicated from stoic.axis rather than imported: reading the name from there
# would require importing stoic.axis, which is exactly what must not happen
# until this variable is set.
AXIS_ENV_VAR = "STOIC_AXIS"


def _bind_axis_from_argv(argv: list[str]) -> str | None:
    """Set STOIC_AXIS from `--axis NAME` / `--axis=NAME`; return the name or None.

    Runs before any `stoic` import. Kept as a plain function so it can be
    tested in-process, without spawning a subprocess to prove the flag works.
    """
    for i, token in enumerate(argv):
        if token == "--axis" and i + 1 < len(argv):
            name = argv[i + 1]
        elif token.startswith("--axis="):
            name = token.split("=", 1)[1]
        else:
            continue
        os.environ[AXIS_ENV_VAR] = name
        return name
    return None


# Order matters: capture what the environment said BEFORE the flag overwrites it.
_ENV_AXIS_AT_IMPORT = os.environ.get(AXIS_ENV_VAR)
_FLAG_AXIS = _bind_axis_from_argv(sys.argv)

from stoic import config  # noqa: E402  (must follow the axis binding above)
from stoic.axis import ACTIVE  # noqa: E402
from stoic.stages import (  # noqa: E402
    calibrate_stage,
    corpus_stage,
    pairs_stage,
    stage0,
    stage1,
    stage2,
    stage3,
    stage4,
    style_check,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stoic",
        description=f"active axis: {ACTIVE.name} ({ACTIVE.display})",
    )
    # Default read back from the environment, which _bind_axis_from_argv has
    # already set if the flag was given — so this agrees with ACTIVE either way.
    parser.add_argument(
        "--axis", default=os.environ.get(AXIS_ENV_VAR, "stoic"),
        help="behavioral axis to run against (a directory under axes/). "
             "Also settable as STOIC_AXIS.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for c in ("stage0", "stage1", "stage2", "all"):
        sub.add_parser(c)
    p3 = sub.add_parser("stage3")
    p3.add_argument("--author", choices=list(ACTIVE.arms), default=None,
                    help=f"run one arm only (default: all {len(ACTIVE.arms)})")
    p3.add_argument("--seeds", type=int, default=5)
    p3.add_argument("--sampled", action="store_true",
                    help="matched-SAMPLED comparison (both baseline+steered sampled, temp 0.6)")
    ps = sub.add_parser("style")
    ps.add_argument("--seeds", type=int, default=5)
    sub.add_parser("stage4")
    sub.add_parser("corpus")
    pp = sub.add_parser("pairs")
    pp.add_argument("--num-pairs", type=int, default=63)
    pc = sub.add_parser("calibrate")
    pc.add_argument("--items", default=str(ACTIVE.candidates_file),
                    help="candidate dilemmas_v3 JSON (default: this axis's candidates_file)")
    pc.add_argument("--tolerance", type=float, default=0.05,
                    help="per-cell gate: |mean P(stoic) - 0.5| <= tolerance")
    pc.add_argument("--cell-size", type=int, default=None,
                    help="expected items per cell (structural check; default: any)")
    pc.add_argument("--validate-only", action="store_true",
                    help="structural checks only — no model load, $0")
    return parser


def main():
    # A flag that disagrees with the environment is an error, not a preference:
    # picking one silently would attribute a run's numbers to the wrong axis,
    # which is the failure this whole binding path exists to prevent.
    if _FLAG_AXIS and _ENV_AXIS_AT_IMPORT and _FLAG_AXIS != _ENV_AXIS_AT_IMPORT:
        raise SystemExit(
            f"--axis {_FLAG_AXIS!r} disagrees with {AXIS_ENV_VAR}={_ENV_AXIS_AT_IMPORT!r}. "
            f"Unset {AXIS_ENV_VAR} or drop the flag."
        )

    args = build_parser().parse_args()
    # Invariant, cheap to check: whichever path set the axis, the parsed value
    # and the axis actually loaded must be the same one.
    assert args.axis == ACTIVE.name, (
        f"axis binding diverged: --axis/{AXIS_ENV_VAR} says {args.axis!r}, "
        f"loaded axis is {ACTIVE.name!r}"
    )

    # Corpus/pairs (Pass B) don't need the model — dispatch before loading it.
    if args.cmd == "corpus":
        corpus_stage()
        return
    if args.cmd == "pairs":
        pairs_stage(args.num_pairs)
        return
    if args.cmd == "calibrate" and args.validate_only:
        from stoic.calibrate import load_candidates, validate_items

        items = load_candidates(args.items)
        problems = validate_items(items, cell_size=args.cell_size)
        print(f"{args.items}: {len(items)} items")
        for p in problems:
            print(f"  ✗ {p}")
        print("valid ✓" if not problems else f"{len(problems)} problem(s)")
        raise SystemExit(0 if not problems else 1)

    from stoic.model import load_model

    model, tokenizer = load_model()

    if args.cmd == "stage0":
        stage0(model, tokenizer)
    elif args.cmd == "stage1":
        stage1(model, tokenizer)
    elif args.cmd == "stage2":
        stage2(model, tokenizer)
    elif args.cmd == "stage3":
        authors = [args.author] if args.author else None
        stage3(model, tokenizer, authors=authors, n_seeds=args.seeds, sampled=args.sampled)
    elif args.cmd == "style":
        style_check(model, tokenizer, n_seeds=args.seeds)
    elif args.cmd == "stage4":
        stage4(model, tokenizer)
    elif args.cmd == "calibrate":
        calibrate_stage(model, tokenizer, args.items,
                        tolerance=args.tolerance, cell_size=args.cell_size)
    elif args.cmd == "all":
        stage0(model, tokenizer)
        _, baseline = stage1(model, tokenizer)
        stage2(model, tokenizer, baseline=baseline)


if __name__ == "__main__":
    main()
