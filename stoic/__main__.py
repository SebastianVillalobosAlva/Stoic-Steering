"""CLI: python -m stoic <command>

Pass A, Stage 0-2 checkpoints (all $0, local CPU):

    python -m stoic stage0     # deterministic decoding
    python -m stoic stage1     # base P(stoic) == 0.542  (load-bearing)
    python -m stoic stage2     # vector cosine >=0.99 + steered dilemmas flat
    python -m stoic all        # run 0,1,2 in one model load

Each command writes one JSON checkpoint under results/<stage>/. Stage logic
lives in stoic/stages/; this module only parses arguments and dispatches.
"""

from __future__ import annotations

import argparse

from stoic import config
from stoic.stages import (
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
    parser = argparse.ArgumentParser(prog="stoic")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for c in ("stage0", "stage1", "stage2", "all"):
        sub.add_parser(c)
    p3 = sub.add_parser("stage3")
    p3.add_argument("--author", choices=list(config.AUTHORS), default=None,
                    help="run one author only (default: all three)")
    p3.add_argument("--seeds", type=int, default=5)
    p3.add_argument("--sampled", action="store_true",
                    help="matched-SAMPLED comparison (both baseline+steered sampled, temp 0.6)")
    ps = sub.add_parser("style")
    ps.add_argument("--seeds", type=int, default=5)
    sub.add_parser("stage4")
    sub.add_parser("corpus")
    pp = sub.add_parser("pairs")
    pp.add_argument("--num-pairs", type=int, default=63)
    return parser


def main():
    args = build_parser().parse_args()

    # Corpus/pairs (Pass B) don't need the model — dispatch before loading it.
    if args.cmd == "corpus":
        corpus_stage()
        return
    if args.cmd == "pairs":
        pairs_stage(args.num_pairs)
        return

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
    elif args.cmd == "all":
        stage0(model, tokenizer)
        _, baseline = stage1(model, tokenizer)
        stage2(model, tokenizer, baseline=baseline)


if __name__ == "__main__":
    main()
