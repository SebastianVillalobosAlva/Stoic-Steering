"""The CLI surface: every documented command parses, flags keep their defaults."""

import pytest

from stoic.__main__ import build_parser


def test_all_documented_commands_exist():
    parser = build_parser()
    sub = parser._subparsers._group_actions[0]
    assert set(sub.choices) == {
        "stage0", "stage1", "stage2", "stage3", "stage4",
        "all", "style", "corpus", "pairs", "calibrate",
    }


def test_stage3_flags():
    args = build_parser().parse_args(["stage3", "--author", "seneca", "--seeds", "2", "--sampled"])
    assert args.author == "seneca" and args.seeds == 2 and args.sampled is True
    args = build_parser().parse_args(["stage3"])
    assert args.author is None and args.seeds == 5 and args.sampled is False


def test_pairs_default_matches_frozen_generation():
    args = build_parser().parse_args(["pairs"])
    assert args.num_pairs == 63


def test_calibrate_flags():
    args = build_parser().parse_args(
        ["calibrate", "--items", "x.json", "--tolerance", "0.1",
         "--cell-size", "10", "--validate-only"])
    assert args.items == "x.json" and args.tolerance == 0.1
    assert args.cell_size == 10 and args.validate_only is True
    args = build_parser().parse_args(["calibrate"])
    assert args.items.endswith("data/generated/dilemmas_v3_candidates.json")
    assert args.tolerance == 0.05 and args.cell_size is None
    assert args.validate_only is False


def test_unknown_command_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["stage9"])
