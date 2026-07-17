"""The CLI surface: every documented command parses, flags keep their defaults."""

import pytest

from stoic.__main__ import build_parser


def test_all_documented_commands_exist():
    parser = build_parser()
    sub = parser._subparsers._group_actions[0]
    assert set(sub.choices) == {
        "stage0", "stage1", "stage2", "stage3", "stage4",
        "all", "style", "corpus", "pairs",
    }


def test_stage3_flags():
    args = build_parser().parse_args(["stage3", "--author", "seneca", "--seeds", "2", "--sampled"])
    assert args.author == "seneca" and args.seeds == 2 and args.sampled is True
    args = build_parser().parse_args(["stage3"])
    assert args.author is None and args.seeds == 5 and args.sampled is False


def test_pairs_default_matches_frozen_generation():
    args = build_parser().parse_args(["pairs"])
    assert args.num_pairs == 63


def test_unknown_command_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["stage9"])
