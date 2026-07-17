"""The statistics the READMEs cite must be computable from the checked-in JSONs."""

import glob
import json
import math
from pathlib import Path

import pytest

from stoic.dilemmas import paired_stats, sign_test, _logit

ROOT = Path(__file__).resolve().parent.parent


def _latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(ROOT / pattern)))
    return Path(hits[-1]) if hits else None


def test_sign_test_reproduces_published_numbers():
    """Marcus 27+/13- p=.038, Seneca 25+/15- p=.154 (the cited n.s.),
    Epictetus 17+/23- p=.430 — from the stage 1 + stage 4 checkpoint JSONs."""
    base_file = _latest("results/stage1_dilemma_baseline/baseline_*.json")
    lora_file = _latest("results/stage4_lora_dilemmas/lora_dilemmas_*.json")
    if base_file is None or lora_file is None:
        pytest.skip("checkpoint JSONs not present")

    base = json.load(open(base_file))["baseline_p_stoic"]
    per_author = json.load(open(lora_file))["per_author"]

    expected = {
        "marcus": (27, 13, 0.0385),
        "seneca": (25, 15, 0.1539),
        "epictetus": (17, 23, 0.4296),
    }
    for author, (pos, neg, p) in expected.items():
        steered = per_author[author]["steered_p_stoic"]
        s = sign_test({k: steered[k] - base[k] for k in base})
        assert (s["pos"], s["neg"]) == (pos, neg), author
        assert s["p_two_sided"] == pytest.approx(p, abs=1e-4), author


def test_sign_test_all_ties():
    s = sign_test([0.0, 0.0, 0.0])
    assert s == {"pos": 0, "neg": 0, "ties": 3, "n": 0, "p_two_sided": 1.0}


def test_sign_test_symmetric_is_one():
    s = sign_test([1, -1, 2, -2])
    assert s["p_two_sided"] == pytest.approx(1.0)


def test_sign_test_extreme():
    # 10/10 one direction: p = 2 * (1/2)^10
    s = sign_test([0.1] * 10)
    assert s["pos"] == 10 and s["neg"] == 0
    assert s["p_two_sided"] == pytest.approx(2 / 1024)


def test_sign_test_tolerance_treats_tiny_as_tie():
    s = sign_test([1e-12, -1e-12, 0.5])
    assert s["ties"] == 2 and s["pos"] == 1


def test_paired_stats_hand_values():
    st = paired_stats([1.0, 2.0, 3.0])
    assert st["n"] == 3
    assert st["mean_delta"] == pytest.approx(2.0)
    assert st["std"] == pytest.approx(1.0)
    assert st["t_stat"] == pytest.approx(2.0 / (1.0 / math.sqrt(3)))


def test_paired_stats_p_value_matches_scipy_if_available():
    scipy = pytest.importorskip("scipy")
    from scipy import stats as sps

    deltas = [0.3, -0.1, 0.2, 0.5, 0.0]
    st = paired_stats(deltas)
    ref = sps.ttest_1samp(deltas, 0.0)
    assert st["p_value"] == pytest.approx(float(ref.pvalue))
    assert st["t_stat"] == pytest.approx(float(ref.statistic))


def test_logit_basics():
    assert _logit(0.5) == pytest.approx(0.0)
    assert _logit(0.9) == pytest.approx(-_logit(0.1))
    # clipping keeps extremes finite
    assert math.isfinite(_logit(0.0)) and math.isfinite(_logit(1.0))
