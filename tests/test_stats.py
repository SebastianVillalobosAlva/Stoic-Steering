"""The statistics the READMEs cite must be computable from the checked-in JSONs."""

import glob
import json
import math
from pathlib import Path

import pytest

from stoic.dilemmas import paired_stats, sign_test, _logit

ROOT = Path(__file__).resolve().parent.parent


# The runs the READMEs actually cite, named explicitly. Pinning matters: these
# numbers are a claim about a specific historical run, not about whatever ran
# most recently. Resolving them by "latest file" meant the next stage-4 run
# would be compared against the previous run's expectations and fail for a
# reason unrelated to the code.
PUBLISHED_STAGE4 = (
    "results/stage4_lora_dilemmas/lora_dilemmas_20260705_225558.json",
    "results/stage4_lora_dilemmas/lora_dilemmas_20260717_152132.json",
)
PUBLISHED_SIGN_TESTS = {
    "marcus": (27, 13, 0.0385),
    "seneca": (25, 15, 0.1539),
    "epictetus": (17, 23, 0.4296),
}


def _sign_tests_from(path: Path) -> dict:
    """Recompute each arm's sign test from ONE file's own numbers.

    Each stage-4 checkpoint embeds the `baseline_p_stoic` it was measured
    against, so baseline and steered always come from the same run. Pairing a
    baseline file with an adapter file by two independent globs could silently
    mix runs.
    """
    payload = json.load(open(path))
    base = payload["baseline_p_stoic"]
    return {
        arm: sign_test({k: v["steered_p_stoic"][k] - base[k] for k in base})
        for arm, v in payload["per_author"].items()
    }


@pytest.mark.parametrize("rel_path", PUBLISHED_STAGE4)
def test_sign_test_reproduces_published_numbers(rel_path):
    """Marcus 27+/13- p=.038, Seneca 25+/15- p=.154 (the cited n.s.),
    Epictetus 17+/23- p=.430.

    CAVEAT (2026-07-28): these are not version-stable. Under torch 2.8.0 against
    the pinned 2.5.1, Marcus becomes 24+/16- p=.268 — three items whose |delta|
    sat inside ~4e-3 of fp16 CPU drift flipped sign. Seneca and Epictetus hold.
    Every magnitude statistic (means, t, bucket p) reproduces to ~4 dp. See
    docs/decisions.md. This test asserts what the named files contain, which is
    exactly the published record, and is unaffected by newer runs.
    """
    path = ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present")
    computed = _sign_tests_from(path)
    for arm, (pos, neg, p) in PUBLISHED_SIGN_TESTS.items():
        s = computed[arm]
        assert (s["pos"], s["neg"]) == (pos, neg), arm
        assert s["p_two_sided"] == pytest.approx(p, abs=1e-4), arm


def test_stored_sign_tests_match_recomputation():
    """Every stage-4 checkpoint's stored `sign_test` must equal a fresh
    recomputation from the same file.

    Version-agnostic on purpose: this is the check that should cover *new*
    runs. It asserts internal consistency rather than a historical value, so a
    future run under a different torch adds coverage instead of turning the
    suite red.
    """
    files = sorted(glob.glob(str(ROOT / "results/stage4_lora_dilemmas/lora_dilemmas_*.json")))
    if not files:
        pytest.skip("no stage-4 checkpoints present")
    checked = 0
    for f in files:
        payload = json.load(open(f))
        computed = _sign_tests_from(Path(f))
        for arm, v in payload["per_author"].items():
            stored = v.get("sign_test")
            if stored is None:
                continue  # pre-dates sign_test being wired into the stage
            s = computed[arm]
            assert (stored["pos"], stored["neg"], stored["ties"]) == \
                   (s["pos"], s["neg"], s["ties"]), f"{Path(f).name}:{arm}"
            assert stored["p_two_sided"] == pytest.approx(s["p_two_sided"]), f"{Path(f).name}:{arm}"
            checked += 1
    if checked == 0:
        pytest.skip("no checkpoint carries a stored sign_test")


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
