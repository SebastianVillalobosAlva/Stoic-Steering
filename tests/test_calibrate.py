"""dilemmas_v3 calibration harness: structural validation and the gate math."""

from pathlib import Path

import pytest

from stoic.calibrate import (
    CELLS,
    calibration_report,
    cell_of,
    load_candidates,
    validate_items,
)

FIXTURE = Path(__file__).parent / "fixtures" / "dilemmas_v3_example.json"


def _item(id, topic="core", phrasing="plain", stance="accepting"):
    return {
        "id": id, "topic_axis": topic, "phrasing": phrasing,
        "stoic_stance": stance, "situation": "s", "stoic": "a", "nonstoic": "b",
    }


def _balanced_set(per_cell=2):
    """One stance-balanced set covering all four cells."""
    items = []
    for t in ("core", "offtopic"):
        for p in ("plain", "idiom"):
            for i in range(per_cell):
                stance = "accepting" if i % 2 == 0 else "active"
                items.append(_item(f"{t}_{p}_{i}", t, p, stance))
    return items


# --- structural validation -------------------------------------------------

def test_example_fixture_is_valid():
    items = load_candidates(FIXTURE)
    assert len(items) == 8
    assert validate_items(items, cell_size=2) == []
    assert {cell_of(d) for d in items} == set(CELLS)


def test_valid_synthetic_set_passes():
    assert validate_items(_balanced_set(), cell_size=2) == []


def test_missing_field_flagged():
    items = _balanced_set()
    items[0]["situation"] = "  "
    assert any("situation" in p for p in validate_items(items))


def test_bad_axis_values_flagged():
    items = _balanced_set()
    items[0]["topic_axis"] = "meta"
    items[1]["phrasing"] = "florid"
    items[2]["stoic_stance"] = "neutral"
    problems = validate_items(items)
    assert any("topic_axis" in p for p in problems)
    assert any("phrasing" in p for p in problems)
    assert any("stoic_stance" in p for p in problems)


def test_duplicate_id_flagged():
    items = _balanced_set()
    items[1]["id"] = items[0]["id"]
    assert any("duplicate id" in p for p in validate_items(items))


def test_empty_cell_flagged():
    items = [d for d in _balanced_set() if cell_of(d) != "offtopic_idiom"]
    assert any("offtopic_idiom: empty" in p for p in validate_items(items))


def test_wrong_cell_size_flagged():
    problems = validate_items(_balanced_set(per_cell=2), cell_size=10)
    assert sum("expected 10" in p for p in problems) == 4


def test_stance_imbalance_flagged_even_cell():
    items = _balanced_set()
    for d in items:
        if cell_of(d) == "core_plain":
            d["stoic_stance"] = "accepting"  # 2/0 in an even cell
    assert any("core_plain: stance imbalance" in p for p in validate_items(items))


def test_stance_off_by_one_allowed_in_odd_cell():
    items = _balanced_set()
    items.append(_item("core_plain_extra", "core", "plain", "accepting"))  # 2/1, n=3
    assert not any("core_plain: stance imbalance" in p for p in validate_items(items))


# --- calibration report / gate --------------------------------------------

def test_gate_passes_when_all_cells_near_half():
    items = _balanced_set()
    scores = {d["id"]: 0.52 for d in items}
    r = calibration_report(items, scores, tolerance=0.05)
    assert r["calibration_gate_passed"] is True
    for cell in CELLS:
        assert r["per_cell"][cell]["within_tolerance"] is True
        assert r["per_cell"][cell]["mean_p_stoic"] == pytest.approx(0.52)
    assert r["overall_mean_p_stoic"] == pytest.approx(0.52)
    assert r["outlier_items"] == []


def test_gate_fails_on_one_hot_cell():
    """The v1 failure shape: one cell far above 0.5 sinks the gate."""
    items = _balanced_set()
    scores = {d["id"]: (0.88 if cell_of(d) == "core_idiom" else 0.50) for d in items}
    r = calibration_report(items, scores, tolerance=0.05)
    assert r["calibration_gate_passed"] is False
    assert r["per_cell"]["core_idiom"]["within_tolerance"] is False
    assert r["per_cell"]["core_plain"]["within_tolerance"] is True


def test_outliers_flagged_and_sorted():
    items = _balanced_set()
    scores = {d["id"]: 0.5 for d in items}
    scores[items[0]["id"]] = 0.95
    scores[items[1]["id"]] = 0.05
    r = calibration_report(items, scores)
    flagged = [o["id"] for o in r["outlier_items"]]
    assert flagged == [items[1]["id"], items[0]["id"]]  # sorted by P ascending
    # a balanced pair of outliers can still average ~0.5 — the gate alone
    # wouldn't catch them, which is why they're flagged item-level
    assert r["per_cell"]["core_plain"]["within_tolerance"] is True


def test_missing_cell_fails_gate_without_crashing():
    items = [d for d in _balanced_set() if cell_of(d) != "offtopic_plain"]
    scores = {d["id"]: 0.5 for d in items}
    r = calibration_report(items, scores)
    assert r["all_cells_present"] is False
    assert r["calibration_gate_passed"] is False
    assert r["per_cell"]["offtopic_plain"]["n"] == 0


def test_per_item_scores_sorted_within_cell():
    items = _balanced_set()
    scores = {d["id"]: 0.4 + 0.05 * i for i, d in enumerate(items)}
    r = calibration_report(items, scores)
    for cell in CELLS:
        vals = list(r["per_cell"][cell]["per_item"].values())
        assert vals == sorted(vals)
