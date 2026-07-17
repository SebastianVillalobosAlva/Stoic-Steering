"""Canonical config invariants — the ground-truth numbers CLAUDE.md freezes."""

from stoic import config


def test_canonical_layers_and_coeff():
    assert config.AUTHORS["marcus"].layer == 26
    assert config.AUTHORS["seneca"].layer == 4
    assert config.AUTHORS["epictetus"].layer == 8
    assert all(a.coeff == 0.11 for a in config.AUTHORS.values())


def test_dilemma_baseline():
    assert config.DILEMMA_BASELINE == 0.542


def test_model_is_3b_float16():
    assert config.MODEL_NAME == "meta-llama/Llama-3.2-3B"
    assert "1B" not in config.MODEL_NAME
    import torch

    assert config.DTYPE == torch.float16


def test_write_targets_are_outside_reference():
    """The reference wall: every directory the pipeline writes to must live
    outside data/reference/."""
    write_dirs = [
        config.GENERATED_DIR,
        config.GEN_RAW_DIR,
        config.GEN_PROCESSED_DIR,
        config.GEN_CHUNKED_DIR,
        config.RESULTS_DIR,
        config.MODELS_DIR,
    ]
    for d in write_dirs:
        assert not d.is_relative_to(config.REFERENCE_DIR), d


def test_author_inputs_read_from_reference():
    for a in config.AUTHORS.values():
        assert a.pairs_file.is_relative_to(config.REFERENCE_DIR)
        assert a.vector_file.is_relative_to(config.REFERENCE_DIR)
        assert a.adapter_dir.is_relative_to(config.MODELS_DIR)


def test_results_dir_stays_under_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    d = config.results_dir("some_stage")
    assert d.is_relative_to(tmp_path / "results")
    assert d.exists()
