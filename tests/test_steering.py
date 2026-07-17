"""Hook hygiene — the one piece of real state in the pipeline.

The design rule: a leaked steering hook must be structurally impossible.
These tests pin it on a tiny stand-in with the same module path Llama has
(model.model.layers[L].mlp), no 3B download needed.
"""

import pytest
import torch
from torch import nn

from stoic.steering import steering, _mlp


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Linear(d, d)


class _Inner(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = nn.ModuleList([_Block(d) for _ in range(n)])


class TinyModel(nn.Module):
    """Minimal model exposing Llama's `model.layers[L].mlp` path."""

    def __init__(self, d=8, n=3):
        super().__init__()
        self.model = _Inner(d, n)

    @property
    def device(self):
        return next(self.parameters()).device


def _n_hooks(model):
    return sum(len(b.mlp._forward_hooks) for b in model.model.layers)


def test_hook_registered_only_during_context():
    m = TinyModel()
    vec = torch.ones(8)
    assert _n_hooks(m) == 0
    with steering(m, 1, vec, 0.5):
        assert len(m.model.layers[1].mlp._forward_hooks) == 1
        assert len(m.model.layers[0].mlp._forward_hooks) == 0  # only layer L
    assert _n_hooks(m) == 0


def test_hook_removed_on_exception():
    m = TinyModel()
    vec = torch.ones(8)
    with pytest.raises(RuntimeError):
        with steering(m, 0, vec, 0.5):
            raise RuntimeError("boom mid-steering")
    assert _n_hooks(m) == 0


def test_steering_adds_coeff_times_vector():
    torch.manual_seed(0)
    m = TinyModel()
    vec = torch.ones(8)
    x = torch.randn(1, 8)
    base = _mlp(m, 2)(x)
    with steering(m, 2, vec, 0.25):
        steered = _mlp(m, 2)(x)
    assert torch.allclose(steered, base + 0.25 * vec)
    # and cleanly reverts
    assert torch.allclose(_mlp(m, 2)(x), base)


def test_steering_casts_vector_to_model_dtype():
    m = TinyModel().to(torch.float64)
    vec = torch.ones(8, dtype=torch.float32)
    x = torch.randn(1, 8, dtype=torch.float64)
    with steering(m, 0, vec, 1.0):
        out = _mlp(m, 0)(x)
    assert out.dtype == torch.float64
