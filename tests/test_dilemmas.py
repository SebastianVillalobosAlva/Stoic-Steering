"""The forced-choice ruler's math: label-order debiasing and single-token labels."""

from types import SimpleNamespace

import pytest
import torch

from stoic.dilemmas import (
    deltas_by_stance,
    mean,
    p_stoic,
    _p_first_label,
    _single_token_id,
)

VOCAB = 100
TOK_A, TOK_B = 65, 66


class FakeTokenizer:
    """Single-token ' A'/' B'; records the last prompt for the fake model."""

    def __init__(self):
        self.last_prompt = None

    def encode(self, text, add_special_tokens=False):
        return {" A": [TOK_A], " B": [TOK_B]}.get(text, [1, 2])


    def __call__(self, prompt, return_tensors=None):
        self.last_prompt = prompt

        class _Batch(dict):
            def to(self, device):
                return self

        return _Batch(input_ids=torch.tensor([[1]]))


class PositionBiasModel:
    """Always favors label A by a fixed logit bias, regardless of content —
    the pure positional/label bias the both-orders average must cancel."""

    device = "cpu"

    def __init__(self, bias=2.0):
        self.bias = bias

    def __call__(self, **inputs):
        logits = torch.zeros(1, 1, VOCAB)
        logits[0, -1, TOK_A] = self.bias
        return SimpleNamespace(logits=logits)


class ContentModel:
    """Favors whichever label carries the marker text — a genuine preference."""

    device = "cpu"

    def __init__(self, tokenizer, marker, strength=2.0):
        self.tok = tokenizer
        self.marker = marker
        self.strength = strength

    def __call__(self, **inputs):
        logits = torch.zeros(1, 1, VOCAB)
        prompt = self.tok.last_prompt
        a_option = prompt.split("A) ")[1].split("\n")[0]
        tok = TOK_A if self.marker in a_option else TOK_B
        logits[0, -1, tok] = self.strength
        return SimpleNamespace(logits=logits)


DILEMMA = {
    "id": "t1",
    "situation": "Something happened.",
    "stoic": "ACCEPT_IT",
    "nonstoic": "RAGE_AT_IT",
    "stoic_stance": "accepting",
}


def test_single_token_id():
    tok = FakeTokenizer()
    assert _single_token_id(tok, " A") == TOK_A
    with pytest.raises(ValueError):
        _single_token_id(tok, "not a single token")


def test_p_first_label_is_normalized_softmax():
    tok = FakeTokenizer()
    p = _p_first_label(PositionBiasModel(bias=0.0), tok, "prompt", TOK_A, TOK_B)
    assert p == pytest.approx(0.5)
    p = _p_first_label(PositionBiasModel(bias=2.0), tok, "prompt", TOK_A, TOK_B)
    assert p == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())


def test_pure_label_bias_cancels_exactly():
    """A model that always prefers 'A' must score P(stoic) = 0.5 — the
    both-label-orders average is what makes the 0.542 baseline meaningful."""
    tok = FakeTokenizer()
    p = p_stoic(PositionBiasModel(bias=3.0), tok, DILEMMA, TOK_A, TOK_B)
    assert p == pytest.approx(0.5)


def test_genuine_preference_survives_debiasing():
    tok = FakeTokenizer()
    model = ContentModel(tok, marker="ACCEPT_IT", strength=2.0)
    p = p_stoic(model, tok, DILEMMA, TOK_A, TOK_B)
    expected = torch.sigmoid(torch.tensor(2.0)).item()  # favored in BOTH orders
    assert p == pytest.approx(expected)
    anti = ContentModel(tok, marker="RAGE_AT_IT", strength=2.0)
    assert p_stoic(anti, tok, DILEMMA, TOK_A, TOK_B) == pytest.approx(1 - expected)


def test_mean_and_stance_buckets():
    assert mean({"a": 0.0, "b": 1.0}) == pytest.approx(0.5)
    dilemmas = [
        {"id": "x", "stoic_stance": "accepting"},
        {"id": "y", "stoic_stance": "active"},
        {"id": "z", "stoic_stance": "accepting"},
    ]
    buckets = deltas_by_stance(dilemmas, {"x": 0.1, "y": -0.2, "z": 0.3})
    assert buckets["accepting"]["n"] == 2
    assert buckets["active"]["n"] == 1
    assert buckets["accepting"]["mean_delta"] == pytest.approx(0.2)
