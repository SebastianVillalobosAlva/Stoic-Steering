"""The mismatched-decoding bug must stay unwritable: one canonical generate()."""

import torch

from stoic.config import GEN_KWARGS
from stoic.model import generate


def test_gen_kwargs_are_the_canonical_set():
    assert GEN_KWARGS == {
        "max_new_tokens": 100,
        "do_sample": False,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
    }
    # no sampling knobs may sneak in
    assert "temperature" not in GEN_KWARGS and "top_p" not in GEN_KWARGS


class _Batch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __call__(self, prompt, return_tensors=None):
        return _Batch(input_ids=torch.tensor([[1, 2]]))

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.captured = None

    def generate(self, **kwargs):
        self.captured = kwargs
        return torch.tensor([[1, 2, 3]])


def test_generate_sends_exactly_the_canonical_kwargs():
    m = _FakeModel()
    out = generate(m, _FakeTokenizer(), "hello")
    assert out == "decoded"
    sent = {k: v for k, v in m.captured.items() if k != "input_ids"}
    assert sent == GEN_KWARGS


def test_generate_override_changes_only_what_is_passed():
    m = _FakeModel()
    generate(m, _FakeTokenizer(), "hello", max_new_tokens=7)
    assert m.captured["max_new_tokens"] == 7
    assert m.captured["do_sample"] is False
    assert m.captured["repetition_penalty"] == 1.3
    assert m.captured["no_repeat_ngram_size"] == 3
