"""CAA steering: extract a direction, inject it — at the SAME site.

Two functions and one context manager. The steering hook is the only real
state in the whole pipeline, so it lives inside a `with` block: on exit the
hook is always removed. A leaked hook (the old repo's recurring bug) is
structurally impossible here.

Injection/extraction site: the MLP output of decoder block L
(`model.model.layers[L].mlp`). Extracting and injecting at the same site is
what makes "layer L is where the Stoic direction lives" a real claim — the
old code extracted at the final layer and injected at L, so its layer sweep
was meaningless.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch


def _mlp(model, layer: int):
    return model.model.layers[layer].mlp


def load_pairs(pairs_file: str | Path) -> list[dict]:
    with open(pairs_file) as f:
        data = json.load(f)
    pairs = data["pairs"] if isinstance(data, dict) else data
    return pairs


@torch.no_grad()
def _mean_activation(model, tokenizer, text: str, layer: int) -> torch.Tensor:
    """Mean (over sequence) of the layer-L MLP output for `text` → (hidden_dim,)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(
        model.device
    )
    captured = {}

    def hook(_module, _inp, out):
        captured["h"] = out.detach()

    handle = _mlp(model, layer).register_forward_hook(hook)
    try:
        model(**inputs)
    finally:
        handle.remove()
    # out: (batch, seq, hidden) -> mean over seq -> (hidden,)
    return captured["h"].mean(dim=1).squeeze(0)


def extract_vector(model, tokenizer, pairs: list[dict], layer: int) -> torch.Tensor:
    """CAA steering vector at `layer`: mean over pairs of (stoic − neutral).

    `layer` is required and is the same site the vector is injected at (see
    `steering()`), so extraction and injection can never drift apart.
    """
    total = None
    n = len(pairs)
    for i, pair in enumerate(pairs, 1):
        print(f"  extract L{layer}: pair {i}/{n}", end="\r")
        stoic = _mean_activation(model, tokenizer, pair["stoic_text"], layer)
        neutral = _mean_activation(model, tokenizer, pair["neutral_text"], layer)
        diff = stoic - neutral
        total = diff if total is None else total + diff
    print()
    return total / n


@contextmanager
def steering(model, layer: int, vector: torch.Tensor, coeff: float):
    """Add `coeff * vector` to the layer-L MLP output for the duration of the block.

    The hook is registered on enter and removed on exit no matter what.
    """
    vec = vector.to(dtype=next(model.parameters()).dtype, device=model.device)

    def hook(_module, _inp, out):
        return out + coeff * vec

    handle = _mlp(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def load_reference_vector(vector_file: str | Path, layer: int) -> torch.Tensor:
    """Load a frozen `{layer: tensor}` steering-vector dict and pick one layer."""
    loaded = torch.load(vector_file, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"{vector_file} is not a {{layer: tensor}} dict (old format).")
    if layer not in loaded:
        raise KeyError(f"No vector for layer {layer}; available: {sorted(loaded)}")
    return loaded[layer]
