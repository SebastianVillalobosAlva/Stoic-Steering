"""Model loading and the ONE canonical generate().

Decoding lives here and nowhere else. Because every caller routes through
`generate()`, the old "steered run used different sampling than the baseline"
bug is unwritable — there is only one decoding config (`GEN_KWARGS`).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stoic.config import DEVICE, DTYPE, GEN_KWARGS, MODEL_NAME


def load_model(model_name: str = MODEL_NAME, dtype: torch.dtype = DTYPE, device: str = DEVICE):
    """Load base model + tokenizer. Tokenizer always comes from the BASE model,
    never from an adapter folder (those ship a redundant tokenizer.json)."""
    print(f"Loading {model_name} ({dtype}, {device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    print("✓ Model loaded")
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt: str, **overrides) -> str:
    """Deterministic greedy generation with the canonical decoding config.

    Pass overrides only to change *length* etc. for a specific call; the
    sampling behaviour (do_sample=False, repetition controls) stays fixed so
    outputs are reproducible.
    """
    kwargs = {**GEN_KWARGS, **overrides}
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
