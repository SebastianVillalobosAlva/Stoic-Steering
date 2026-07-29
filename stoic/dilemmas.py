"""Forced-choice dilemma harness — the judge-free ruler.

Measures whether an intervention shifts *decisions*, not just style. Each
dilemma is a situation with two options labelled A/B; we take ONE forward pass
and compare next-token probability mass on the two label tokens. Every item is
run in BOTH label orders (target=A, then target=B) and averaged, which cancels
positional/label bias exactly.

    P(target) = softmax over {logit(' A'), logit(' B')}, averaged over both orders

No generation, no judge — pure logits. The unsteered baseline mean over the v2
set is the load-bearing checkpoint: 0.542.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from stoic.axis import ACTIVE
from stoic.steering import steering

PROMPT_TEMPLATE = (
    "Consider the following situation and choose the better course of action.\n\n"
    "Situation: {situation}\n\n"
    "A) {option_a}\n"
    "B) {option_b}\n\n"
    "The better course of action is option"
)


def load_dilemmas(path: str | Path | None = None) -> list[dict]:
    """Load the axis's forced-choice items.

    Items are returned EXACTLY as stored — no key renaming, no normalization.
    Which keys mean what is resolved at read time through `DilemmaFields`
    (see `p_target`), so callers that read raw item keys directly — notably
    `scripts/exp12_*.py`, which does `d["stoic"]` — keep working unchanged.
    """
    path = path if path is not None else ACTIVE.dilemmas_file
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return payload[ACTIVE.dilemmas_collection_key]


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Label {text!r} is not a single token ({ids}).")
    return ids[0]


@torch.no_grad()
def _p_first_label(model, tokenizer, prompt: str, tok_a: int, tok_b: int) -> float:
    """P(label 'A') normalized over {A, B} from one forward pass."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1]
    two = torch.stack([logits[tok_a], logits[tok_b]]).float()
    return torch.softmax(two, dim=0)[0].item()


def p_target(model, tokenizer, dilemma: dict, tok_a: int, tok_b: int, *, fields=None) -> float:
    """Order-debiased P(target option): mean over both label orders.

    `fields` names which item keys hold the situation and the two options; it
    defaults to the active axis (stoic: situation/stoic/nonstoic). The item
    itself is never modified.
    """
    f = fields or ACTIVE.fields
    p1 = _p_first_label(
        model, tokenizer,
        PROMPT_TEMPLATE.format(
            situation=dilemma[f.situation],
            option_a=dilemma[f.target],
            option_b=dilemma[f.foil],
        ),
        tok_a, tok_b,
    )  # target is A -> want P(A)
    p2 = _p_first_label(
        model, tokenizer,
        PROMPT_TEMPLATE.format(
            situation=dilemma[f.situation],
            option_a=dilemma[f.foil],
            option_b=dilemma[f.target],
        ),
        tok_a, tok_b,
    )  # target is B -> want P(B) = 1 - P(A)
    return 0.5 * (p1 + (1.0 - p2))


def eval_dilemmas(
    model,
    tokenizer,
    dilemmas: list[dict],
    *,
    steer: tuple[int, torch.Tensor, float] | None = None,
    fields=None,
) -> dict[str, float]:
    """P(target) for every dilemma. `steer=(layer, vector, coeff)` injects a CAA
    vector for the duration; `steer=None` is the unsteered baseline."""
    f = fields or ACTIVE.fields
    label_a, label_b = ACTIVE.label_tokens
    tok_a = _single_token_id(tokenizer, label_a)
    tok_b = _single_token_id(tokenizer, label_b)
    ctx = steering(model, *steer) if steer is not None else nullcontext()
    with ctx:
        return {d[f.id]: p_target(model, tokenizer, d, tok_a, tok_b, fields=f)
                for d in dilemmas}


def mean(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def paired_stats(deltas: list[float]) -> dict:
    """Paired-sample summary (mean Δ, std, t vs 0). scipy optional for p-value."""
    n = len(deltas)
    m = sum(deltas) / n
    var = sum((d - m) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    se = std / math.sqrt(n) if n > 0 else float("nan")
    t = m / se if se > 0 else float("nan")
    out = {"n": n, "mean_delta": m, "std": std, "t_stat": t, "p_value": None}
    if n > 1:
        try:
            from scipy import stats as sps

            out["p_value"] = float(sps.ttest_rel([0.0] * n, [-d for d in deltas]).pvalue)
        except ImportError:
            pass
    return out


def sign_test(deltas: dict[str, float] | list[float], tol: float = 1e-9) -> dict:
    """Exact two-sided sign test on per-item deltas (ties dropped, standard
    practice). Asks only 'did most items move the same way' — ignores magnitude."""
    vals = list(deltas.values()) if isinstance(deltas, dict) else list(deltas)
    pos = sum(1 for d in vals if d > tol)
    neg = sum(1 for d in vals if d < -tol)
    ties = len(vals) - pos - neg
    n = pos + neg
    if n == 0:
        return {"pos": pos, "neg": neg, "ties": ties, "n": 0, "p_two_sided": 1.0}
    k = max(pos, neg)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n * 2
    return {"pos": pos, "neg": neg, "ties": ties, "n": n, "p_two_sided": min(p, 1.0)}


def deltas_by_stance(
    dilemmas: list[dict], deltas: dict[str, float], *, fields=None
) -> dict[str, dict]:
    f = fields or ACTIVE.fields
    buckets: dict[str, list[float]] = {}
    for d in dilemmas:
        buckets.setdefault(d[f.stance], []).append(deltas[d[f.id]])
    return {k: paired_stats(v) for k, v in buckets.items()}
