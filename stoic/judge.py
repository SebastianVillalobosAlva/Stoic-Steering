"""LLM-as-judge content scoring + seed eval (Stage 3, Exp 9).

Measures the *content* effect of CAA steering: does steering toward a
philosopher make the model's prose reason in a more Stoic way, as scored by an
independent judge? The canonical judge is Gemini (gemini-2.5-flash).

`content = (Δphilosophical_depth + Δstoic_alignment) / 2`, averaged over the
prompt set. The judge is nondeterministic, so we run n_seeds judgements of the
same (greedy) generations and report mean ± std — the effect "survives" if
±1σ excludes zero. Reference (Exp 9, clean pairs, coeff 0.11, n_seeds=5):
Marcus L26 +0.408, Seneca L4 +0.583, Epictetus L8 +0.767.
"""

from __future__ import annotations

import json
import re
import statistics
import time

import torch

from stoic.config import DEFAULT_PROMPTS, GEN_KWARGS
from stoic.model import generate
from stoic.steering import steering

JUDGE_MODEL = "gemini-2.5-flash"

DIMENSIONS = [
    "philosophical_depth",
    "stoic_alignment",
    "coherence",
    "stylistic_authenticity",
]

STOIC_RUBRIC = """
Score the following text on how well it reflects Stoic philosophical principles.
Evaluate on these 4 dimensions, each scored 1-5:

1. PHILOSOPHICAL DEPTH (1-5)
   1 = No philosophical content, generic or off-topic
   3 = Some philosophical ideas but surface-level
   5 = Deep engagement with Stoic concepts (virtue, reason, nature, acceptance)

2. STOIC ALIGNMENT (1-5)
   1 = Contradicts Stoic principles or is philosophically neutral
   3 = Loosely aligned with Stoic ideas
   5 = Clearly reflects core Stoic doctrines (dichotomy of control, virtue as
       sole good, living according to nature, rational acceptance)

3. COHERENCE (1-5)
   1 = Incoherent, repetitive, or nonsensical
   3 = Readable but disorganized or partially repetitive
   5 = Clear, well-structured, logically flowing

4. STYLISTIC AUTHENTICITY (1-5)
   1 = Modern casual language, no philosophical register
   3 = Some philosophical tone but inconsistent
   5 = Reads like translated ancient philosophical text (aphoristic,
       contemplative, uses philosophical vocabulary naturally)

Respond ONLY with a JSON object in this exact format, no other text:
{"philosophical_depth": X, "stoic_alignment": X, "coherence": X, "stylistic_authenticity": X, "reasoning": "brief explanation"}
""".strip()


def make_gemini_client(api_key: str, model: str = JUDGE_MODEL):
    """Return (client, model_name) for the Gemini judge."""
    from google import genai

    return genai.Client(api_key=api_key), model


def _extract_json(text: str) -> dict:
    """Pull a JSON score object out of a judge response (fences/prose tolerant).
    Salvages the 4 integer dimensions by regex even when the trailing
    `reasoning` string is truncated (the judge sometimes runs out of output
    tokens mid-string). Only returns zeros if not even the numbers survive."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Salvage: pull each numeric dimension directly (survives truncated reasoning).
    salvaged = {}
    for d in DIMENSIONS:
        m = re.search(rf'"{d}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if m:
            salvaged[d] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    if len(salvaged) == len(DIMENSIONS):
        salvaged["reasoning"] = "salvaged (truncated response)"
        return salvaged
    print(f"⚠ JSON PARSE FAILED — scoring as zeros. Raw:\n{text[:300]}\n")
    return {d: 0 for d in DIMENSIONS} | {"reasoning": f"parse fail: {text[:200]}"}


def score(client, model: str, text: str, prompt: str = "", max_retries: int = 5) -> dict:
    """Judge one text against the Stoic rubric → dim scores + aggregate.

    Retries transient API failures (503 overload, 429 rate limit, timeouts) with
    exponential backoff so a momentary spike doesn't kill a long run."""
    msg = f"{STOIC_RUBRIC}\n\n"
    if prompt:
        msg += f"PROMPT: {prompt}\n\n"
    msg += f"TEXT TO EVALUATE:\n{text}"

    resp = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=msg,
                config={"response_mime_type": "application/json"},
            )
            break
        except Exception as e:  # transient server/rate errors → back off and retry
            transient = any(
                s in str(e) for s in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "timeout", "500")
            )
            if not transient or attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt * 5, 60)
            print(f"  ⚠ judge API transient error (attempt {attempt+1}/{max_retries}); "
                  f"retrying in {wait}s: {str(e)[:80]}")
            time.sleep(wait)

    try:
        scores = json.loads(resp.text)
    except json.JSONDecodeError:
        scores = _extract_json(resp.text)

    scores["aggregate"] = sum(scores.get(d, 0) for d in DIMENSIONS) / len(DIMENSIONS)
    return scores


def content_score(deltas: dict) -> float:
    """Stoic content delta = (Δphilosophical_depth + Δstoic_alignment) / 2."""
    return (deltas["philosophical_depth"] + deltas["stoic_alignment"]) / 2


def evaluate_steering(
    client, model: str, prompts, steered_outputs, unsteered_outputs, delay: float = 0.5
) -> dict:
    """Judge steered vs unsteered across prompts; return avg deltas + content."""
    per_dim = {d: [] for d in DIMENSIONS + ["aggregate"]}
    steered_dim = {d: [] for d in DIMENSIONS + ["aggregate"]}
    for i, (p, s, u) in enumerate(zip(prompts, steered_outputs, unsteered_outputs), 1):
        us = score(client, model, u, p)
        time.sleep(delay)
        ss = score(client, model, s, p)
        if i < len(prompts):
            time.sleep(delay)
        for d in per_dim:
            per_dim[d].append(ss.get(d, 0) - us.get(d, 0))
            steered_dim[d].append(ss.get(d, 0))
    avg_deltas = {d: sum(v) / len(v) for d, v in per_dim.items()}
    avg_steered = {d: sum(v) / len(v) for d, v in steered_dim.items()}
    return {
        "avg_deltas": avg_deltas,
        "avg_steered": avg_steered,
        "content": content_score(avg_deltas),
    }


@torch.no_grad()
def _generate_all(model, tokenizer, prompts, steer=None) -> list[str]:
    """Greedy generation for every prompt; `steer=(layer, vector, coeff)` steers."""
    from contextlib import nullcontext

    ctx = steering(model, *steer) if steer is not None else nullcontext()
    with ctx:
        return [generate(model, tokenizer, p) for p in prompts]


@torch.no_grad()
def generate_all_sampled(
    model, tokenizer, prompts, seed: int, steer=None,
    temperature: float = 0.6, top_p: float = 0.9, max_new_tokens: int = 100,
) -> list[str]:
    """Sampled generation for every prompt under a fixed seed. `steer` steers.
    Baseline and steered call this with the SAME seed → paired comparison."""
    from contextlib import nullcontext

    torch.manual_seed(seed)
    ctx = steering(model, *steer) if steer is not None else nullcontext()
    with ctx:
        return [
            generate(
                model, tokenizer, p,
                do_sample=True, temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            for p in prompts
        ]


def seed_eval_sampled(
    model,
    tokenizer,
    client,
    judge_model: str,
    *,
    layer: int,
    vector: torch.Tensor,
    coeff: float,
    author: str,
    baselines_by_seed: dict[int, list[str]],
    prompts=DEFAULT_PROMPTS,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_new_tokens: int = 100,
) -> dict:
    """vary='generation' seed eval: BOTH baseline and steered sampled at the same
    seed/temperature/length, judged once each. Measures whether steering moves the
    output distribution (fair matched-decoding positive test). Baselines are
    unsteered so they are computed once and shared across authors."""
    print(f"\n=== sampled seed eval: {author} L{layer} c{coeff} "
          f"× {len(baselines_by_seed)} seeds (temp {temperature}, {max_new_tokens} tok) ===")
    contents, aggregates = [], []
    last_steered = None
    for s in sorted(baselines_by_seed):
        steered = generate_all_sampled(
            model, tokenizer, prompts, s, steer=(layer, vector, coeff),
            temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
        )
        last_steered = steered
        er = evaluate_steering(client, judge_model, prompts, steered, baselines_by_seed[s])
        contents.append(er["content"])
        aggregates.append(er["avg_steered"]["aggregate"])
        print(f"  seed {s}: content={er['content']:+.3f}  agg={er['avg_steered']['aggregate']:.3f}")

    c_mean = statistics.mean(contents)
    c_std = statistics.stdev(contents) if len(contents) > 1 else 0.0
    survives = (c_mean - c_std) > 0 or (c_mean + c_std) < 0
    print(f"  content = {c_mean:+.3f} ± {c_std:.3f}  "
          f"({'SURVIVES ±1σ' if survives else 'not distinguishable from 0'})")
    return {
        "author": author,
        "layer": layer,
        "coefficient": coeff,
        "n_seeds": len(baselines_by_seed),
        "vary": "generation",
        "decoding": {"do_sample": True, "temperature": temperature,
                     "top_p": top_p, "max_new_tokens": max_new_tokens},
        "content_scores": contents,
        "aggregate_scores": aggregates,
        "content_mean": c_mean,
        "content_std": c_std,
        "content_ci_excludes_zero": survives,
        "steered_outputs_last_seed": last_steered,
    }


def seed_eval(
    model,
    tokenizer,
    client,
    judge_model: str,
    *,
    layer: int,
    vector: torch.Tensor,
    coeff: float,
    author: str,
    prompts=DEFAULT_PROMPTS,
    n_seeds: int = 5,
) -> dict:
    """vary='judge' seed eval: generate ONCE (greedy) steered + baseline, then
    judge n_seeds times to isolate judge variance. Returns content mean ± std."""
    print(f"\n=== seed eval: {author} L{layer} c{coeff} × {n_seeds} judge seeds ===")
    baseline = _generate_all(model, tokenizer, prompts)  # unsteered, greedy
    steered = _generate_all(model, tokenizer, prompts, steer=(layer, vector, coeff))

    contents, aggregates = [], []
    for s in range(n_seeds):
        er = evaluate_steering(client, judge_model, prompts, steered, baseline)
        contents.append(er["content"])
        aggregates.append(er["avg_steered"]["aggregate"])
        print(f"  seed {s}: content={er['content']:+.3f}  agg={er['avg_steered']['aggregate']:.3f}")

    c_mean = statistics.mean(contents)
    c_std = statistics.stdev(contents) if len(contents) > 1 else 0.0
    survives = (c_mean - c_std) > 0 or (c_mean + c_std) < 0
    print(f"  content = {c_mean:+.3f} ± {c_std:.3f}  "
          f"({'SURVIVES ±1σ' if survives else 'not distinguishable from 0'})")
    return {
        "author": author,
        "layer": layer,
        "coefficient": coeff,
        "n_seeds": n_seeds,
        "vary": "judge",
        "content_scores": contents,
        "aggregate_scores": aggregates,
        "content_mean": c_mean,
        "content_std": c_std,
        "content_ci_excludes_zero": survives,
        "baseline_outputs": baseline,
        "steered_outputs": steered,
    }
