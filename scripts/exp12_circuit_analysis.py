"""Exp 12 — circuit-topology comparison on the CLEAN adapters (new work, $0).

Runs BOTH interventions through ModelLens `circuit_discovery` (not the
hand-rolled bridge scripts) with a content-relevant metric, on matched data:

- Metric: logit(" A") − logit(" B") at the last position of a forced-choice
  dilemma prompt. Clean input has the STOIC option as A; corrupted input is
  the SAME dilemma with the options swapped. The only difference between the
  inputs is *which option is the Stoic one*, so components whose patching
  moves this metric are the ones carrying Stoic-content recognition — the
  project's decision-level instrument turned into a patching metric. (This is
  the "logit difference between Stoic and neutral tokens" variant; the
  default max-logit metric is never used.)
- Item selection: among dilemmas whose two label orders tokenize to the SAME
  length (a hard patching requirement), pick the one with the largest content
  signal c = (m_clean − m_corrupted)/2 on the base model.
- Conditions per author: BASE (control, computed once — author-independent),
  CAA (frozen clean vector at the canonical layer/coeff, active via the
  steering() context manager during every patching forward), LoRA (the
  Stage-4-verified lora_{author}_clean adapter merged onto a fresh base).

Usage (analysis needs the stoic-llm env; plotting needs matplotlib):
    python scripts/exp12_circuit_analysis.py analyze
    python scripts/exp12_circuit_analysis.py plot results/exp12_circuits/exp12_<ts>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELLENS_PATH = (
    "/Users/sebastianvillalobos/Downloads/DSAN/Spring 2026/"
    "Neural Nets - 6600/Final Project - Seb Version/modellens"
)
sys.path.insert(0, str(REPO))
sys.path.insert(0, MODELLENS_PATH)

OUT_DIR = REPO / "results" / "exp12_circuits"
IMPORTANCE_THRESHOLD = 0.15


# --- ModelLens hotfix (upstream bug, patched locally; do not edit ModelLens) --
def _install_capture_hotfix():
    """modellens.analysis.activation_patching._capture_activations has an
    indentation bug: `return hook_fn` sits inside hook_fn instead of
    make_hook, so make_hook(name) returns None and the capture pass crashes
    with "'NoneType' object is not callable". Replace it with a fixed copy."""
    import torch
    from modellens.analysis import activation_patching as ap

    def _fixed_capture_activations(model, available, inputs, layer_names, **kwargs):
        activations = {}
        with ap._hook_context() as hooks:
            for name in layer_names:

                def make_hook(n):
                    def hook_fn(module, module_in, module_out):
                        if isinstance(module_out, tuple):
                            activations[n] = module_out[0].detach().clone()
                        else:
                            activations[n] = module_out.detach().clone()

                    return hook_fn  # returned from make_hook, NOT from hook_fn

                hooks.append(available[name].register_forward_hook(make_hook(name)))
            with torch.no_grad():
                output = ap._forward(model, inputs, **kwargs)
        return activations, output

    ap._capture_activations = _fixed_capture_activations


# --- analysis ---------------------------------------------------------------
def pick_dilemma(model, tokenizer, dilemmas, tok_a, tok_b):
    """Pick the equal-length dilemma with the largest base-model content
    signal c = (metric_clean − metric_corrupted)/2."""
    import torch

    from stoic.dilemmas import PROMPT_TEMPLATE

    def prompts_for(d):
        clean = PROMPT_TEMPLATE.format(
            situation=d["situation"], option_a=d["stoic"], option_b=d["nonstoic"]
        )
        corrupted = PROMPT_TEMPLATE.format(
            situation=d["situation"], option_a=d["nonstoic"], option_b=d["stoic"]
        )
        return clean, corrupted

    @torch.no_grad()
    def logit_diff(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        logits = model(**inputs).logits[0, -1]
        return (logits[tok_a] - logits[tok_b]).float().item()

    best = None
    print("Selecting dilemma (equal-length orders, max |content signal| on base):")
    for d in dilemmas:
        clean, corrupted = prompts_for(d)
        n1 = len(tokenizer(clean)["input_ids"])
        n2 = len(tokenizer(corrupted)["input_ids"])
        if n1 != n2:
            continue
        c = 0.5 * (logit_diff(clean) - logit_diff(corrupted))
        print(f"  {d['id']:>12s}  len={n1:>3d}  c={c:+.3f}")
        if best is None or abs(c) > abs(best[1]):
            best = (d, c)
    d, c = best
    clean, corrupted = prompts_for(d)
    print(f"-> chose {d['id']} (content signal c={c:+.3f})")
    return d, clean, corrupted


def sanitize(obj):
    """Make a circuit dict JSON-safe: tensors become shape strings."""
    import torch

    if isinstance(obj, torch.Tensor):
        return f"<tensor {tuple(obj.shape)}>"
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def run_analysis():
    import torch

    _install_capture_hotfix()
    from modellens import ModelLens
    from modellens.analysis.circuit_discovery import discover_circuit, summarize_circuit

    from stoic import config
    from stoic.dilemmas import _single_token_id, load_dilemmas
    from stoic.lora import merge_adapter
    from stoic.model import load_model
    from stoic.steering import load_reference_vector, steering

    t0 = time.time()
    model, tokenizer = load_model()
    tok_a = _single_token_id(tokenizer, " A")
    tok_b = _single_token_id(tokenizer, " B")

    def metric_fn(output):
        logits = output.logits if hasattr(output, "logits") else output
        return (logits[0, -1, tok_a] - logits[0, -1, tok_b]).float().item()

    dilemma, clean_prompt, corrupted_prompt = pick_dilemma(
        model, tokenizer, load_dilemmas(), tok_a, tok_b
    )
    clean_inputs = tokenizer(clean_prompt, return_tensors="pt")
    corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt")

    lens = ModelLens(model)
    lens.adapter.set_tokenizer(tokenizer)
    patchable = lens.adapter.get_patchable_layers()
    print(f"\nPatchable sublayers: {len(patchable)}  (threshold {IMPORTANCE_THRESHOLD})")

    def circuit(active_lens, label):
        print(f"\n=== circuit: {label} ===")
        t = time.time()
        c = discover_circuit(
            active_lens,
            clean_inputs,
            corrupted_inputs,
            metric_fn=metric_fn,
            importance_threshold=IMPORTANCE_THRESHOLD,
        )
        print(f"  clean {c['clean_metric']:+.3f}  corrupted {c['corrupted_metric']:+.3f}  "
              f"total {c['total_effect']:+.3f}  nodes {c.get('num_components', 0)}  "
              f"edges {c.get('num_connections', 0)}  [{time.time() - t:.0f}s]")
        print(summarize_circuit(c))
        return sanitize(c)

    results = {
        "experiment": 12,
        "description": "Circuit-topology comparison, CAA vs LoRA (clean adapters), via ModelLens discover_circuit",
        "metric": "logit(' A') - logit(' B') at last position; clean = stoic option as A, corrupted = options swapped",
        "dilemma": {"id": dilemma["id"], "concept": dilemma["concept"],
                    "stoic_stance": dilemma["stoic_stance"],
                    "clean_prompt": clean_prompt, "corrupted_prompt": corrupted_prompt},
        "importance_threshold": IMPORTANCE_THRESHOLD,
        "patchable_sublayers": len(patchable),
        "note": "New work (Exp 12) — no frozen reference. Base circuit is the shared control.",
        "modellens_hotfix": "_capture_activations closure-return bug patched locally (upstream fix pending)",
        "circuits": {},
    }

    # Base control (author-independent) — computed once.
    results["circuits"]["base"] = circuit(lens, "BASE (control)")

    for name, author in config.AUTHORS.items():
        # CAA: frozen clean vector, canonical layer/coeff, hook active for
        # every forward inside discover_circuit (capture + all patches).
        vec = load_reference_vector(author.vector_file, author.layer)
        with steering(model, author.layer, vec, author.coeff):
            results["circuits"][f"caa_{name}"] = circuit(
                lens, f"CAA {name} (L{author.layer}, c={author.coeff})"
            )

        # LoRA: Stage-4-verified clean adapter merged onto a FRESH base.
        merged = merge_adapter(author.adapter_dir)
        lens_merged = ModelLens(merged)
        lens_merged.adapter.set_tokenizer(tokenizer)
        try:
            results["circuits"][f"lora_{name}"] = circuit(
                lens_merged, f"LoRA {name} ({author.adapter_dir.name})"
            )
        finally:
            del lens_merged, merged
            gc.collect()

    results["runtime_sec"] = round(time.time() - t0, 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"exp12_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n↳ wrote {out.relative_to(REPO)}  ({results['runtime_sec']}s total)")
    return out


# --- plotting (needs matplotlib; run in an env that has it) ------------------
ROLE_COLORS = {"critical": "#d62728", "booster": "#2ca02c",
               "gate": "#9467bd", "processor": "#7f7f7f"}
FAMILY_Y = {"attention": 1.0, "mlp": 0.0}


def _draw_circuit(ax, circ, title, n_blocks=28):
    nodes = circ.get("nodes", [])
    edges = circ.get("edges", [])
    pos = {}
    for n in nodes:
        x = n["block_num"] if n["block_num"] is not None else -1
        y = FAMILY_Y.get(n["family"], 0.5)
        pos[n["name"]] = (x, y)
    for e in edges:
        if e["from"] in pos and e["to"] in pos:
            (x1, y1), (x2, y2) = pos[e["from"]], pos[e["to"]]
            style = "-" if e["type"] == "sequential" else "--"
            ax.plot([x1, x2], [y1, y2], style, color="#bbbbbb",
                    lw=0.8 + 2.0 * min(abs(e.get("weight", 0.3)), 1.0), zorder=1)
    for n in nodes:
        x, y = pos[n["name"]]
        size = 60 + 900 * min(abs(n["normalized_effect"]), 1.0)
        ax.scatter([x], [y], s=size, c=ROLE_COLORS.get(n["role"], "#7f7f7f"),
                   edgecolors="black", linewidths=0.6, zorder=2)
    ax.set_title(f"{title}\nclean {circ.get('clean_metric', 0):+.2f} / "
                 f"corr {circ.get('corrupted_metric', 0):+.2f} / "
                 f"{len(nodes)} nodes", fontsize=9)
    ax.set_xlim(-1.5, n_blocks + 0.5)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["mlp", "attn"], fontsize=8)
    ax.set_xlabel("block", fontsize=8)
    ax.tick_params(labelsize=7)


def run_plot(json_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with open(json_path) as f:
        results = json.load(f)
    circuits = results["circuits"]
    authors = sorted({k.split("_", 1)[1] for k in circuits if k.startswith("caa_")})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for a in authors:
        fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), sharey=True)
        _draw_circuit(axes[0], circuits["base"], "BASE (control)")
        _draw_circuit(axes[1], circuits[f"caa_{a}"], f"CAA — {a}")
        _draw_circuit(axes[2], circuits[f"lora_{a}"], f"LoRA — {a} (clean adapter)")
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markeredgecolor="black", markersize=8, label=r)
                   for r, c in ROLE_COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(
            f"Exp 12 — Stoic-content circuit ({results['dilemma']['id']}): "
            f"node size = |normalized patch effect| (threshold "
            f"{results['importance_threshold']})", fontsize=10)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out = OUT_DIR / f"exp12_circuit_{a}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"↳ wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("analyze")
    pp = sub.add_parser("plot")
    pp.add_argument("json_path")
    args = p.parse_args()
    if args.cmd == "analyze":
        run_analysis()
    else:
        run_plot(args.json_path)
