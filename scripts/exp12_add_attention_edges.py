"""Exp 12 add-on: attention-routing edges via eager attention.

The main Exp 12 run loads the model with sdpa attention, which cannot
materialize attention weights — so `discover_circuit` fell back to
sequential-only edges. This script recovers the attention-routing edges
WITHOUT re-running the (expensive) activation patching: attention maps need
exactly one forward per condition on a model reloaded with
`attn_implementation="eager"`. Edges are rebuilt from the saved nodes with
ModelLens's own `_build_attention_edges` and merged into the JSON in place.

Usage:
    python scripts/exp12_add_attention_edges.py results/exp12_circuits/exp12_<ts>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(
    0,
    "/Users/sebastianvillalobos/Downloads/DSAN/Spring 2026/"
    "Neural Nets - 6600/Final Project - Seb Version/modellens",
)


def main(json_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from modellens import ModelLens
    from modellens.analysis.attention import run_attention_analysis
    from modellens.analysis.circuit_discovery import _build_attention_edges

    from stoic import config
    from stoic.lora import merge_adapter
    from stoic.steering import load_reference_vector, steering

    path = Path(json_path).resolve()
    with open(path) as f:
        results = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    clean_inputs = tokenizer(results["dilemma"]["clean_prompt"], return_tensors="pt")

    print(f"Loading base with eager attention ({config.MODEL_NAME}) ...")
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=config.DTYPE,
        device_map=config.DEVICE,
        attn_implementation="eager",
    )
    base.eval()
    lens = ModelLens(base)
    lens.adapter.set_tokenizer(tokenizer)

    def add_edges(condition: str, active_lens) -> None:
        circ = results["circuits"][condition]
        nodes = circ.get("nodes", [])
        if not nodes:
            print(f"[{condition}] no nodes above threshold — skipped")
            return
        attn = run_attention_analysis(active_lens, clean_inputs)
        new_edges = _build_attention_edges(nodes, attn)
        existing = {(e["from"], e["to"]) for e in circ.get("edges", [])}
        added = [e for e in new_edges if (e["from"], e["to"]) not in existing]
        circ["edges"] = circ.get("edges", []) + added
        circ["num_connections"] = len(circ["edges"])
        circ["attention_results"] = {
            "backend": "eager",
            "num_layers": attn.get("num_layers"),
            "note": "weights not stored; attention_routing edges merged below",
        }
        print(f"[{condition}] +{len(added)} attention_routing edges "
              f"(total {len(circ['edges'])})")

    add_edges("base", lens)

    for name, author in config.AUTHORS.items():
        vec = load_reference_vector(author.vector_file, author.layer)
        with steering(base, author.layer, vec, author.coeff):
            add_edges(f"caa_{name}", lens)

        merged = merge_adapter(author.adapter_dir, attn_implementation="eager")
        lens_m = ModelLens(merged)
        lens_m.adapter.set_tokenizer(tokenizer)
        try:
            add_edges(f"lora_{name}", lens_m)
        finally:
            del lens_m, merged
            gc.collect()

    results["attention_edges_added"] = time.strftime("%Y%m%d_%H%M%S")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"↳ updated {path.relative_to(REPO)} in place")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("json_path")
    main(p.parse_args().json_path)
