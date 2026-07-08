"""Exp 12c — stance-balanced circuit sweep (8 items × base/Seneca/Marcus).

Reuses the EXACT anchor harness (same discover_circuit, same content
metric_fn, same threshold) via import from exp12_circuit_analysis. Primary
outcome: signed Δc = c_adapter − c_base per item (Δc > 0 = pushed toward the
Stoic option). The |sensitivity|-% is reported as a LEGACY, CONFOUNDED
secondary only (base metric is stance-dependent in sign and magnitude —
see exp12_sweep_precheck_*.json).

Modes:
    analyze  — run 24 circuits (stoic-llm env, ~2.5-3 hr CPU)
    report   — pre-registered readout incl. the ctrl_03/duty_01 anchors
    plot     — per-item base|seneca|marcus figures (needs matplotlib)
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
sys.path.insert(0, str(REPO / "scripts"))

import exp12_circuit_analysis as harness  # exact same config/hotfix/threshold

OUT_DIR = harness.OUT_DIR

ITEMS = {
    "accepting": ["ctrl_06", "mort_03", "ext_04", "emot_01"],
    "active": ["ctrl_05", "emot_03", "duty_02", "emot_02"],
}
SWEEP_AUTHORS = ["seneca", "marcus"]  # CAA + epictetus deferred (settled)

ANCHORS = {  # already-run items -> their saved single-item JSONs
    "ctrl_03": ("accepting", "exp12_20260706_231538.json"),
    "duty_01": ("active", "exp12_20260707_163121.json"),
}


def _c(circ) -> float:
    """Signed content signal c from a circuit's stored metrics."""
    return 0.5 * (circ["clean_metric"] - circ["corrupted_metric"])


def _late_resolution(circ) -> dict:
    late = [n for n in circ.get("nodes", []) if (n.get("block_num") or 0) >= 23]
    late_max = max((abs(n["normalized_effect"]) for n in late), default=0.0)
    return {
        "n_late_nodes": len(late),
        "max_late_effect": late_max,
        "resolvable": len(late) >= 3 and late_max >= 0.2,
    }


def run_analyze():
    import torch

    harness._install_capture_hotfix()
    from modellens import ModelLens
    from modellens.analysis import activation_patching as ap
    from modellens.analysis.circuit_discovery import discover_circuit

    from stoic import config
    from stoic.dilemmas import PROMPT_TEMPLATE, _single_token_id, load_dilemmas, p_stoic
    from stoic.lora import merge_adapter
    from stoic.model import load_model

    t0 = time.time()
    model, tokenizer = load_model()
    tok_a = _single_token_id(tokenizer, " A")
    tok_b = _single_token_id(tokenizer, " B")

    def metric_fn(output):
        logits = output.logits if hasattr(output, "logits") else output
        return (logits[0, -1, tok_a] - logits[0, -1, tok_b]).float().item()

    assert metric_fn is not ap._default_metric, "metric_fn must not be max-logit"
    print("guardrail: content logit-diff metric (NOT _default_metric) ✓")

    all_items = {d["id"]: d for d in load_dilemmas()}
    sweep_ids = [i for ids in ITEMS.values() for i in ids]
    inputs = {}
    p_start = {}
    for iid in sweep_ids:
        d = all_items[iid]
        clean = PROMPT_TEMPLATE.format(situation=d["situation"], option_a=d["stoic"], option_b=d["nonstoic"])
        corr = PROMPT_TEMPLATE.format(situation=d["situation"], option_a=d["nonstoic"], option_b=d["stoic"])
        ci, xi = tokenizer(clean, return_tensors="pt"), tokenizer(corr, return_tensors="pt")
        assert ci["input_ids"].shape == xi["input_ids"].shape, f"{iid} length mismatch"
        inputs[iid] = (ci, xi)
        p_start[iid] = p_stoic(model, tokenizer, d, tok_a, tok_b)
        print(f"guardrail: base P(stoic) {iid} start = {p_start[iid]:.6f}")

    results = {
        "experiment": "12c_sweep",
        "primary_outcome": "signed dc = c_adapter - c_base (dc>0 = toward Stoic option)",
        "legacy_secondary": "|sensitivity|-% is CONFOUNDED (base metric stance-dependent); reported for anchor continuity only",
        "items": {iid: {"stance": s} for s, ids in ITEMS.items() for iid in ids},
        "conditions": ["base"] + [f"lora_{a}" for a in SWEEP_AUTHORS],
        "importance_threshold": harness.IMPORTANCE_THRESHOLD,
        "circuits": {iid: {} for iid in sweep_ids},
    }

    def circuit(active_lens, iid, label):
        ci, xi = inputs[iid]
        t = time.time()
        c = discover_circuit(active_lens, ci, xi, metric_fn=metric_fn,
                             importance_threshold=harness.IMPORTANCE_THRESHOLD)
        print(f"  [{iid}] {label}: clean {c['clean_metric']:+.3f} corr {c['corrupted_metric']:+.3f} "
              f"c={_c(c):+.3f} nodes {c.get('num_components', 0)} [{time.time()-t:.0f}s]")
        return harness.sanitize(c)

    lens = ModelLens(model)
    lens.adapter.set_tokenizer(tokenizer)
    print("\n=== BASE circuits (8 items) ===")
    for iid in sweep_ids:
        results["circuits"][iid]["base"] = circuit(lens, iid, "base")

    for author_name in SWEEP_AUTHORS:
        author = config.AUTHORS[author_name]
        print(f"\n=== LoRA {author_name} circuits (fresh base merge) ===")
        merged = merge_adapter(author.adapter_dir)
        lens_m = ModelLens(merged)
        lens_m.adapter.set_tokenizer(tokenizer)
        try:
            for iid in sweep_ids:
                results["circuits"][iid][f"lora_{author_name}"] = circuit(
                    lens_m, iid, f"lora_{author_name}")
        finally:
            del lens_m, merged
            gc.collect()

    print("\nguardrail: base integrity end-of-run")
    integrity = {}
    for iid in sweep_ids:
        p_end = p_stoic(model, tokenizer, all_items[iid], tok_a, tok_b)
        drift = abs(p_end - p_start[iid])
        integrity[iid] = {"p_start": p_start[iid], "p_end": p_end, "drift": drift}
        flag = "" if drift == 0.0 else "  !! VIOLATION !!"
        print(f"  {iid}: {p_start[iid]:.6f} -> {p_end:.6f}  drift {drift:.6f}{flag}")
    results["base_integrity"] = integrity

    results["runtime_sec"] = round(time.time() - t0, 1)
    out = OUT_DIR / f"exp12_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n↳ wrote {out.relative_to(REPO)}  ({results['runtime_sec']}s)")
    return out


def _load_anchor(fname, author):
    with open(OUT_DIR / fname) as f:
        d = json.load(f)
    return d["circuits"]["base"], d["circuits"][f"lora_{author}"]


def run_report(json_path: str):
    with open(json_path) as f:
        sweep = json.load(f)

    rows = []  # per (item, author): stance, base_c, dc, legacy%, shifts...
    stances = {}

    def add(iid, stance, base_circ, adapter_circs):
        stances[iid] = stance
        bc = _c(base_circ)
        b_eff = {n["name"]: n["normalized_effect"] for n in base_circ["nodes"]}
        res = _late_resolution(base_circ)
        for author, circ in adapter_circs.items():
            ac = _c(circ)
            tot_b, tot_a = abs(base_circ["total_effect"]), abs(circ["total_effect"])
            shifts = [abs(n["normalized_effect"] - b_eff[n["name"]])
                      for n in circ["nodes"] if n["name"] in b_eff]
            b_roles = {n["name"]: n["role"] for n in base_circ["nodes"] if (n.get("block_num") or 0) >= 23}
            a_roles = {n["name"]: n["role"] for n in circ["nodes"] if (n.get("block_num") or 0) >= 23}
            rows.append(dict(
                item=iid, stance=stance, author=author,
                base_c=bc, adapter_c=ac, dc=ac - bc,
                legacy_sens_pct=100 * (tot_a - tot_b) / tot_b,
                max_node_shift=max(shifts) if shifts else 0.0,
                n_nodes_base=len(base_circ["nodes"]), n_nodes=len(circ["nodes"]),
                late_resolvable=res["resolvable"],
                late_gate_changed=b_roles != a_roles,
            ))

    for iid, (stance, fname) in ANCHORS.items():
        base_circ, _ = _load_anchor(fname, "seneca")
        adapters = {a: _load_anchor(fname, a)[1] for a in SWEEP_AUTHORS}
        add(iid, stance, base_circ, adapters)
    for iid, entry in sweep["circuits"].items():
        adapters = {a: entry[f"lora_{a}"] for a in SWEEP_AUTHORS}
        add(iid, sweep["items"][iid]["stance"], entry["base"], adapters)

    print("=== PER-ITEM TABLE (primary: signed Δc; legacy %-sens CONFOUNDED) ===")
    print(f"{'item':>9} {'stance':>10} {'base c':>8} | "
          f"{'Sen Δc':>8} {'Sen c':>7} {'sens%':>7} {'shift':>6} {'gateΔ':>5} | "
          f"{'Mar Δc':>8} {'Mar c':>7} {'sens%':>7} {'shift':>6} {'gateΔ':>5} | res")
    by_item = {}
    for r in rows:
        by_item.setdefault(r["item"], {})[r["author"]] = r
    for iid, d in by_item.items():
        s, m = d["seneca"], d["marcus"]
        print(f"{iid:>9} {s['stance']:>10} {s['base_c']:>+8.3f} | "
              f"{s['dc']:>+8.3f} {s['adapter_c']:>+7.3f} {s['legacy_sens_pct']:>+6.1f}% {s['max_node_shift']:>6.3f} {str(s['late_gate_changed'])[:1]:>5} | "
              f"{m['dc']:>+8.3f} {m['adapter_c']:>+7.3f} {m['legacy_sens_pct']:>+6.1f}% {m['max_node_shift']:>6.3f} {str(m['late_gate_changed'])[:1]:>5} | "
              f"{'clean' if s['late_resolvable'] else 'MUSHY'}")

    import statistics as st
    print("\n=== PRE-REGISTERED ANSWERS ===")
    for author in SWEEP_AUTHORS:
        ar = [r for r in rows if r["author"] == author and r["late_resolvable"]]
        excl = [r["item"] for r in rows if r["author"] == author and not r["late_resolvable"]]
        for stance in ("accepting", "active"):
            g = [r for r in ar if r["stance"] == stance]
            vals = ", ".join(f"{r['item']}={r['dc']:+.2f}" for r in g)
            print(f"{author:>7} Δc {stance:>10}: mean {st.mean([r['dc'] for r in g]):+.3f}  [{vals}]")
        pos = [r for r in ar if r["base_c"] > 0]
        for stance in ("accepting", "active"):
            g = [r for r in pos if r["stance"] == stance]
            if g:
                print(f"{author:>7} Δc {stance:>10} (c>0 subset): mean {st.mean([r['dc'] for r in g]):+.3f} (n={len(g)})")
        if excl:
            print(f"{author:>7} floor-excluded items: {excl}")
    print("\nSeneca-vs-Marcus |max node shift| per item (largest modifier?):")
    for iid, d in by_item.items():
        s, m = d["seneca"]["max_node_shift"], d["marcus"]["max_node_shift"]
        print(f"  {iid:>9}: seneca {s:.3f} vs marcus {m:.3f} -> {'seneca' if s > m else 'MARCUS'}")
    print("\nΔc vs base-c (setpoint check, seneca):")
    for r in sorted((r for r in rows if r["author"] == "seneca"), key=lambda r: r["base_c"]):
        print(f"  base_c {r['base_c']:+.3f} -> Δc {r['dc']:+.3f}  ({r['item']}, {r['stance']})")


def run_plot(json_path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with open(json_path) as f:
        sweep = json.load(f)
    for iid, entry in sweep["circuits"].items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), sharey=True)
        harness._draw_circuit(axes[0], entry["base"], "BASE (control)")
        harness._draw_circuit(axes[1], entry["lora_seneca"], "LoRA — seneca")
        harness._draw_circuit(axes[2], entry["lora_marcus"], "LoRA — marcus")
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markeredgecolor="black", markersize=8, label=r)
                   for r, c in harness.ROLE_COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.04))
        stance = sweep["items"][iid]["stance"]
        fig.suptitle(f"Exp 12c sweep — {iid} ({stance}): node size = |normalized patch effect|", fontsize=10)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out = OUT_DIR / f"exp12_sweep_{iid}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"↳ wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("analyze")
    for name in ("report", "plot"):
        sp = sub.add_parser(name)
        sp.add_argument("json_path")
    args = p.parse_args()
    if args.cmd == "analyze":
        run_analyze()
    elif args.cmd == "report":
        run_report(args.json_path)
    else:
        run_plot(args.json_path)
