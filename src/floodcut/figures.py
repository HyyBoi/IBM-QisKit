"""Render the report figures from a SAVED benchmark JSON (no numbers are invented here).

The scenario geometry is regenerated from the saved scenario config and checked against
the saved fingerprint before drawing, so the figures cannot silently drift from the data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ScenarioConfig, Weights
from .objective import build_problem
from .pareto import pareto_analysis
from .plotting import (METHOD_COLORS, architecture_figure, burden_figure, convergence_figure, counts_figure,
                       map_figure, method_maps_figure, pareto_figure, plt)
from .results_io import load_json
from .scenario import generate_scenario


def scenario_from_instance(inst: dict):
    cfg = ScenarioConfig(**inst["scenario_config"])
    sc = generate_scenario(cfg)
    if sc.fingerprint() != inst["settings"]["scenario_fingerprint"]:
        raise RuntimeError("regenerated scenario does not match the saved fingerprint")
    return sc


def default_instance(bench: dict) -> dict:
    d = ScenarioConfig()
    for inst in bench["instances"]:
        if inst["settings"]["scenario_seed"] == d.seed and inst["settings"]["n"] == d.n_sites:
            return inst
    return bench["instances"][0]


def pick_qaoa(inst: dict, init="dicke", p=2, seed=0):
    for r in inst["qaoa"]:
        c = r["config"]
        if c["init_state"] == init and c["p"] == p and c["seed"] == seed:
            return r
    return None


def popt_figure(aggregate: list[dict], figsize=(7.2, 3.4)):
    """P(opt) relative to uniform sampling, per (n, init, p) group: mean bar + min/max range."""
    fig, ax = plt.subplots(figsize=figsize)
    labels, means, lo, hi, colors = [], [], [], [], []
    for g in aggregate:
        labels.append(f"n={g['n']}\n{g['init_state']}\np={g['p']}")
        s = g["p_opt_ratio_vs_uniform"]
        means.append(s["mean"])
        lo.append(s["mean"] - s["min"])
        hi.append(s["max"] - s["mean"])
        colors.append(METHOD_COLORS["qaoa"] if g["init_state"] == "dicke" else "#B8B6AE")
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, width=0.6)
    ax.errorbar(x, means, yerr=[lo, hi], fmt="none", ecolor="#333333", capsize=3, lw=1)
    ax.axhline(1.0, color="#52514e", ls="--", lw=1, label="uniform random over feasible plans (= 1)")
    from matplotlib.patches import Patch

    handles = [Patch(color=METHOD_COLORS["qaoa"], label="Dicke start"), Patch(color="#B8B6AE", label="basis start (+ β0 layer)"),
               ax.get_lines()[-1]]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="upper left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("P(optimal plan) / uniform 1/C(n,k)")
    ax.set_title("Exact optimal-plan probability of the optimized circuit (bar = mean, whisker = min–max over runs)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def make_all_figures(results_path: Path, outdir: Path) -> list[Path]:
    bench = load_json(results_path)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        p = outdir / name
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    save(architecture_figure(), "architecture.png")

    inst = default_instance(bench)
    sc = scenario_from_instance(inst)
    st = inst["settings"]
    save(map_figure(sc, [], "Synthetic scenario: all 10 candidate sites with illustrative spill areas", show_all_spills=True,
                    figsize=(6, 6)), "map_scenario.png")

    q = pick_qaoa(inst) or inst["qaoa"][0]
    sa0 = inst["annealing"][0]
    plans = {
        "Exact optimum": (inst["exact"]["plan"], f"E={inst['exact']['E']:+.3f}"),
        "QAOA (most frequent)": (q["most_frequent"]["plan"], f"E={q['most_frequent']['E']:+.3f}, rank {q['most_frequent']['rank']}"),
        "Greedy": (inst["greedy"]["plan"], f"E={inst['greedy']['E']:+.3f}, rank {inst['greedy']['rank']}"),
        "Annealing (seed 0)": (sa0["plan"], f"E={sa0['E']:+.3f}, rank {sa0['rank']}"),
    }
    save(method_maps_figure(sc, plans, figsize=(14, 4.2)), "method_maps.png")

    save(burden_figure({
        "exact": inst["exact"]["burdens"],
        "qaoa": q["most_frequent"]["burdens"],
        "greedy": inst["greedy"]["burdens"],
        "annealing": sa0["burdens"],
    }), "burdens.png")

    pb = build_problem(sc, st["k"], Weights(**st["weights"]), st["tau"])
    par = pareto_analysis(pb)
    overlays = []
    markers = ["*", "P", "D", "X", "s", "^"]
    if bench.get("weight_sweep"):
        groups: dict = {}
        for row in bench["weight_sweep"]:  # identical plans share one marker listing all their weights
            m = row["qaoa_most_frequent"]
            groups.setdefault(m["label"], {"m": m, "ws": []})["ws"].append(row["weights"])
        for j, (label, g) in enumerate(groups.items()):
            ws = "; ".join(f"({w['w_H']:g},{w['w_F']:g},{w['w_R']:g})" for w in g["ws"])
            overlays.append({"R": g["m"]["R"], "H": g["m"]["H"], "marker": markers[j % len(markers)], "s": 130,
                             "color": METHOD_COLORS["qaoa"], "label": f"QAOA {label} for w={ws}"})
    save(pareto_figure(par["R"], par["H"], par["F"], par["nd3_mask"], par["nd2_mask"], overlays, highlight=st["tau"],
                       figsize=(7.4, 5.0)), "pareto.png")

    if q.get("top_samples"):
        opt_labels = {"{" + ", ".join(f"S{i}" for i in p) + "}" for p in inst["exact"]["optimal_plans"]}
        top = [(r["label"], r["count"]) for r in q["top_samples"]]
        save(counts_figure(top, next(iter(opt_labels))), "qaoa_counts.png")

    hist = {}
    for init in ("dicke", "basis"):
        for p in (1, 2):
            r = pick_qaoa(inst, init, p, 0)
            if r and r.get("history"):
                hist[f"{init} p={p}"] = r["history"]
    if hist:
        save(convergence_figure(hist), "convergence.png")

    save(popt_figure(bench["qaoa_aggregate"]), "p_opt_summary.png")
    return written
