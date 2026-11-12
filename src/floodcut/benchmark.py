"""Benchmark protocol: every method evaluated on the same Problem, with provenance."""

from __future__ import annotations

import math
import time

import numpy as np

from .classical import exact_solve, gap_metrics, greedy, simulated_annealing, summarize
from .config import DEFAULT_K, DEFAULT_TAU, QAOAConfig, SAConfig, ScenarioConfig, Weights
from .objective import Problem, build_problem, ids_to_plan, validate_problem
from .pareto import pareto_analysis
from .qaoa import run_qaoa
from .results_io import provenance
from .scenario import generate_scenario

PRESETS = {
    "quick": {
        "instances": [{"seed": 1, "n": 10}],
        "k": DEFAULT_K,
        "qaoa_seeds": [0],
        "ps": [1],
        "inits": ["dicke"],
        "maxiter": 50,
        "shots": 1024,
        "sa_seeds": [0, 1, 2],
        "sa_steps": 200,
        "weight_sweep": False,
    },
    "full": {
        "instances": [{"seed": 0, "n": 10}, {"seed": 1, "n": 10}, {"seed": 2, "n": 10}, {"seed": 1, "n": 12}],
        "k": DEFAULT_K,
        "qaoa_seeds": [0, 1, 2],
        "ps": [1, 2],
        "inits": ["dicke", "basis"],
        "maxiter": None,  # 150 for p=1, 300 for p=2
        "shots": 4096,
        "sa_seeds": list(range(10)),
        "sa_steps": 200,
        "weight_sweep": True,
    },
}

WEIGHT_SWEEP = [
    Weights(1.0, 0.5, 1.0),
    Weights(0.0, 0.0, 1.0),
    Weights(1.0, 0.0, 1.0),
    Weights(2.0, 0.5, 1.0),
    Weights(1.0, 2.0, 1.0),
    Weights(1.0, 0.5, 0.25),
]


def qaoa_summary(pb: Problem, res: dict, exact: dict, keep_counts: bool = True, keep_history: bool = False) -> dict:
    a = res["analysis"]
    shown = summarize(pb, "qaoa", ids_to_plan(a["most_frequent"]["plan"], pb.n), exact)
    best = summarize(pb, "qaoa-best-observed", ids_to_plan(a["best_observed"]["plan"], pb.n), exact) if a["best_observed"] else None
    span = exact["E_max"] - exact["E_star"]
    out = {
        "config": res["config"],
        "displayed_plan_rule": res["displayed_plan_rule"],
        "most_frequent": {**shown, "count": a["most_frequent"]["count"], "bitstring": a["most_frequent"]["bitstring"]},
        "best_observed": {**best, "count": a["best_observed"]["count"]} if best else None,
        "expected_E_initial": res["expected_E_initial"],
        "expected_E_opt": res["expected_E_opt"],
        "expected_E_normalized_gap": (res["expected_E_opt"] - exact["E_star"]) / span if span > 0 else 0.0,
        "uniform_feasible_mean_E": res["uniform_feasible_mean_E"],
        "exact_p_opt": res["exact_p_opt"],
        "empirical_p_opt": a["empirical_p_opt"],
        "uniform_p_opt": res["uniform_p_opt"],
        "p_opt_ratio_vs_uniform": res["exact_p_opt"] / res["uniform_p_opt"],
        "p_see_opt_within_m_shots": res["p_see_opt_within_m_shots"],
        "feasible_rate": a["feasible_rate"],
        "leakage_outside_k_subspace": res["leakage_outside_k_subspace"],
        "max_plan_probability": res["max_plan_probability"],
        "distinct_bitstrings": a["distinct_bitstrings"],
        "n_evaluations": res["n_evaluations"],
        "maxiter": res["maxiter"],
        "optimizer_message": res["optimizer_message"],
        "theta0": res["theta0"],
        "theta_opt": res["theta_opt"],
        "setup_s": res["setup_s"],
        "optimize_s": res["optimize_s"],
        "sample_s": res["sample_s"],
        "circuit": res["circuit"],
        "shots": res["shots"],
        "sampler_seed": res["sampler_seed"],
        "top_samples": a["top"],
    }
    if keep_counts:
        out["counts"] = res["counts"]
    if keep_history:
        out["history"] = res.get("history", [])
    return out


def run_instance(scfg: ScenarioConfig, k: int = DEFAULT_K, weights: Weights | None = None, tau: float = DEFAULT_TAU,
                 qaoa_cfgs: list[QAOAConfig] | None = None, sa_seeds=(0,), sa_steps: int = 200,
                 keep_history: bool = False, log=print) -> dict:
    weights = weights or Weights()
    t0 = time.perf_counter()
    sc = generate_scenario(scfg)
    pb = build_problem(sc, k, weights, tau)
    setup_s = time.perf_counter() - t0
    validation = validate_problem(pb)
    exact = exact_solve(pb)
    E_star_plan = ids_to_plan(exact["optimal_plans"][0], pb.n)
    exact_row = summarize(pb, "exact", E_star_plan, exact, {"evaluations": exact["evaluations"], "runtime_s": exact["runtime_s"]})
    g = greedy(pb)
    greedy_row = summarize(pb, "greedy", g["plan"], exact, {"evaluations": g["evaluations"], "runtime_s": g["runtime_s"],
                                                             "order": g["order"]})
    sa_rows = []
    for s in sa_seeds:
        r = simulated_annealing(pb, SAConfig(steps=sa_steps, seed=s))
        sa_rows.append(summarize(pb, "annealing", r["plan"], exact, {"evaluations": r["evaluations"], "runtime_s": r["runtime_s"],
                                                                      "seed": s, "steps": sa_steps, "T0": r["T0"]}))
    qaoa_rows = []
    for qc in qaoa_cfgs or []:
        t = time.perf_counter()
        res = run_qaoa(pb, qc, exact, keep_history=keep_history)
        row = qaoa_summary(pb, res, exact, keep_history=keep_history)
        qaoa_rows.append(row)
        log(f"  QAOA n={pb.n} seed={scfg.seed} init={qc.init_state} p={qc.p} opt-seed={qc.seed}: "
            f"<E>={row['expected_E_opt']:+.4f} (E*={exact['E_star']:+.4f}) P(opt)={row['exact_p_opt']:.3f} "
            f"[uniform {row['uniform_p_opt']:.3f}] most-frequent rank {row['most_frequent']['rank']} "
            f"max-prob {row['max_plan_probability']:.3f} ({time.perf_counter() - t:.1f}s)")
    par = pareto_analysis(pb)
    return {
        "settings": pb.settings_dict(),
        "scenario_config": scfg.to_dict(),
        "normalizers": pb.norm.to_dict(),
        "qubo": {"c0": pb.c0, "q": pb.q, "Q_upper": pb.Q},
        "coefficients": {"r": sc.r, "o": sc.o, "h": sc.h, "s": sc.s, "a": sc.a, "P": sc.P},
        "coefficient_table": sc.coefficient_table(),
        "validation": validation,
        "setup_s": setup_s,
        "exact": {
            **exact_row,
            "E_star": exact["E_star"],
            "E_max": exact["E_max"],
            "E_mean_uniform_feasible": exact["E_mean_uniform_feasible"],
            "optimal_plans": exact["optimal_plans"],
        },
        "greedy": greedy_row,
        "annealing": sa_rows,
        "qaoa": qaoa_rows,
        "pareto": {k_: v for k_, v in par.items() if k_ not in ("X",)},
    }


def weight_sweep(scfg: ScenarioConfig, k: int, tau: float, qcfg: QAOAConfig, weights_list=WEIGHT_SWEEP, log=print) -> list[dict]:
    """QAOA under a small sweep of weights on ONE scenario (normalizers are weight independent)."""
    sc = generate_scenario(scfg)
    out = []
    base = build_problem(sc, k, Weights(), tau)
    par = pareto_analysis(base)
    nd_labels = {r["label"] for r in par["nondominated_rows"]}
    for w in weights_list:
        pb = build_problem(sc, k, w, tau)
        exact = exact_solve(pb)
        res = run_qaoa(pb, qcfg, exact, keep_history=False)
        row = qaoa_summary(pb, res, exact, keep_counts=False)
        mf = row["most_frequent"]
        ex_plan = ids_to_plan(exact["optimal_plans"][0], pb.n)
        ex_eval = pb.evaluate(ex_plan)
        out.append({
            "weights": w.to_dict(),
            "qaoa_most_frequent": {k_: mf[k_] for k_ in ("plan", "label", "R", "H", "F", "E", "rank", "is_optimal", "meets_threshold")},
            "qaoa_in_3obj_nondominated": mf["label"] in nd_labels,
            "exact_weighted_optimum": {k_: ex_eval[k_] for k_ in ("plan", "label", "R", "H", "F", "E", "meets_threshold")},
            "exact_p_opt": row["exact_p_opt"],
            "uniform_p_opt": row["uniform_p_opt"],
        })
        log(f"  sweep w={w.to_dict()}: QAOA {mf['label']} rank {mf['rank']} | exact {ex_eval['label']}")
    return out


def aggregate(instances: list[dict]) -> list[dict]:
    """Mean / std / min / max of QAOA metrics grouped by (n, init_state, p)."""
    groups: dict = {}
    for inst in instances:
        for row in inst["qaoa"]:
            key = (inst["settings"]["n"], row["config"]["init_state"], row["config"]["p"])
            groups.setdefault(key, []).append(row)
    out = []
    for (n, init, p), rows in sorted(groups.items()):
        def stats(vals):
            v = np.asarray(vals, dtype=float)
            return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "min": float(v.min()), "max": float(v.max())}
        out.append({
            "n": n, "init_state": init, "p": p, "runs": len(rows),
            "exact_p_opt": stats([r["exact_p_opt"] for r in rows]),
            "p_opt_ratio_vs_uniform": stats([r["p_opt_ratio_vs_uniform"] for r in rows]),
            "expected_E_normalized_gap": stats([r["expected_E_normalized_gap"] for r in rows]),
            "most_frequent_normalized_gap": stats([r["most_frequent"]["normalized_gap"] for r in rows]),
            "most_frequent_is_optimal_rate": float(np.mean([r["most_frequent"]["is_optimal"] for r in rows])),
            "runs_p_opt_above_uniform": int(sum(r["exact_p_opt"] > r["uniform_p_opt"] for r in rows)),
            "runs_expected_E_below_uniform_mean": int(sum(r["expected_E_opt"] < r["uniform_feasible_mean_E"] for r in rows)),
            "runs_expected_E_below_initial": int(sum(r["expected_E_opt"] < r["expected_E_initial"] for r in rows)),
            "max_plan_probability": stats([r["max_plan_probability"] for r in rows]),
            "max_leakage": float(max(r["leakage_outside_k_subspace"] for r in rows)),
            "min_feasible_rate": float(min(r["feasible_rate"] for r in rows)),
            "n_evaluations": stats([r["n_evaluations"] for r in rows]),
            "optimize_s": stats([r["optimize_s"] for r in rows]),
            "sample_s": stats([r["sample_s"] for r in rows]),
            "transpiled_cx": int(rows[0]["circuit"]["transpiled_cx"]),
            "transpiled_depth": int(rows[0]["circuit"]["transpiled_depth"]),
            "layers_transpiled_cx": int(rows[0]["circuit"]["layers_transpiled_cx"]),
            "prep_transpiled_cx": int(rows[0]["circuit"]["prep_transpiled_cx"]),
        })
    return out


def sa_budget_study(instances_specs, k: int, steps_list=(10, 25, 50, 200), seeds=range(10)) -> list[dict]:
    """Success rate of swap-move annealing versus its evaluation budget (same objective)."""
    out = []
    for spec in instances_specs:
        sc = generate_scenario(ScenarioConfig(seed=spec["seed"], n_sites=spec["n"]))
        pb = build_problem(sc, k)
        exact = exact_solve(pb)
        for steps in steps_list:
            rows = [summarize(pb, "annealing", simulated_annealing(pb, SAConfig(steps=steps, seed=s))["plan"], exact)
                    for s in seeds]
            out.append({
                "scenario_seed": spec["seed"], "n": spec["n"], "steps": steps,
                "evaluations": 1 + SAConfig().calibration_swaps + steps,
                "n_feasible": exact["evaluations"],
                "runs": len(rows),
                "success_rate": float(np.mean([r["is_optimal"] for r in rows])),
                "mean_additive_gap": float(np.mean([r["additive_gap"] for r in rows])),
                "worst_rank": int(max(r["rank"] for r in rows)),
            })
    return out


def classical_aggregate(instances: list[dict]) -> list[dict]:
    out = []
    for inst in instances:
        sa = inst["annealing"]
        out.append({
            "scenario_seed": inst["settings"]["scenario_seed"],
            "n": inst["settings"]["n"],
            "n_feasible": inst["exact"]["n_feasible"],
            "exact_E_star": inst["exact"]["E_star"],
            "exact_label": inst["exact"]["label"],
            "exact_meets_threshold": inst["exact"]["meets_threshold"],
            "exact_runtime_s": inst["exact"]["runtime_s"],
            "greedy_label": inst["greedy"]["label"],
            "greedy_rank": inst["greedy"]["rank"],
            "greedy_additive_gap": inst["greedy"]["additive_gap"],
            "sa_runs": len(sa),
            "sa_success_rate": float(np.mean([r["is_optimal"] for r in sa])) if sa else None,
            "sa_mean_additive_gap": float(np.mean([r["additive_gap"] for r in sa])) if sa else None,
            "sa_worst_rank": int(max(r["rank"] for r in sa)) if sa else None,
            "sa_evaluations": int(sa[0]["evaluations"]) if sa else None,
            "n_meet_threshold": inst["validation"]["n_meet_threshold"],
        })
    return out


def run_benchmark(preset: str = "full", log=print, overrides: dict | None = None) -> dict:
    spec = {**PRESETS[preset], **(overrides or {})}
    t0 = time.perf_counter()
    instances = []
    for inst_spec in spec["instances"]:
        scfg = ScenarioConfig(seed=inst_spec["seed"], n_sites=inst_spec["n"])
        cfgs = [QAOAConfig(p=p, seed=s, init_state=init, maxiter=spec["maxiter"], shots=spec["shots"])
                for init in spec["inits"] for p in spec["ps"] for s in spec["qaoa_seeds"]]
        log(f"instance seed={scfg.seed} n={scfg.n_sites} k={spec['k']}: C(n,k)={math.comb(scfg.n_sites, spec['k'])}")
        is_default = scfg.seed == ScenarioConfig().seed and scfg.n_sites == 10
        instances.append(run_instance(scfg, spec["k"], Weights(), DEFAULT_TAU, cfgs, spec["sa_seeds"], spec["sa_steps"],
                                      keep_history=is_default, log=log))
    sweep = None
    if spec["weight_sweep"]:
        log("weight sweep on default scenario (QAOA dicke p=2 seed 0)")
        sweep = weight_sweep(ScenarioConfig(), spec["k"], DEFAULT_TAU, QAOAConfig(p=2, seed=0, shots=spec["shots"]), log=log)
    total = time.perf_counter() - t0
    return {
        "preset": preset,
        "spec": spec,
        "provenance": provenance(),
        "total_runtime_s": total,
        "completed_qaoa_runs": int(sum(len(i["qaoa"]) for i in instances)),
        "completed_sa_runs": int(sum(len(i["annealing"]) for i in instances)),
        "instances": instances,
        "qaoa_aggregate": aggregate(instances),
        "classical_aggregate": classical_aggregate(instances),
        "sa_budget_study": sa_budget_study(spec["instances"], spec["k"]),
        "weight_sweep": sweep,
    }


def quick_compare(scfg: ScenarioConfig, k: int, weights: Weights, tau: float, qcfg: QAOAConfig, sa_seed: int = 0,
                  sa_steps: int = 200, log=print) -> dict:
    """One live comparison of all four methods (used by the CLI and the demo)."""
    inst = run_instance(scfg, k, weights, tau, [qcfg], [sa_seed], sa_steps, keep_history=True, log=log)
    inst["provenance"] = provenance()
    inst["mode"] = "live"
    return inst


__all__ = ["PRESETS", "run_benchmark", "run_instance", "quick_compare", "weight_sweep", "aggregate", "gap_metrics"]
