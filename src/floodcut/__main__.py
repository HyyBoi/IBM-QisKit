"""Command-line entry point: python -m floodcut {validate,quick,bench,figures}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_K, DEFAULT_TAU, QAOAConfig, ScenarioConfig, Weights

ROOT = Path(__file__).resolve().parents[2]


def _fmt_row(name, r):
    thr = "meets" if r["meets_threshold"] else "BELOW"
    return (f"{name:<22} {r['label']:<18} E={r['E']:+.4f}  R={r['R']:.4f}  H={r['H']:.4f}  F={r['F']:.5f}  "
            f"threshold:{thr:<5}  gap={r['additive_gap']:.4f}  rank={r['rank']}/{r['n_feasible']}")


def cmd_validate(a):
    from .objective import build_problem, validate_problem
    from .scenario import generate_scenario

    sc = generate_scenario(ScenarioConfig(seed=a.seed, n_sites=a.n))
    pb = build_problem(sc, a.k, Weights(a.wH, a.wF, a.wR), a.tau)
    rep = validate_problem(pb)
    print(json.dumps(rep, indent=1))
    print("normalizers:", json.dumps(pb.norm.to_dict()))
    return 1 if rep["hard_failures"] else 0


def cmd_quick(a):
    from .benchmark import quick_compare
    from .results_io import save_json

    inst = quick_compare(ScenarioConfig(seed=a.seed, n_sites=a.n), a.k, Weights(a.wH, a.wF, a.wR), a.tau,
                         QAOAConfig(p=a.p, seed=a.opt_seed, maxiter=a.maxiter, shots=a.shots, init_state=a.init))
    q = inst["qaoa"][0]
    print(f"\nSYNTHETIC scenario seed={a.seed} n={a.n} k={a.k} weights=({a.wH},{a.wF},{a.wR}) tau={a.tau}")
    print(f"feasible plans meeting threshold: {inst['validation']['n_meet_threshold']}/{inst['validation']['n_feasible']}")
    print(_fmt_row("exact", inst["exact"]))
    print(_fmt_row("greedy", inst["greedy"]))
    print(_fmt_row("annealing (seed 0)", inst["annealing"][0]))
    print(_fmt_row("QAOA most-frequent", q["most_frequent"]))
    print(_fmt_row("QAOA best-observed*", q["best_observed"]))
    print(f"QAOA: <E> {q['expected_E_initial']:+.4f} -> {q['expected_E_opt']:+.4f} (uniform mean {q['uniform_feasible_mean_E']:+.4f}); "
          f"P(opt) exact {q['exact_p_opt']:.4f} vs uniform {q['uniform_p_opt']:.4f}; feasible rate {q['feasible_rate']:.4f}; "
          f"leakage {q['leakage_outside_k_subspace']:.1e}; evals {q['n_evaluations']}; "
          f"transpiled depth {q['circuit']['transpiled_depth']}, CX {q['circuit']['transpiled_cx']}")
    print("* best-observed is chosen by classically evaluating sampled plans; with many shots over few plans it is "
          "close to exhaustive search and is not evidence of quantum performance.")
    out = save_json(ROOT / "results" / "quick_run.json", inst)
    print(f"saved {out}")
    return 0


def cmd_bench(a):
    from .benchmark import run_benchmark
    from .results_io import save_json

    res = run_benchmark(a.preset)
    out = Path(a.out) if a.out else ROOT / "results" / f"benchmark_{a.preset}.json"
    save_json(out, res)
    print(f"\ncompleted {res['completed_qaoa_runs']} QAOA runs, {res['completed_sa_runs']} SA runs in {res['total_runtime_s']:.1f}s")
    for g in res["qaoa_aggregate"]:
        print(f"  n={g['n']} init={g['init_state']:<5} p={g['p']}: P(opt) mean {g['exact_p_opt']['mean']:.3f} "
              f"(x{g['p_opt_ratio_vs_uniform']['mean']:.1f} uniform), most-frequent optimal in "
              f"{g['most_frequent_is_optimal_rate']:.0%} of {g['runs']} runs, max-prob mean {g['max_plan_probability']['mean']:.3f}")
    print(f"saved {out}")
    return 0


def cmd_figures(a):
    from .figures import make_all_figures

    paths = make_all_figures(Path(a.results), ROOT / "docs" / "figures")
    for p in paths:
        print("wrote", p)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="floodcut", description="FloodCut (synthetic, educational) fixed-k QAOA prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--seed", type=int, default=ScenarioConfig().seed, help="scenario seed")
        p.add_argument("--n", type=int, default=10, choices=[10, 12])
        p.add_argument("--k", type=int, default=DEFAULT_K)
        p.add_argument("--wH", type=float, default=Weights().w_H)
        p.add_argument("--wF", type=float, default=Weights().w_F)
        p.add_argument("--wR", type=float, default=Weights().w_R)
        p.add_argument("--tau", type=float, default=DEFAULT_TAU)

    p = sub.add_parser("validate", help="validate scenario + QUBO on every feasible plan")
    common(p)
    p.set_defaults(func=cmd_validate)
    p = sub.add_parser("quick", help="one live comparison: exact, greedy, annealing, QAOA")
    common(p)
    p.add_argument("--p", type=int, default=1)
    p.add_argument("--opt-seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--init", choices=["dicke", "basis"], default="dicke")
    p.set_defaults(func=cmd_quick)
    p = sub.add_parser("bench", help="run the benchmark suite and save JSON")
    p.add_argument("--preset", choices=["quick", "full"], default="full")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_bench)
    p = sub.add_parser("figures", help="render report figures from saved results")
    p.add_argument("--results", default=str(ROOT / "results" / "benchmark_full.json"))
    p.set_defaults(func=cmd_figures)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
