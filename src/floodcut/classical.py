"""Classical baselines on the SAME Problem object: exact enumeration, greedy, annealing."""

from __future__ import annotations

import math
import time

import numpy as np

from .config import SAConfig
from .objective import Problem, feasible_plans, plan_label, plan_to_ids

TIE_TOL = 1e-12


def exact_solve(pb: Problem) -> dict:
    """Enumerate all C(n,k) feasible plans."""
    t0 = time.perf_counter()
    X = feasible_plans(pb.n, pb.k)
    E = pb.energy_qubo(X)
    runtime = time.perf_counter() - t0
    E_star = float(E.min())
    opt_rows = np.flatnonzero(E <= E_star + TIE_TOL)
    best = X[opt_rows[0]]
    return {
        "method": "exact",
        "X": X,
        "E_all": E,
        "E_star": E_star,
        "E_max": float(E.max()),
        "E_mean_uniform_feasible": float(E.mean()),
        "optimal_plans": [list(plan_to_ids(X[r])) for r in opt_rows],
        "plan": best,
        "evaluations": int(len(X)),
        "runtime_s": runtime,
    }


def gap_metrics(E: float, exact: dict) -> dict:
    """Additive gap E - E*, normalized gap (E - E*)/(E_max - E*) in [0, 1], and rank.

    Percentage gaps relative to E* are NOT used: E* can be zero or negative.
    Rank is 1 + the number of feasible plans strictly better than E.
    """
    E_star, E_max = exact["E_star"], exact["E_max"]
    span = E_max - E_star
    rank = 1 + int(np.sum(exact["E_all"] < E - TIE_TOL))
    return {
        "additive_gap": float(E - E_star),
        "normalized_gap": float((E - E_star) / span) if span > 0 else 0.0,
        "rank": rank,
        "n_feasible": int(len(exact["E_all"])),
        "is_optimal": bool(E <= E_star + TIE_TOL),
    }


def greedy(pb: Problem) -> dict:
    """Start empty; repeatedly add the site with the lowest marginal change of E.

    Marginal of adding i to S: q_i + sum_{j in S} Q_sym[i, j]  (c0 is constant).
    Ties are broken by lowest index. Deterministic. Evaluations counted as marginal
    computations.
    """
    t0 = time.perf_counter()
    Qs = pb.Q_sym()
    selected: list[int] = []
    evals = 0
    for _ in range(pb.k):
        best_i, best_d = None, math.inf
        for i in range(pb.n):
            if i in selected:
                continue
            d = pb.q[i] + sum(Qs[i, j] for j in selected)
            evals += 1
            if d < best_d - TIE_TOL:
                best_i, best_d = i, d
        selected.append(best_i)
    x = np.zeros(pb.n, dtype=np.int8)
    x[selected] = 1
    return {"method": "greedy", "plan": x, "order": selected, "evaluations": evals,
            "runtime_s": time.perf_counter() - t0}


def simulated_annealing(pb: Problem, cfg: SAConfig | None = None) -> dict:
    """Swap-move simulated annealing that always keeps exactly k sites selected.

    Move: swap one selected site with one unselected site (uniformly at random).
    Acceptance: Metropolis. Schedule: geometric from T0 to T0 * t_end_ratio over
    cfg.steps steps. T0 = mean |dE| of cfg.calibration_swaps random swaps from the
    initial state (no information from exact enumeration is used).
    """
    cfg = cfg or SAConfig()
    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    n, k = pb.n, pb.k
    Qs = pb.Q_sym()
    x = np.zeros(n, dtype=np.int8)
    x[rng.choice(n, size=k, replace=False)] = 1

    def energy(v):
        return float(pb.energy_qubo(v)[0])

    def swap_delta(v, i_out, j_in):
        # remove i_out, add j_in
        sel = np.flatnonzero(v)
        d = -pb.q[i_out] - sum(Qs[i_out, s] for s in sel if s != i_out)
        d += pb.q[j_in] + sum(Qs[j_in, s] for s in sel if s != i_out)
        return float(d)

    def random_swap(v):
        sel = np.flatnonzero(v)
        uns = np.flatnonzero(v == 0)
        return int(rng.choice(sel)), int(rng.choice(uns))

    E = energy(x)
    evals = 1
    deltas = []
    for _ in range(cfg.calibration_swaps):
        i, j = random_swap(x)
        deltas.append(abs(swap_delta(x, i, j)))
        evals += 1
    T0 = max(float(np.mean(deltas)), 1e-9)
    T_end = T0 * cfg.t_end_ratio
    best_x, best_E = x.copy(), E
    trace = [E]
    accepted = 0
    for step in range(cfg.steps):
        T = T0 * (T_end / T0) ** (step / max(cfg.steps - 1, 1))
        i, j = random_swap(x)
        d = swap_delta(x, i, j)
        evals += 1
        if d <= 0 or rng.uniform() < math.exp(-d / T):
            x[i], x[j] = 0, 1
            E += d
            accepted += 1
            if E < best_E - TIE_TOL:
                best_x, best_E = x.copy(), E
        trace.append(E)
    assert int(best_x.sum()) == k
    return {
        "method": "annealing",
        "plan": best_x,
        "evaluations": evals,
        "accepted": accepted,
        "T0": T0,
        "trace": trace,
        "runtime_s": time.perf_counter() - t0,
        "config": cfg.to_dict(),
    }


def summarize(pb: Problem, method: str, plan, exact: dict, extra: dict | None = None) -> dict:
    ev = pb.evaluate(plan)
    out = {"method": method, **ev, **gap_metrics(ev["E"], exact)}
    if extra:
        out.update(extra)
    out["label"] = plan_label(plan)
    return out
