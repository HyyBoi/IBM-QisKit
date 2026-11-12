"""Exact Pareto (nondominated) analysis over all feasible plans of a small instance."""

from __future__ import annotations

import numpy as np

from .objective import Problem, burdens, fairness, feasible_plans, harm, plan_label, relief

TOL = 1e-12


def dominates(a: np.ndarray, b: np.ndarray, tol: float = TOL) -> bool:
    """a dominates b (all objectives MINIMIZED): a <= b everywhere and a < b somewhere."""
    return bool(np.all(a <= b + tol) and np.any(a < b - tol))


def nondominated_mask(objs: np.ndarray, tol: float = TOL) -> np.ndarray:
    """objs: (m, d) objectives to MINIMIZE. Exact duplicates are both kept."""
    objs = np.asarray(objs, dtype=float)
    m = objs.shape[0]
    mask = np.ones(m, dtype=bool)
    for i in range(m):
        le = np.all(objs <= objs[i] + tol, axis=1)
        lt = np.any(objs < objs[i] - tol, axis=1)
        if np.any(le & lt):
            mask[i] = False
    return mask


def weighted_sum_supported(objs: np.ndarray, nd_mask: np.ndarray, grid: int = 60) -> np.ndarray:
    """Grid-based check: which nondominated points are the unique-or-tied minimizer of
    SOME weighted sum w . f(x) with w on a simplex grid (objectives scaled to [0, 1]).
    Points never selected are (grid-)unsupported: a weighted-sum sweep would miss them."""
    f = np.asarray(objs, dtype=float)
    span = f.max(axis=0) - f.min(axis=0)
    span[span == 0] = 1.0
    fn = (f - f.min(axis=0)) / span
    hit = np.zeros(len(f), dtype=bool)
    d = f.shape[1]
    if d == 3:
        for a in range(grid + 1):
            for b in range(grid + 1 - a):
                w = np.array([a, b, grid - a - b], dtype=float) / grid
                vals = fn @ w
                hit |= vals <= vals.min() + 1e-12
    else:
        for a in range(grid + 1):
            w = np.array([a, grid - a], dtype=float) / grid
            vals = fn @ w
            hit |= vals <= vals.min() + 1e-12
    return hit & nd_mask


def pareto_analysis(pb: Problem) -> dict:
    sc = pb.scenario
    X = feasible_plans(sc.n, pb.k)
    R, H, F = relief(sc, X), harm(sc, X), fairness(sc, X)
    objs3 = np.column_stack([-R, H, F])
    nd3 = nondominated_mask(objs3)
    nd2 = nondominated_mask(np.column_stack([-R, H]))
    supported3 = weighted_sum_supported(objs3, nd3)
    # Fairness/harm tension inside the nondominated set: pairs where one plan is fairer
    # (lower F) but causes more total harm (higher H).
    idx = np.flatnonzero(nd3)
    tension_pairs = 0
    for a in idx:
        for b in idx:
            if F[a] < F[b] - TOL and H[a] > H[b] + TOL:
                tension_pairs += 1
    rows = []
    for i in idx:
        rows.append({
            "plan": [int(v) for v in np.flatnonzero(X[i])],
            "label": plan_label(X[i]),
            "R": float(R[i]), "H": float(H[i]), "F": float(F[i]),
            "burdens": [float(v) for v in burdens(sc, X[i])[0]],
            "meets_threshold": bool(R[i] >= pb.tau),
            "weighted_sum_supported_grid": bool(supported3[i]),
        })
    rows.sort(key=lambda r: -r["R"])
    return {
        "X": X, "R": R, "H": H, "F": F,
        "nd3_mask": nd3, "nd2_mask": nd2,
        "n_feasible": int(len(X)),
        "n_nondominated_3obj": int(nd3.sum()),
        "n_frontier_2obj": int(nd2.sum()),
        "n_unsupported_3obj_grid": int((nd3 & ~supported3).sum()),
        "fairness_harm_tension_pairs": int(tension_pairs),
        "nondominated_rows": rows,
    }
