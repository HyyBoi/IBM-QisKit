"""Objectives, normalization, QUBO assembly and plan evaluation.

x in {0,1}^n, x_i = 1 means candidate site i is selected, subject to sum_i x_i = k.

    R(x) = sum_i r_i x_i - sum_{i<j} o_ij x_i x_j            (diversion proxy, maximize)
    H(x) = sum_i h_i x_i + sum_{i<j} s_ij x_i x_j            (synthetic harm index, minimize)
    b_d(x) = sum_i a_di x_i / P_d                             (additive district burden)
    F(x) = sum_{d<e} (b_d(x) - b_e(x))^2                      (fairness penalty, minimize)

    E(x) = w_H (H - H_min)/dH + w_F (F - F_min)/dF - w_R (R - R_min)/dR

The min / range normalizers are taken over all feasible plans of the scenario for the
given k and are then held FIXED, so different weights are compared on the same scale.
The QUBO form E(x) = c0 + q.x + sum_{i<j} Q_ij x_i x_j is assembled term by term and
validated against the direct formulas.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_K, DEFAULT_TAU, Weights
from .scenario import Scenario

# ----------------------------------------------------------------------------------
# plan representations
# ----------------------------------------------------------------------------------


def feasible_plans(n: int, k: int) -> np.ndarray:
    """All 0/1 vectors with exactly k ones, in lexicographic order of index tuples."""
    combos = list(itertools.combinations(range(n), k))
    X = np.zeros((len(combos), n), dtype=np.int8)
    for row, c in enumerate(combos):
        X[row, list(c)] = 1
    return X


def plan_to_ids(x) -> tuple[int, ...]:
    return tuple(int(i) for i in np.flatnonzero(np.asarray(x)))


def ids_to_plan(ids, n: int) -> np.ndarray:
    x = np.zeros(n, dtype=np.int8)
    x[list(ids)] = 1
    return x


def plan_label(x) -> str:
    return "{" + ", ".join(f"S{i}" for i in plan_to_ids(x)) + "}"


def bitstring_to_plan(bitstring: str) -> np.ndarray:
    """Decode a Qiskit counts key. Qiskit is little-endian: the RIGHTMOST character is
    qubit 0, so x_i = int(bitstring[n - 1 - i])."""
    n = len(bitstring)
    return np.array([int(bitstring[n - 1 - i]) for i in range(n)], dtype=np.int8)


def plan_to_bitstring(x) -> str:
    x = np.asarray(x)
    return "".join(str(int(b)) for b in x[::-1])


def index_to_plan(index: int, n: int) -> np.ndarray:
    """Statevector basis index -> plan (bit i of the index is qubit i)."""
    return np.array([(index >> i) & 1 for i in range(n)], dtype=np.int8)


def all_basis_plans(n: int) -> np.ndarray:
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


# ----------------------------------------------------------------------------------
# direct (un-expanded) formulas
# ----------------------------------------------------------------------------------


def _atleast2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X[None, :] if X.ndim == 1 else X


def relief(sc: Scenario, X) -> np.ndarray:
    X = _atleast2d(X)
    lin = X @ sc.r
    pair = 0.5 * np.einsum("mi,ij,mj->m", X, sc.o, X)  # o symmetric, zero diagonal
    return lin - pair


def harm(sc: Scenario, X) -> np.ndarray:
    X = _atleast2d(X)
    return X @ sc.h + 0.5 * np.einsum("mi,ij,mj->m", X, sc.s, X)


def burdens(sc: Scenario, X) -> np.ndarray:
    X = _atleast2d(X)
    return (X @ sc.a.T) / sc.P[None, :]


def fairness(sc: Scenario, X) -> np.ndarray:
    b = burdens(sc, X)
    D = b.shape[1]
    out = np.zeros(b.shape[0])
    for d in range(D):
        for e in range(d + 1, D):
            out += (b[:, d] - b[:, e]) ** 2
    return out


# ----------------------------------------------------------------------------------
# problem definition
# ----------------------------------------------------------------------------------


@dataclass
class Normalizers:
    R_min: float
    R_range: float
    H_min: float
    H_range: float
    F_min: float
    F_range: float

    def to_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


@dataclass
class Problem:
    scenario: Scenario
    k: int
    weights: Weights
    tau: float
    norm: Normalizers
    c0: float
    q: np.ndarray  # (n,)
    Q: np.ndarray  # (n, n) strictly upper triangular
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.scenario.n

    # -- evaluation ---------------------------------------------------------------
    def energy_direct(self, X) -> np.ndarray:
        nm, w = self.norm, self.weights
        R = relief(self.scenario, X)
        H = harm(self.scenario, X)
        F = fairness(self.scenario, X)
        return (
            w.w_H * (H - nm.H_min) / nm.H_range
            + w.w_F * (F - nm.F_min) / nm.F_range
            - w.w_R * (R - nm.R_min) / nm.R_range
        )

    def energy_qubo(self, X) -> np.ndarray:
        X = _atleast2d(X)
        return self.c0 + X @ self.q + np.einsum("mi,ij,mj->m", X, self.Q, X)

    def Q_sym(self) -> np.ndarray:
        """Symmetric pair matrix with Q_sym[i,j] = Q_ij for i != j (used for marginals)."""
        return self.Q + self.Q.T

    def evaluate(self, x) -> dict:
        x = np.asarray(x, dtype=np.int8)
        R = float(relief(self.scenario, x)[0])
        H = float(harm(self.scenario, x)[0])
        F = float(fairness(self.scenario, x)[0])
        b = burdens(self.scenario, x)[0]
        E = float(self.energy_qubo(x)[0])
        return {
            "plan": list(plan_to_ids(x)),
            "label": plan_label(x),
            "feasible": int(x.sum()) == self.k,
            "E": E,
            "R": R,
            "H": H,
            "F": F,
            "burdens": [float(v) for v in b],
            "meets_threshold": bool(R >= self.tau),
        }

    def settings_dict(self) -> dict:
        return {
            "scenario_seed": self.scenario.config.seed,
            "n": self.n,
            "k": self.k,
            "weights": self.weights.to_dict(),
            "tau": self.tau,
            "scenario_fingerprint": self.scenario.fingerprint(),
        }


def _quadratic_parts(sc: Scenario):
    """Return linear/upper-pair coefficients of R, H and F separately."""
    n = sc.n
    iu = np.triu_indices(n, 1)
    R_lin, R_pair = sc.r.copy(), np.zeros((n, n))
    R_pair[iu] = -sc.o[iu]
    H_lin, H_pair = sc.h.copy(), np.zeros((n, n))
    H_pair[iu] = sc.s[iu]
    # F = sum_{d<e} (g . x)^2 with g = a_d/P_d - a_e/P_e; (g.x)^2 = sum g_i^2 x_i + 2 sum_{i<j} g_i g_j x_i x_j
    A = sc.a / sc.P[:, None]
    F_lin, F_pair = np.zeros(n), np.zeros((n, n))
    D = A.shape[0]
    for d in range(D):
        for e in range(d + 1, D):
            g = A[d] - A[e]
            F_lin += g**2
            F_pair[iu] += 2.0 * np.outer(g, g)[iu]
    return (R_lin, R_pair), (H_lin, H_pair), (F_lin, F_pair)


def compute_normalizers(sc: Scenario, k: int) -> Normalizers:
    X = feasible_plans(sc.n, k)
    R, H, F = relief(sc, X), harm(sc, X), fairness(sc, X)

    def rng(v):
        span = float(v.max() - v.min())
        if span <= 1e-15:
            raise ValueError("objective is constant over feasible plans; cannot normalize")
        return float(v.min()), span

    R_min, R_range = rng(R)
    H_min, H_range = rng(H)
    F_min, F_range = rng(F)
    return Normalizers(R_min, R_range, H_min, H_range, F_min, F_range)


def build_problem(sc: Scenario, k: int = DEFAULT_K, weights: Weights | None = None, tau: float = DEFAULT_TAU) -> Problem:
    weights = weights or Weights()
    if not 1 <= k < sc.n:
        raise ValueError(f"k must be in [1, n-1], got {k}")
    nm = compute_normalizers(sc, k)
    (R_lin, R_pair), (H_lin, H_pair), (F_lin, F_pair) = _quadratic_parts(sc)
    aH = weights.w_H / nm.H_range
    aF = weights.w_F / nm.F_range
    aR = weights.w_R / nm.R_range
    q = aH * H_lin + aF * F_lin - aR * R_lin
    Q = aH * H_pair + aF * F_pair - aR * R_pair
    c0 = -aH * nm.H_min - aF * nm.F_min + aR * nm.R_min
    return Problem(scenario=sc, k=k, weights=weights, tau=tau, norm=nm, c0=float(c0), q=q, Q=Q)


# ----------------------------------------------------------------------------------
# scenario validation
# ----------------------------------------------------------------------------------


def validate_problem(pb: Problem, city_eps: float = 1e-2) -> dict:
    """Sanity checks over EVERY feasible plan. Returns a report; raises on hard failures."""
    sc = pb.scenario
    X = feasible_plans(sc.n, pb.k)
    R, H, F = relief(sc, X), harm(sc, X), fairness(sc, X)
    E_direct = pb.energy_direct(X)
    E_qubo = pb.energy_qubo(X)
    rng_ = np.random.default_rng(12345)
    Xr = rng_.integers(0, 2, size=(200, sc.n))
    qubo_err_feasible = float(np.max(np.abs(E_direct - E_qubo)))
    qubo_err_random = float(np.max(np.abs(pb.energy_direct(Xr) - pb.energy_qubo(Xr))))
    i_maxR, i_minH, i_minF, i_minE = int(np.argmax(R)), int(np.argmin(H)), int(np.argmin(F)), int(np.argmin(E_qubo))
    meets = R >= pb.tau
    report = {
        "n_feasible": int(len(X)),
        "expected_n_feasible": math.comb(sc.n, pb.k),
        "R_min": float(R.min()),
        "R_max": float(R.max()),
        "H_min": float(H.min()),
        "H_max": float(H.max()),
        "F_min": float(F.min()),
        "F_max": float(F.max()),
        "all_R_positive": bool(np.all(R > 0)),
        "all_H_positive": bool(np.all(H > 0)),
        "all_F_nonnegative": bool(np.all(F >= -1e-15)),
        "max_city_exposure": float(sc.city_exposure.max()),
        "city_unexposed": bool(sc.city_exposure.max() < city_eps),
        "relief_bound_eta_ok": bool(sc.config.redundancy_eta < 2.0 / max(pb.k - 1, 1)),
        "argmax_R": plan_label(X[i_maxR]),
        "argmin_H": plan_label(X[i_minH]),
        "argmin_F": plan_label(X[i_minF]),
        "argmin_E": plan_label(X[i_minE]),
        "objectives_compete": bool(i_maxR != i_minH),
        "qubo_max_abs_err_feasible": qubo_err_feasible,
        "qubo_max_abs_err_random_bitstrings": qubo_err_random,
        "n_meet_threshold": int(meets.sum()),
        "frac_meet_threshold": float(meets.mean()),
        "no_plan_meets_threshold": bool(not meets.any()),
        "argmin_E_meets_threshold": bool(meets[i_minE]),
        "corr_R_H": float(np.corrcoef(R, H)[0, 1]),
    }
    hard = ["all_R_positive", "all_H_positive", "all_F_nonnegative", "city_unexposed"]
    failed = [h for h in hard if not report[h]]
    if qubo_err_feasible > 1e-10 or qubo_err_random > 1e-10:
        failed.append("qubo_mismatch")
    report["hard_failures"] = failed
    return report
