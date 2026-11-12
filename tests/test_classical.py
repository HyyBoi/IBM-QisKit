import numpy as np

from floodcut.classical import exact_solve, gap_metrics, greedy, simulated_annealing
from floodcut.config import SAConfig
from floodcut.objective import plan_to_ids


def test_exact_matches_independent_bruteforce(problem_variants):
    pb = problem_variants
    ex = exact_solve(pb)
    best, best_E = None, np.inf
    for idx in range(2**pb.n):  # independent enumeration over all 2^n states using the direct formulas
        if bin(idx).count("1") != pb.k:
            continue
        x = np.array([(idx >> i) & 1 for i in range(pb.n)])
        E = pb.energy_direct(x)[0]
        if E < best_E - 1e-12:
            best, best_E = x, E
    assert abs(best_E - ex["E_star"]) < 1e-10
    assert list(plan_to_ids(best)) in ex["optimal_plans"]


def test_greedy_and_sa_feasible(problem_variants):
    pb = problem_variants
    g = greedy(pb)
    assert int(g["plan"].sum()) == pb.k
    for s in range(5):
        r = simulated_annealing(pb, SAConfig(seed=s, steps=60))
        assert int(r["plan"].sum()) == pb.k


def test_greedy_first_pick_is_best_single_site(default_problem):
    pb = default_problem
    g = greedy(pb)
    assert g["order"][0] == int(np.argmin(pb.q))


def test_gap_metrics(default_problem):
    ex = exact_solve(default_problem)
    m = gap_metrics(ex["E_star"], ex)
    assert m["additive_gap"] == 0 and m["rank"] == 1 and m["is_optimal"]
    m = gap_metrics(ex["E_max"], ex)
    assert abs(m["normalized_gap"] - 1) < 1e-12 and m["rank"] >= 1


def test_sa_is_seeded(default_problem):
    a = simulated_annealing(default_problem, SAConfig(seed=7))
    b = simulated_annealing(default_problem, SAConfig(seed=7))
    assert a["trace"] == b["trace"]
