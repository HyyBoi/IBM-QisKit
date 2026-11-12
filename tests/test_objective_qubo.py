import numpy as np

from floodcut.config import Weights
from floodcut.objective import (all_basis_plans, bitstring_to_plan, build_problem, fairness, feasible_plans,
                                index_to_plan, plan_to_bitstring)


def test_direct_equals_qubo_on_every_feasible_plan(problem_variants):
    pb = problem_variants
    X = feasible_plans(pb.n, pb.k)
    np.testing.assert_allclose(pb.energy_direct(X), pb.energy_qubo(X), atol=1e-10, rtol=0)


def test_direct_equals_qubo_on_all_bitstrings(default_problem):
    X = all_basis_plans(default_problem.n)  # the identity holds for every x, not only feasible ones
    np.testing.assert_allclose(default_problem.energy_direct(X), default_problem.energy_qubo(X), atol=1e-10, rtol=0)


def test_qubo_for_several_weights(default_problem):
    sc = default_problem.scenario
    X = feasible_plans(sc.n, 3)
    for w in [Weights(0, 0, 1), Weights(1, 0, 0), Weights(0, 1, 0), Weights(2.5, 0.3, 0.7)]:
        pb = build_problem(sc, 3, w)
        np.testing.assert_allclose(pb.energy_direct(X), pb.energy_qubo(X), atol=1e-10, rtol=0)


def test_normalizers_fixed_and_independent_of_weights(default_problem):
    sc = default_problem.scenario
    a = build_problem(sc, 3, Weights(1, 0.5, 1)).norm.to_dict()
    b = build_problem(sc, 3, Weights(0.2, 3, 0.1)).norm.to_dict()
    assert a == b


def test_fairness_zero_for_no_selection(default_problem):
    sc = default_problem.scenario
    assert np.all(fairness(sc, np.zeros(sc.n)) == 0)


def test_bitstring_roundtrip_little_endian():
    x = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1], dtype=np.int8)
    s = plan_to_bitstring(x)
    assert s == "1000001001"  # rightmost character = qubit 0
    np.testing.assert_array_equal(bitstring_to_plan(s), x)
    idx = sum(int(b) << i for i, b in enumerate(x))
    np.testing.assert_array_equal(index_to_plan(idx, 10), x)
