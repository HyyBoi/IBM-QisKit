import math

import numpy as np
from matplotlib.path import Path

from floodcut.config import ScenarioConfig
from floodcut.objective import build_problem, feasible_plans, validate_problem
from floodcut.scenario import DISTRICT_NAMES, generate_scenario


def test_deterministic_and_fingerprint():
    a = generate_scenario(ScenarioConfig(seed=3))
    b = generate_scenario(ScenarioConfig(seed=3))
    c = generate_scenario(ScenarioConfig(seed=4))
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    np.testing.assert_array_equal(a.r, b.r)


def test_geometry_consistent_with_coefficients():
    sc = generate_scenario(ScenarioConfig())
    # district membership used for coefficients == polygon drawn on the map
    for v, d in enumerate(sc.village_district):
        assert Path(sc.district_polygons[DISTRICT_NAMES[d]]).contains_point(tuple(sc.village_xy[v]))
    # harm is the sum of district burdens (consistency between H and F inputs)
    np.testing.assert_allclose(sc.h, sc.a.sum(axis=0))
    np.testing.assert_allclose(sc.P, [sc.village_pop[sc.village_district == d].sum() for d in range(3)])
    # symmetric, non-negative interaction matrices with zero diagonal
    for M in (sc.o, sc.s):
        np.testing.assert_allclose(M, M.T)
        assert np.all(M >= 0) and np.all(np.diag(M) == 0)
    # spills stay on their own bank
    for i in range(sc.n):
        assert np.all(sc.exposure[i, sc.village_bank != sc.site_bank[i]] == 0)


def test_exactly_k_generation():
    for n, k in [(10, 3), (12, 3), (10, 5), (6, 1)]:
        X = feasible_plans(n, k)
        assert X.shape == (math.comb(n, k), n)
        assert np.all(X.sum(axis=1) == k)
        assert len({tuple(r) for r in X}) == len(X)


def test_validation_over_all_plans(problem_variants):
    rep = validate_problem(problem_variants)
    assert rep["hard_failures"] == []
    assert rep["n_feasible"] == rep["expected_n_feasible"]
    assert rep["objectives_compete"]


def test_relief_lower_bound_holds():
    sc = generate_scenario(ScenarioConfig())
    for k in (2, 3, 4):
        pb = build_problem(sc, k)
        rep = validate_problem(pb)
        bound = (1 - sc.config.redundancy_eta * (k - 1) / 2) * np.sort(sc.r)[:k].sum()
        assert rep["R_min"] >= bound - 1e-12
