import numpy as np

from floodcut.pareto import dominates, nondominated_mask, pareto_analysis


def test_dominance_basic_and_ties():
    assert dominates(np.array([1, 1]), np.array([2, 1]))
    assert not dominates(np.array([1, 1]), np.array([1, 1]))  # equal points: no dominance
    assert not dominates(np.array([1, 3]), np.array([2, 1]))


def test_nondominated_mask_hand_case():
    pts = np.array([
        [1, 5],  # nondominated
        [2, 2],  # nondominated
        [5, 1],  # nondominated
        [3, 3],  # dominated by [2, 2]
        [2, 2],  # exact duplicate of a nondominated point: kept
        [6, 6],  # dominated
    ])
    assert nondominated_mask(pts).tolist() == [True, True, True, False, True, False]


def test_pareto_on_instance(default_problem):
    par = pareto_analysis(default_problem)
    nd3 = par["nd3_mask"]
    objs = np.column_stack([-par["R"], par["H"], par["F"]])
    for i in range(len(objs)):  # brute-force check of the mask
        dominated = any(np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]) for j in range(len(objs)))
        assert nd3[i] == (not dominated)
    # with strictly positive weights the weighted optimum is nondominated
    E = default_problem.energy_qubo(par["X"])
    assert nd3[int(np.argmin(E))]
