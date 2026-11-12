"""Fast end-to-end smoke test: scenario -> QUBO -> Ising -> QAOA circuit -> samples -> comparison."""

from floodcut.benchmark import quick_compare
from floodcut.config import QAOAConfig, ScenarioConfig, Weights


def test_end_to_end_quick():
    inst = quick_compare(ScenarioConfig(), 3, Weights(), 0.5, QAOAConfig(p=1, maxiter=20, shots=512, seed=0),
                         log=lambda *a: None)
    q = inst["qaoa"][0]
    assert q["feasible_rate"] == 1.0
    assert q["leakage_outside_k_subspace"] < 1e-12
    assert q["n_evaluations"] <= 21
    assert inst["exact"]["rank"] == 1
    assert sum(q["counts"].values()) == 512
    for row in [inst["exact"], inst["greedy"], inst["annealing"][0], q["most_frequent"]]:
        assert len(row["plan"]) == 3
        assert isinstance(row["meets_threshold"], bool)
