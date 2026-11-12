import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from floodcut.classical import exact_solve
from floodcut.config import QAOAConfig
from floodcut.objective import all_basis_plans, bitstring_to_plan, plan_to_bitstring
from floodcut.qaoa import (FixedKQAOA, analyze_counts, append_dicke_state, append_mixer, initial_positions,
                           ring_edge_layers, run_qaoa)

PX = np.array([[0, 1], [1, 0]])
PY = np.array([[0, -1j], [1j, 0]])


def test_pair_exchange_is_exact_exponential():
    b = 0.83
    qc = QuantumCircuit(2)
    qc.rxx(b, 0, 1)
    qc.ryy(b, 0, 1)
    target = expm(-1j * b / 2 * (np.kron(PX, PX) + np.kron(PY, PY)))
    assert np.allclose(Operator(qc).data, target)


def test_ring_product_is_not_full_exponential():
    """Documented distinction: the parity-ordered product != exp of the full ring XY Hamiltonian."""
    n, b = 4, 0.9
    qc = QuantumCircuit(n)
    append_mixer(qc, b, n)
    H = np.zeros((2**n, 2**n), dtype=complex)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        for P in (PX, PY):
            ops = [np.eye(2)] * n
            ops[i], ops[j] = P, P
            M = ops[0]
            for o in ops[1:]:
                M = np.kron(o, M)  # little-endian ordering
            H += M
    full = expm(-1j * b / 2 * H)
    assert not np.allclose(Operator(qc).data, full, atol=1e-6)


@pytest.mark.parametrize("n", [4, 5, 10, 12])
def test_ring_edges_cover_ring(n):
    edges = [e for layer in ring_edge_layers(n) for e in layer]
    assert sorted(tuple(sorted(e)) for e in edges) == sorted(tuple(sorted((i, (i + 1) % n))) for i in range(n))
    for layer in ring_edge_layers(n):  # gates inside a layer act on disjoint qubits
        qs = [q for e in layer for q in e]
        assert len(qs) == len(set(qs))


def test_mixer_conserves_hamming_weight():
    n = 8
    beta = Parameter("b")
    qc = QuantumCircuit(n)
    append_mixer(qc, beta, n, reps=2)
    U = Operator(qc.assign_parameters({beta: 1.234})).data
    hw = all_basis_plans(n).sum(axis=1)
    # U must be block diagonal in Hamming weight
    assert np.max(np.abs(U[np.not_equal.outer(hw, hw)])) < 1e-12


@pytest.mark.parametrize("n,k", [(n, k) for n in (3, 6, 10, 12) for k in range(1, n)])
def test_dicke_state_is_uniform_over_k_subspace(n, k):
    qc = QuantumCircuit(n)
    append_dicke_state(qc, n, k)
    probs = Statevector(qc).probabilities()
    hw = all_basis_plans(n).sum(axis=1)
    target = (hw == k) / math.comb(n, k)
    assert np.max(np.abs(probs - target)) < 1e-12


@pytest.mark.parametrize("init", ["dicke", "basis"])
def test_no_leakage_and_cached_equals_full(default_problem, init):
    s = FixedKQAOA(default_problem, QAOAConfig(p=2, init_state=init))
    rng = np.random.default_rng(1)
    for _ in range(3):
        th = rng.uniform(0, 2 * np.pi, len(s.params))
        assert s.leakage(th) < 1e-12
        assert np.max(np.abs(s.probabilities(th) - s.probabilities_full_circuit(th))) < 1e-12


def test_initial_positions():
    assert initial_positions(10, 3) == [0, 3, 7]
    assert len(initial_positions(12, 5)) == 5


def test_basis_start_without_premixer_ignores_gamma(default_problem):
    """Documented limitation: |x0> is an eigenstate of U_C, so at p=1 gamma_1 has no effect."""
    s = FixedKQAOA(default_problem, QAOAConfig(p=1, init_state="basis", pre_mixer=False))
    p1 = s.probabilities([0.3, 0.7])
    p2 = s.probabilities([2.1, 0.7])
    assert np.max(np.abs(p1 - p2)) < 1e-12


@pytest.mark.parametrize("init", ["dicke", "basis"])
def test_parameters_change_distribution(default_problem, init):
    s = FixedKQAOA(default_problem, QAOAConfig(p=1, init_state=init))
    th = np.full(len(s.params), 0.6)
    tvs = []
    for delta in (0.5, 1.0, 1.5, 2.0):
        th2 = th.copy()
        th2[-2] += delta  # change gamma_1 only
        tvs.append(0.5 * np.abs(s.probabilities(th) - s.probabilities(th2)).sum())
    assert max(tvs) > 0.05  # total-variation distance: the cost angle genuinely matters


def test_expectation_matches_operator(default_problem):
    s = FixedKQAOA(default_problem, QAOAConfig(p=1))
    th = [0.4, 0.9]
    sv = Statevector(s.bind(th))
    op = s.ising.to_sparse_pauli_op()
    assert abs(sv.expectation_value(op).real - s.expectation(th)) < 1e-10


def test_sampler_endianness_and_decoding():
    n = 10
    x = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
    qc = QuantumCircuit(n)
    for i in np.flatnonzero(x):
        qc.x(int(i))
    qc.measure_all()
    counts = StatevectorSampler(default_shots=50, seed=0).run([qc]).result()[0].data.meas.get_counts()
    assert list(counts) == [plan_to_bitstring(x)]
    np.testing.assert_array_equal(bitstring_to_plan(next(iter(counts))), x)


def test_reported_plans_come_from_samples(default_problem):
    ex = exact_solve(default_problem)
    res = run_qaoa(default_problem, QAOAConfig(p=1, maxiter=15, shots=256, seed=3), ex)
    keys = set(res["counts"])
    a = res["analysis"]
    assert a["most_frequent"]["bitstring"] in keys
    assert plan_to_bitstring(np.isin(np.arange(10), a["best_observed"]["plan"]).astype(int)) in keys
    assert res["displayed_plan"] == a["most_frequent"]["plan"]
    assert a["most_frequent"]["count"] == max(res["counts"].values())
    assert sum(res["counts"].values()) == 256
    assert a["feasible_rate"] == 1.0


def test_sampling_is_seeded(default_problem):
    s = FixedKQAOA(default_problem, QAOAConfig(p=1, shots=300, seed=5))
    th = [0.5, 1.1]
    assert s.sample(th)["counts"] == s.sample(th)["counts"]


def test_analyze_counts_flags_infeasible(default_problem):
    counts = {"0000000111": 5, "0000001111": 9}  # second key has weight 4 (infeasible for k=3)
    a = analyze_counts(default_problem, counts)
    assert a["most_frequent"]["feasible"] is False
    assert a["best_observed"]["plan"] == [0, 1, 2]
    assert abs(a["feasible_rate"] - 5 / 14) < 1e-12
