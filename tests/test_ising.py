import numpy as np
from qiskit.quantum_info import Statevector

from floodcut.ising import problem_to_ising, qubo_to_ising
from floodcut.objective import all_basis_plans


def test_ising_diagonal_matches_qubo_including_offset(problem_variants):
    pb = problem_variants
    ising = problem_to_ising(pb)
    np.testing.assert_allclose(ising.diagonal(), pb.energy_qubo(all_basis_plans(pb.n)), atol=1e-10, rtol=0)


def test_sparse_pauli_op_expectation_matches(default_problem):
    pb = default_problem
    ising = problem_to_ising(pb)
    op = ising.to_sparse_pauli_op()
    diag = pb.energy_qubo(all_basis_plans(pb.n))
    rng = np.random.default_rng(0)
    for _ in range(3):
        amp = rng.normal(size=2**pb.n) + 1j * rng.normal(size=2**pb.n)
        sv = Statevector(amp / np.linalg.norm(amp))
        assert abs(sv.expectation_value(op).real - sv.probabilities() @ diag) < 1e-10
    # computational basis states |x>, x given as a little-endian index
    for idx in (0, 7, 2**pb.n - 1, 0b1000000101):
        sv = Statevector.from_int(idx, 2**pb.n)
        assert abs(sv.expectation_value(op).real - diag[idx]) < 1e-10


def test_small_hand_example():
    # E = 1 + 2 x0 - 3 x1 + 4 x0 x1
    ising = qubo_to_ising(1.0, np.array([2.0, -3.0]), np.array([[0, 4.0], [0, 0]]))
    for x0 in (0, 1):
        for x1 in (0, 1):
            z = np.array([[1 - 2 * x0, 1 - 2 * x1]])
            assert abs(ising.energies_from_z(z)[0] - (1 + 2 * x0 - 3 * x1 + 4 * x0 * x1)) < 1e-12
