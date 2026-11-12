"""QUBO -> Ising mapping.

With x_i = (1 - Z_i) / 2 (|0> -> x_i = 0, |1> -> x_i = 1, Z eigenvalues +1 / -1):

    E(x) = c0 + sum_i q_i x_i + sum_{i<j} Q_ij x_i x_j
         = offset + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

    J_ij   = Q_ij / 4
    h_i    = -q_i / 2 - sum_{j != i} Q_ij / 4          (Q taken symmetric over the pair)
    offset = c0 + sum_i q_i / 2 + sum_{i<j} Q_ij / 4
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .objective import Problem, all_basis_plans


@dataclass
class Ising:
    offset: float
    h: np.ndarray  # (n,)
    J: np.ndarray  # (n, n) strictly upper triangular

    @property
    def n(self) -> int:
        return int(self.h.shape[0])

    def energies_from_z(self, Z: np.ndarray) -> np.ndarray:
        """Z: (m, n) array of +/-1 eigenvalues."""
        Z = np.atleast_2d(Z).astype(float)
        return self.offset + Z @ self.h + np.einsum("mi,ij,mj->m", Z, self.J, Z)

    def diagonal(self) -> np.ndarray:
        """Energy of every computational basis state, indexed little-endian
        (bit i of the index = qubit i), INCLUDING the constant offset."""
        X = all_basis_plans(self.n)
        return self.energies_from_z(1 - 2 * X.astype(float))

    def to_sparse_pauli_op(self, include_offset: bool = True) -> SparsePauliOp:
        terms = []
        n = self.n
        if include_offset and self.offset != 0.0:
            terms.append(("", [], self.offset))
        for i in range(n):
            if self.h[i] != 0.0:
                terms.append(("Z", [i], float(self.h[i])))
        for i in range(n):
            for j in range(i + 1, n):
                if self.J[i, j] != 0.0:
                    terms.append(("ZZ", [i, j], float(self.J[i, j])))
        return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify(atol=0.0)


def qubo_to_ising(c0: float, q: np.ndarray, Q: np.ndarray) -> Ising:
    Q = np.triu(np.asarray(Q, dtype=float), 1)
    Qs = Q + Q.T
    J = Q / 4.0
    h = -np.asarray(q, dtype=float) / 2.0 - Qs.sum(axis=1) / 4.0
    offset = float(c0 + np.sum(q) / 2.0 + Q.sum() / 4.0)
    return Ising(offset=offset, h=h, J=J)


def problem_to_ising(pb: Problem) -> Ising:
    return qubo_to_ising(pb.c0, pb.q, pb.Q)
