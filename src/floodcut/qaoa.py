"""Fixed-Hamming-weight (exactly-k) QAOA in Qiskit.

Circuit (one qubit per candidate site, |1> = selected). Two initial-state modes:

  init_state="dicke" (default)
    |0...01..1> (Hamming weight k) -> D_{n,k}: exact Dicke-state preparation
    (Baertschi & Eidenbenz 2019 split-and-cyclic-shift construction; CX, CRY and CCRY
    gates, all Hamming-weight preserving) giving the uniform superposition over all
    C(n,k) feasible plans
    -> for l = 1..p:  U_C(gamma_l) then U_M(beta_l)  -> measure all

  init_state="basis" (as originally specified)
    |x0>  (X on qubits round(j*n/k), Hamming weight k)
    -> U_M(beta_0)                         [pre-mixing layer, see note]
    -> for l = 1..p:  U_C(gamma_l) then U_M(beta_l)  -> measure all

  Finding (see report): with the basis start and shallow ring mixing, the optimizer of
  <E> tends to drive the circuit to a single permuted basis state (max plan probability
  ~1), i.e. a classical relabelling of x0 rather than a quantum superposition search.
  The "max_plan_probability" metric is reported so this is visible.

Cost layer   U_C(gamma) = exp(-i gamma H_C) with H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j,
             implemented as RZ(2 gamma h_i) and RZZ(2 gamma J_ij). The constant Ising
             offset only adds a global phase and is dropped from the circuit (it IS
             included in every reported energy).

Mixer layer  U_M(beta) = prod_{r=1..mixer_reps} prod_{layer in (even edges, odd edges)}
             prod_{(i,j) in layer} RXX(beta/reps) RYY(beta/reps) on (i, j),
             ring edges (0,1),(1,2),...,(n-1,0).
             For one pair, RXX(t) RYY(t) = exp(-i t/2 (XX + YY)) EXACTLY, because XX and
             YY commute. Each factor conserves Hamming weight, so the product does too.
             The product over the ring is a parity-ordered (Trotter-type) product, NOT
             the exact exponential exp(-i beta/2 sum_edges (XX + YY)), because
             neighbouring edges share a qubit and do not commute.

Pre-mixing note: a computational-basis initial state is an eigenstate of the diagonal
cost operator, so U_C(gamma_1) applied directly to |x0> is only a global phase. Without
the beta_0 layer the p = 1 output distribution would not depend on gamma_1 at all (this
is verified in the tests). The beta_0 layer spreads amplitude over the k-subspace first.

Parameter vector order: [gamma_1, beta_1, ..., gamma_p, beta_p] (2p values); with
init_state="basis" and pre_mixer=True: [beta_0, gamma_1, beta_1, ...] (2p + 1 values).
"""

from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from .config import QAOAConfig
from .ising import Ising, problem_to_ising
from .objective import Problem, all_basis_plans, bitstring_to_plan, plan_label, plan_to_ids

TIE_TOL = 1e-12


def initial_positions(n: int, k: int) -> list[int]:
    pos = sorted({int(round(j * n / k)) % n for j in range(k)})
    if len(pos) != k:  # cannot happen for k <= n, kept as a guard
        pos = list(range(k))
    return pos


def ring_edge_layers(n: int) -> list[list[tuple[int, int]]]:
    """Parity-ordered ring: even edges, then odd edges (and the wrap edge for odd n)."""
    if n < 2:
        return []
    if n == 2:
        return [[(0, 1)]]
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    wrap = (n - 1, 0)
    if n % 2 == 0:
        odd.append(wrap)
        return [even, odd]
    return [even, odd, [wrap]]


def append_mixer(qc: QuantumCircuit, beta, n: int, reps: int = 1) -> None:
    for _ in range(reps):
        angle = beta / reps if reps > 1 else beta
        for layer in ring_edge_layers(n):
            for i, j in layer:
                qc.rxx(angle, i, j)
                qc.ryy(angle, i, j)


def append_cost(qc: QuantumCircuit, gamma, ising: Ising, atol: float = 1e-14) -> None:
    n = ising.n
    for i in range(n):
        if abs(ising.h[i]) > atol:
            qc.rz(2.0 * float(ising.h[i]) * gamma, i)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(ising.J[i, j]) > atol:
                qc.rzz(2.0 * float(ising.J[i, j]) * gamma, i, j)


def _scs(qc: QuantumCircuit, m: int, k: int) -> None:
    """Split-and-cyclic-shift unitary SCS_{m,k} (paper indices 1..m -> qubits 0..m-1)."""
    q = lambda i: i - 1  # noqa: E731
    qc.cx(q(m - 1), q(m))
    qc.cry(2 * np.arccos(np.sqrt(1 / m)), q(m), q(m - 1))
    qc.cx(q(m - 1), q(m))
    for l in range(2, k + 1):
        qc.cx(q(m - l), q(m))
        qc.append(RYGate(2 * np.arccos(np.sqrt(l / m))).control(2, annotated=False), [q(m), q(m - l + 1), q(m - l)])
        qc.cx(q(m - l), q(m))


def append_dicke_state(qc: QuantumCircuit, n: int, k: int) -> None:
    """Prepare |D^n_k> = C(n,k)^(-1/2) sum_{|x|=k} |x> from |0^(n-k) 1^k>.

    Deterministic construction of Baertschi & Eidenbenz (2019), verified numerically in
    the tests for every n <= 12 and 1 <= k < n. Every gate preserves Hamming weight.
    """
    for i in range(n - k, n):
        qc.x(i)
    for m in range(n, k, -1):
        _scs(qc, m, k)
    for m in range(k, 1, -1):
        _scs(qc, m, m - 1)


def build_qaoa_parts(ising: Ising, k: int, p: int, pre_mixer: bool = True, mixer_reps: int = 1,
                     init_state: str = "dicke"):
    """Return (state-preparation circuit, parameterized QAOA body, ordered parameters)."""
    n = ising.n
    prep = QuantumCircuit(n, name="init")
    if init_state == "dicke":
        append_dicke_state(prep, n, k)
    elif init_state == "basis":
        for q in initial_positions(n, k):
            prep.x(q)
    else:
        raise ValueError(f"unknown init_state {init_state!r}")
    body = QuantumCircuit(n, name=f"qaoa-layers p={p}")
    params: list[Parameter] = []
    if init_state == "basis" and pre_mixer:
        b0 = Parameter("β0")
        params.append(b0)
        append_mixer(body, b0, n, mixer_reps)
    for layer in range(1, p + 1):
        g = Parameter(f"γ{layer}")
        b = Parameter(f"β{layer}")
        params += [g, b]
        append_cost(body, g, ising)
        append_mixer(body, b, n, mixer_reps)
    return prep, body, params


def build_qaoa_circuit(ising: Ising, k: int, p: int, pre_mixer: bool = True, mixer_reps: int = 1,
                       init_state: str = "dicke"):
    """Full circuit = state preparation, barrier, QAOA layers. Returns (circuit, parameters)."""
    prep, body, params = build_qaoa_parts(ising, k, p, pre_mixer, mixer_reps, init_state)
    return assemble_circuit(prep, body, p), params


def assemble_circuit(prep: QuantumCircuit, body: QuantumCircuit, p: int) -> QuantumCircuit:
    qc = QuantumCircuit(prep.num_qubits, name=f"fixed-k QAOA p={p}")
    qc.compose(prep, inplace=True)
    qc.barrier()
    qc.compose(body, inplace=True)
    return qc


def hamming_weights(n: int) -> np.ndarray:
    return all_basis_plans(n).sum(axis=1)


class FixedKQAOA:
    def __init__(self, pb: Problem, cfg: QAOAConfig | None = None):
        t0 = time.perf_counter()
        self.pb = pb
        self.cfg = cfg or QAOAConfig()
        self.ising = problem_to_ising(pb)
        self.n = pb.n
        args = (self.ising, pb.k, self.cfg.p, self.cfg.pre_mixer, self.cfg.mixer_reps, self.cfg.init_state)
        self.prep_qc, self.body_qc, self.params = build_qaoa_parts(*args)
        self.qc = assemble_circuit(self.prep_qc, self.body_qc, self.cfg.p)
        # The fixed state preparation is simulated once and cached; each objective call
        # evolves only the parameterized layers. Sampling uses the FULL circuit.
        self.init_sv = Statevector(self.prep_qc)
        # Diagonal of E over ALL 2^n basis states (little-endian index), incl. offset.
        self.E_diag = pb.energy_qubo(all_basis_plans(self.n))
        self.weights_hw = hamming_weights(self.n)
        self.history: list[float] = []
        self.setup_s = time.perf_counter() - t0

    # -- exact (statevector) quantities ---------------------------------------------
    def bind(self, theta) -> QuantumCircuit:
        return self.qc.assign_parameters(dict(zip(self.params, np.asarray(theta, dtype=float))))

    def bind_body(self, theta) -> QuantumCircuit:
        return self.body_qc.assign_parameters(dict(zip(self.params, np.asarray(theta, dtype=float))))

    def probabilities(self, theta) -> np.ndarray:
        return self.init_sv.evolve(self.bind_body(theta)).probabilities()

    def probabilities_full_circuit(self, theta) -> np.ndarray:
        """Same quantity simulated from |0...0> through the full circuit (used in tests)."""
        return Statevector(self.bind(theta)).probabilities()

    def expectation(self, theta) -> float:
        return float(self.probabilities(theta) @ self.E_diag)

    def leakage(self, theta) -> float:
        """Probability mass outside the Hamming-weight-k subspace (exact)."""
        probs = self.probabilities(theta)
        return float(probs[self.weights_hw != self.pb.k].sum())

    # -- optimization -----------------------------------------------------------------
    def optimize(self, theta0=None) -> dict:
        rng = np.random.default_rng(self.cfg.seed)
        if theta0 is None:
            theta0 = rng.uniform(0.0, np.pi, size=len(self.params))
        theta0 = np.asarray(theta0, dtype=float)
        self.history = []

        def f(theta):
            val = self.expectation(theta)
            self.history.append(val)
            return val

        t0 = time.perf_counter()
        E_init = self.expectation(theta0)
        res = minimize(f, theta0, method="COBYLA",
                       options={"maxiter": self.cfg.resolved_maxiter(), "rhobeg": self.cfg.rhobeg})
        opt_s = time.perf_counter() - t0
        best_idx = int(np.argmin(self.history))
        theta_best = np.asarray(res.x, dtype=float)
        E_best = self.expectation(theta_best)
        return {
            "theta0": theta0.tolist(),
            "theta_opt": theta_best.tolist(),
            "expected_E_initial": E_init,
            "expected_E_opt": E_best,
            "min_E_seen_in_history": float(self.history[best_idx]),
            "n_evaluations": len(self.history),
            "maxiter": self.cfg.resolved_maxiter(),
            "optimizer": "scipy COBYLA",
            "optimizer_message": str(res.message),
            "optimize_s": opt_s,
            "history": [float(v) for v in self.history],
        }

    # -- sampling ---------------------------------------------------------------------
    def sample(self, theta, shots: int | None = None, seed: int | None = None) -> dict:
        shots = shots or self.cfg.shots
        seed = self.cfg.seed if seed is None else seed
        meas = self.bind(theta).measure_all(inplace=False)
        sampler = StatevectorSampler(default_shots=shots, seed=seed)
        t0 = time.perf_counter()
        result = sampler.run([meas]).result()
        counts = result[0].data.meas.get_counts()
        return {"counts": dict(counts), "shots": shots, "sampler_seed": seed, "sample_s": time.perf_counter() - t0}

    # -- circuit metrics --------------------------------------------------------------
    def circuit_metrics(self, theta) -> dict:
        bound = self.bind(theta)
        logical_ops = {k: int(v) for k, v in bound.count_ops().items() if k != "barrier"}
        two_q_logical = sum(1 for inst in bound.data if inst.operation.num_qubits == 2)
        three_q_logical = sum(1 for inst in bound.data if inst.operation.num_qubits == 3)
        tqc = transpile(bound, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=11)
        t_ops = {k: int(v) for k, v in tqc.count_ops().items() if k != "barrier"}
        tprep = transpile(self.prep_qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=11)
        tbody = transpile(self.bind_body(theta), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                          seed_transpiler=11)
        return {
            "n_qubits": self.n,
            "n_parameters": len(self.params),
            "logical_depth": int(bound.depth()),
            "logical_ops": logical_ops,
            "logical_two_qubit_gates": int(two_q_logical),
            "logical_three_qubit_gates": int(three_q_logical),
            "transpiled_basis": ["rz", "sx", "x", "cx"],
            "transpiled_depth": int(tqc.depth()),
            "transpiled_ops": t_ops,
            "transpiled_cx": int(t_ops.get("cx", 0)),
            "init_state": self.cfg.init_state,
            "prep_transpiled_cx": int(tprep.count_ops().get("cx", 0)),
            "prep_transpiled_depth": int(tprep.depth()),
            "layers_transpiled_cx": int(tbody.count_ops().get("cx", 0)),
            "layers_transpiled_depth": int(tbody.depth()),
            "note": "all-to-all logical connectivity assumed (no hardware coupling map / SWAP routing)",
        }


def analyze_counts(pb: Problem, counts: dict, exact: dict | None = None) -> dict:
    """Decode counts (little-endian keys) and summarize. Plans come ONLY from samples."""
    shots = int(sum(counts.values()))
    rows = []
    for bitstring, c in counts.items():
        x = bitstring_to_plan(bitstring)
        feas = int(x.sum()) == pb.k
        E = float(pb.energy_qubo(x)[0])
        rows.append({"bitstring": bitstring, "plan": list(plan_to_ids(x)), "label": plan_label(x),
                     "count": int(c), "feasible": feas, "E": E})
    rows.sort(key=lambda r: (-r["count"], r["E"]))
    feasible_rows = [r for r in rows if r["feasible"]]
    feasible_shots = sum(r["count"] for r in feasible_rows)
    most_frequent = rows[0]
    best_observed = min(feasible_rows, key=lambda r: (r["E"], -r["count"])) if feasible_rows else None
    out = {
        "shots": shots,
        "distinct_bitstrings": len(rows),
        "feasible_rate": feasible_shots / shots if shots else 0.0,
        "most_frequent": most_frequent,
        "best_observed": best_observed,
        "top": rows[:10],
    }
    if exact is not None:
        opt = {tuple(p) for p in exact["optimal_plans"]}
        opt_shots = sum(r["count"] for r in rows if tuple(r["plan"]) in opt)
        out["empirical_p_opt"] = opt_shots / shots if shots else 0.0
        out["optimum_observed"] = opt_shots > 0
    return out


def exact_p_opt(solver: FixedKQAOA, theta, exact: dict) -> float:
    probs = solver.probabilities(theta)
    E = solver.E_diag
    mask = (solver.weights_hw == solver.pb.k) & (E <= exact["E_star"] + TIE_TOL)
    return float(probs[mask].sum())


def distribution_over_plans(solver: FixedKQAOA, theta) -> dict:
    """Exact probability of every feasible plan (for display / analysis)."""
    probs = solver.probabilities(theta)
    X = all_basis_plans(solver.n)
    out = {}
    for idx in np.flatnonzero(solver.weights_hw == solver.pb.k):
        out[plan_label(X[idx])] = float(probs[idx])
    return out


def run_qaoa(pb: Problem, cfg: QAOAConfig, exact: dict | None = None, keep_history: bool = True) -> dict:
    """Full live pipeline: build -> optimize (exact statevector) -> sample (shots) -> decode."""
    solver = FixedKQAOA(pb, cfg)
    opt = solver.optimize()
    theta = np.array(opt["theta_opt"])
    smp = solver.sample(theta)
    analysis = analyze_counts(pb, smp["counts"], exact)
    metrics = solver.circuit_metrics(theta)
    leak = solver.leakage(theta)
    probs = solver.probabilities(theta)
    res = {
        "method": "qaoa",
        "config": cfg.to_dict(),
        "setup_s": solver.setup_s,
        **{k: v for k, v in opt.items() if keep_history or k != "history"},
        "sample_s": smp["sample_s"],
        "counts": smp["counts"],
        "shots": smp["shots"],
        "sampler_seed": smp["sampler_seed"],
        "analysis": analysis,
        "circuit": metrics,
        "leakage_outside_k_subspace": leak,
        "max_plan_probability": float(probs.max()),
        "displayed_plan": analysis["most_frequent"]["plan"],
        "displayed_plan_rule": "most frequent sampled bitstring",
    }
    if exact is not None:
        p_opt = exact_p_opt(solver, theta, exact)
        res["exact_p_opt"] = p_opt
        res["uniform_p_opt"] = len(exact["optimal_plans"]) / len(exact["E_all"])
        res["p_see_opt_within_m_shots"] = {str(m): 1 - (1 - p_opt) ** m for m in (1, 10, 100)}
        res["uniform_feasible_mean_E"] = exact["E_mean_uniform_feasible"]
    return res
