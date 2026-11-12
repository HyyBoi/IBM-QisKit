# FloodCut

**Selecting k hypothetical flood-diversion sites with fixed-Hamming-weight QAOA in Qiskit, on a synthetic multi-objective scenario.**
Qiskit Fall Fest 2026 Hackathon · Domain: Disaster Response

**Team:** Abdul Hannan (team lead) · Dayyan Abdullah · Muhammad Dawood Ali Khan

> **Synthetic, educational prototype.** The river, city, villages, districts and all coefficients are fictional and
> expressed in normalized units. FloodCut contains no real locations, no breach coordinates, no methods for altering
> infrastructure and no hydrological model. Its scores are a *diversion proxy*, a *synthetic harm index* and a
> *district fairness penalty*. They are not lives, flood depths or damages. Nothing here validates real-world
> safety, and **no quantum speedup or advantage is claimed**.

## What it does

FloodCut chooses exactly `k` of `n` candidate sites (default 3 of 10; optional 3 of 12). It balances three objectives:

| symbol | meaning | form |
|---|---|---|
| R(x) | diversion proxy (maximize) | Σ r_i x_i − Σ_{i<j} o_ij x_i x_j (redundant diversion between nearby sites) |
| H(x) | synthetic harm index (minimize) | Σ h_i x_i + Σ_{i<j} s_ij x_i x_j (overlapping illustrative spill areas) |
| F(x) | district fairness penalty (minimize) | Σ_{d<e} (b_d − b_e)², with b_d = Σ_i a_di x_i / P_d |

The weighted, fixed-normalized objective `E = w_H·Hn + w_F·Fn − w_R·Rn` is an exact QUBO. It is mapped to an Ising
Hamiltonian and solved with **fixed-k QAOA**:

- one qubit per site;
- an exact Dicke-state start (or the weight-k basis-state start, as originally specified);
- an RZ/RZZ cost layer;
- a ring XY mixer built from RXX·RYY exchanges, which conserves Hamming weight.

The optimized circuit is then sampled with Qiskit's `StatevectorSampler`. The QAOA results are compared with exact
enumeration, a greedy heuristic and swap-move simulated annealing, all on the same `Problem` object, and with the
exact 3-objective Pareto set.

A synthetic city-safety threshold `R(x) ≥ τ` is checked **after** solving. Every plan is labelled "meets τ" or
"BELOW τ". The threshold is not encoded as a constraint, because a penalty on a quadratic inequality would be
quartic and therefore not a QUBO.

## Key results (from `results/benchmark_full.json`)

| setting | P(opt) vs uniform 1/C(n,k) | most-frequent sample optimal | max single-plan prob. |
|---|---|---|---|
| n=10, Dicke start, p=2 (9 runs) | 5.8× mean (0.9–8.9×) | 22% | 0.07 |
| n=12, Dicke start, p=2 (3 runs) | 17.3× mean | 100% | 0.08 |
| n=10, basis start + β0, p=1 (9 runs) | 0.6× | 0% | 0.82 (collapse) |

- **Feasibility and leakage:** every sample was feasible, and leakage out of the k-subspace stayed below 1e−30 in ideal simulation.
- **Basis start:** the specified start collapses onto single relabelled basis states, so the default is now a Dicke start. The report explains why.
- **Speed:** exact enumeration takes under 1 ms.
- **Greedy:** found the optimum on 3 of 4 instances and missed it on n=12.
- **Annealing:** fails at small budgets.

The full analysis is in [docs/FloodCut_Report.docx](docs/FloodCut_Report.docx).

## Setup

Tested with Python 3.14.0 on Windows 11. `requirements.txt` pins the direct dependencies, and `requirements-lock.txt` is the full freeze.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

## Commands

| task | command | time |
|---|---|---|
| run the tests (83 checks, including an end-to-end quantum smoke test and headless UI tests) | `pytest` | ~30 s |
| validate the scenario and QUBO on every feasible plan | `python -m floodcut validate` | 1 s |
| one quick live comparison of all four methods | `python -m floodcut quick` | ~3 s |
| quick benchmark preset | `python -m floodcut bench --preset quick` | ~1 s |
| full benchmark (48 QAOA + 40 SA runs, weight sweep, SA budget study) | `python -m floodcut bench --preset full` | ~100 s |
| regenerate the report figures from saved results | `python -m floodcut figures` | ~10 s |
| build the DOCX report from the saved results | `python report/build_report.py` | ~3 s |
| start the interactive demo | `streamlit run app/streamlit_app.py` | — |

`quick` accepts `--seed --n --k --wH --wF --wR --tau --p --init {dicke,basis} --maxiter --shots --opt-seed`.

## Demo

`streamlit run app/streamlit_app.py` opens two clearly separated modes:

- **Live quantum run:** runs exact, greedy, annealing and QAOA on the current sidebar settings, using the shared solver code. Results are hidden as soon as any setting changes.
- **Replay saved benchmark:** shows a saved run only when the scenario fingerprint, k, weights and τ match the sidebar exactly. Otherwise it shows a "Saved results do not match" banner and a button that switches to a live run. Provenance (timestamp and versions) is shown.

The page includes:
- the synthetic map, with selected sites and illustrative spill areas for each method;
- a side-by-side table with threshold badges;
- district burdens;
- the exact Pareto plot;
- a circuit panel showing qubits, depth, CX counts, the sample histogram, P(opt) and leakage.

## Repository layout

```
src/floodcut/   config, scenario (synthetic geometry -> coefficients), objective (R/H/F, QUBO, validation),
                ising (QUBO->Ising), qaoa (fixed-k circuit, optimize, sample, decode), classical (exact, greedy, SA),
                pareto, benchmark, results_io (provenance, replay matching), plotting, figures, __main__ (CLI)
app/            streamlit_app.py
report/         build_report.py (python-docx; all numbers read from results JSON)
results/        benchmark_full.json, benchmark_quick.json, quick_run.json, bench_full.log
docs/           FloodCut_Report.docx, figures/*.png, PITCH.md
tests/          scenario, objective/QUBO, Ising, QAOA, classical, Pareto, results I/O, end-to-end smoke, app (AppTest)
```

## Conventions worth knowing

- **Bit order:** Qiskit counts keys are little-endian, so `x_i = int(key[n-1-i])` (see `objective.bitstring_to_plan`).
- **Angles:** `U_C(γ) = exp(−iγH_C)` is applied as `RZ(2γh_i)` and `RZZ(2γJ_ij)`. Each mixer edge applies `RXX(β)·RYY(β) = exp(−iβ/2·(XX+YY))`. Parameters are ordered `[γ1, β1, …, γp, βp]`, with a leading `β0` for the basis start.
- **Mixer:** a parity-ordered product over the ring. It is *not* the exact exponential of the full ring XY Hamiltonian, and a test checks that they differ.
- **Displayed QAOA plan:** the **most frequent sample**. "Best observed" is shown only with a caveat, because with many shots over few plans it approaches exhaustive search.
- **Normalizers:** min and range over all feasible plans for the scenario and k, held fixed across weights. This needs enumeration, so it only works for small instances.

## What was not tested / limitations

- **Hardware and noise:** no runs on quantum hardware and no noise models. All results are ideal statevector simulation, and transpilation assumes all-to-all connectivity.
- **Python versions:** only Python 3.14 on Windows was tested. Other versions should work but were not run.
- **Model scope:** see report §9. In brief: no hydrodynamics, synthetic and uncertain inputs, pairwise-only interactions, a simplified fairness proxy, no safety guarantee, instances small enough to solve exactly, QAOA parameter sensitivity, and expert review required before any real-world use.

## Remaining external submission steps

1. Make sure the judges can open the repository: it is currently **private** at https://github.com/HyyBoi/IBM-QisKit, so either make it public or add them (and teammates) as collaborators.
2. Submit on the hackathon portal with the repository link and `docs/FloodCut_Report.docx`.

## License and attribution

MIT (see `LICENSE`). Built with Qiskit (Apache-2.0), NumPy, SciPy, Matplotlib, Streamlit and python-docx. The Dicke-state
preparation follows Bärtschi & Eidenbenz (2019). AI assistance (Claude) was used to write code and documentation.
All reported numbers come from the saved results produced by this code.
