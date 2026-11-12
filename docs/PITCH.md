# FloodCut — 3-minute pitch

**0:00–0:30 · The decision.**
During extreme floods, authorities may have to decide where water is allowed to go. Protecting one place shifts risk
onto others, and the options interact. FloodCut turns a **fully synthetic** version of this into an optimization
problem: pick exactly k of n candidate sites. Pakistan's 2022 floods, which affected about 33 million people
(PDNA 2022), motivate the problem. The model itself is fictional: no real places, no operational detail.

**0:30–1:10 · The math.**
Three competing objectives:
- a diversion proxy R, where nearby sites are partly redundant;
- a synthetic harm index H, where overlapping spill areas add harm;
- a district fairness penalty F on population-normalized burdens.

All three are quadratic, so the weighted objective is an exact QUBO. We checked this against the direct formulas on
every plan to about 1e−15. City safety is a threshold checked after solving. Encoding it as a penalty would be
quartic, not a QUBO, and we say so rather than hiding it.

**1:10–2:00 · The quantum part.**
Fixed-k QAOA in Qiskit:
- one qubit per site;
- a Dicke-state start, the uniform superposition over all valid plans;
- an RZ/RZZ cost layer;
- a ring XY mixer built from RXX·RYY, which never leaves the k-subspace (leakage < 1e−30, 100% feasible samples).

**Finding:** the basis-state start we were originally asked to use collapses onto single relabelled states and does no
better than random. We kept it, benchmarked it, and explain it.

**2:00–2:40 · Honest results.**
Across 48 QAOA runs, depth-2 QAOA puts **6× (n=10) to 17× (n=12)** more probability on the optimum than random
sampling. Its most-frequent plan is often good but not optimal. Exact enumeration takes under a millisecond, and greedy
and annealing are fast too. Greedy misses on n=12, and annealing fails at small budgets. **No quantum advantage is
claimed.** The exact Pareto set shows the trade-offs a single weighted score hides.

**2:40–3:00 · Demo and takeaways.**
Live demo: move the weights, watch the selected sites and threshold badge change, and compare QAOA with the classical
methods side by side. Replay mode refuses to show saved results that don't match your settings.
The takeaway is a verified, reproducible pipeline that shows how to encode a resource-limited, multi-objective
decision for QAOA, and how to check the result honestly.
