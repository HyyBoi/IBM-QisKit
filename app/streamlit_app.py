"""FloodCut interactive demo (Streamlit).

Run:  streamlit run app/streamlit_app.py

Every number shown is produced by the shared `floodcut` package: either a live run of
the actual solvers, or a replay of a saved benchmark JSON whose settings and scenario
fingerprint match the current sidebar settings exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from floodcut.benchmark import quick_compare
from floodcut.config import DEFAULT_K, DEFAULT_TAU, QAOAConfig, ScenarioConfig, Weights
from floodcut.objective import build_problem
from floodcut.pareto import pareto_analysis
from floodcut.plotting import METHOD_COLORS, burden_figure, counts_figure, map_figure, pareto_figure, plt
from floodcut.results_io import load_json, settings_match
from floodcut.scenario import DISTRICT_LABELS, generate_scenario

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "results" / "benchmark_full.json"

st.set_page_config(page_title="FloodCut demo", layout="wide")


# --------------------------------------------------------------------------- caching
@st.cache_resource(show_spinner=False)
def get_problem(seed: int, n: int, k: int, wH: float, wF: float, wR: float, tau: float):
    sc = generate_scenario(ScenarioConfig(seed=seed, n_sites=n))
    pb = build_problem(sc, k, Weights(wH, wF, wR), tau)
    return sc, pb, pareto_analysis(pb)


@st.cache_data(show_spinner=False)
def get_bench(path: str, mtime: float):
    return load_json(path)


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("FloodCut")
st.sidebar.caption("Synthetic, educational prototype. Not a flood-management tool.")
mode = st.sidebar.radio("Mode", ["Live quantum run", "Replay saved benchmark"], key="mode")

st.sidebar.subheader("Scenario & objective")
seed = st.sidebar.number_input("Scenario seed", 0, 99, ScenarioConfig().seed, key="seed")
n = st.sidebar.selectbox("Candidate sites n", [10, 12], key="n")
k = st.sidebar.slider("Sites to select k", 2, 5, DEFAULT_K, key="k")
wd = Weights()
wH = st.sidebar.slider("w_H  (synthetic harm index)", 0.0, 3.0, wd.w_H, 0.05, key="wH")
wF = st.sidebar.slider("w_F  (district fairness penalty)", 0.0, 3.0, wd.w_F, 0.05, key="wF")
wR = st.sidebar.slider("w_R  (diversion proxy)", 0.0, 3.0, wd.w_R, 0.05, key="wR")
tau = st.sidebar.slider("Synthetic threshold τ on diversion proxy", 0.2, 0.8, DEFAULT_TAU, 0.01, key="tau",
                        help="Checked AFTER solving. The weighted objective does not guarantee it.")

st.sidebar.subheader("QAOA")
init = st.sidebar.selectbox("Initial state", ["dicke", "basis"], key="init",
                            help="dicke: exact uniform superposition over all k-subsets (default). "
                                 "basis: weight-k basis state + β0 pre-mixing layer (as originally specified).")
p = st.sidebar.selectbox("Depth p", [1, 2], key="p")
opt_seed = st.sidebar.number_input("Optimizer / sampler seed", 0, 99, 0, key="opt_seed")
maxiter = st.sidebar.number_input("COBYLA max evaluations", 10, 600, 150 if p == 1 else 300, 10, key=f"maxiter_{p}")
shots = st.sidebar.select_slider("Shots", [256, 512, 1024, 2048, 4096, 8192], 4096, key="shots")
sa_seed = st.sidebar.number_input("Annealing seed", 0, 99, 0, key="sa_seed")

sc, pb, par = get_problem(int(seed), int(n), int(k), float(wH), float(wF), float(wR), float(tau))
current = pb.settings_dict()

# --------------------------------------------------------------------------- header
st.title("FloodCut — choosing k hypothetical diversion sites with fixed-k QAOA")
st.warning(
    "**Synthetic hackathon model.** Fictional river, city, villages and districts in normalized units. "
    "Scores are a *diversion proxy*, a *synthetic harm index* and a *district fairness penalty* — not lives, "
    "flood depths or real locations. No hydrodynamic calibration; no guarantee of safety. "
    "Any real-world use would require expert, community and regulatory review.",
    icon="⚠️",
)


# --------------------------------------------------------------------------- result source
def qaoa_cfg() -> QAOAConfig:
    return QAOAConfig(p=int(p), seed=int(opt_seed), maxiter=int(maxiter), shots=int(shots), init_state=init)


inst, qrow, source_note = None, None, ""
if mode == "Live quantum run":
    key = (tuple(sorted(current.items(), key=str)).__repr__(), repr(qaoa_cfg().to_dict()), int(sa_seed))
    run = st.button("▶ Run live (exact, greedy, annealing, QAOA)", type="primary")
    if run:
        with st.spinner("Optimizing circuit parameters on the statevector simulator, then sampling…"):
            res = quick_compare(ScenarioConfig(seed=int(seed), n_sites=int(n)), int(k), Weights(wH, wF, wR), float(tau),
                                qaoa_cfg(), sa_seed=int(sa_seed), log=lambda *a: None)
        st.session_state["live"] = {"key": key, "inst": res}
    live = st.session_state.get("live")
    if live and live["key"] == key:
        inst = live["inst"]
        qrow = inst["qaoa"][0]
        source_note = "Live run (this session) — results computed just now by the shared solver code."
    elif live:
        st.info("Settings changed since the last live run. Press **Run live** to recompute; the old results are hidden.")
    else:
        st.info("Press **Run live** to run all four methods on the current settings (≈2–20 s).")
else:
    if not BENCH_PATH.exists():
        st.error(f"No saved benchmark found at `{BENCH_PATH.relative_to(ROOT)}`. Run `python -m floodcut bench --preset full`.")
    else:
        bench = get_bench(str(BENCH_PATH), BENCH_PATH.stat().st_mtime)
        match, mism = None, {}
        for cand in bench["instances"]:
            ok, bad = settings_match(cand["settings"], current)
            if ok:
                match = cand
                break
            mism[f"seed {cand['settings']['scenario_seed']}, n={cand['settings']['n']}"] = bad
        with st.expander("Provenance of saved results", expanded=False):
            st.json(bench["provenance"])
            st.write(f"Preset `{bench['preset']}` — {bench['completed_qaoa_runs']} QAOA runs, "
                     f"{bench['completed_sa_runs']} annealing runs, total {bench['total_runtime_s']:.0f} s.")
        if match is None:
            st.error("**Saved results do not match the current settings**, so nothing is replayed. "
                     "Saved instances use the default weights (1, 0.5, 1), k = 3, τ = 0.5 and scenario seeds "
                     + ", ".join(sorted({f"{i['settings']['scenario_seed']} (n={i['settings']['n']})" for i in bench['instances']}))
                     + ". Mismatching keys: " + "; ".join(f"{a}: {', '.join(b)}" for a, b in mism.items()))
            def _to_live():
                st.session_state["mode"] = "Live quantum run"

            st.button("Switch to a live run with these settings", on_click=_to_live)
        else:
            runs = match["qaoa"]
            avail = sorted({(r["config"]["init_state"], r["config"]["p"], r["config"]["seed"]) for r in runs})
            want = (init, int(p), int(opt_seed))
            if want not in avail:
                st.error(f"No saved QAOA run with init={init}, p={p}, seed={opt_seed}. Saved runs: "
                         + ", ".join(f"{a}/p={b}/seed={c}" for a, b, c in avail) + ". Adjust the QAOA settings or run live.")
            else:
                inst = match
                qrow = next(r for r in runs if (r["config"]["init_state"], r["config"]["p"], r["config"]["seed"]) == want)
                cfg = qrow["config"]
                if (cfg["maxiter"], cfg["shots"]) != (int(maxiter), int(shots)):
                    st.info(f"Replaying the saved run's own budget (max evaluations {cfg['maxiter']}, {cfg['shots']} shots); "
                            "the sidebar budget/shot settings apply to live runs only.")
                source_note = (f"Replay of saved benchmark ({bench['provenance']['created_utc']}, qiskit "
                               f"{bench['provenance']['qiskit']}). Scenario fingerprint verified: "
                               f"`{match['settings']['scenario_fingerprint'][:16]}…`")

# --------------------------------------------------------------------------- scenario info (always)
v = inst["validation"] if inst else None
col_map, col_info = st.columns([1.05, 1])

if inst is None:
    with col_map:
        st.pyplot(map_figure(sc, [], f"Synthetic scenario (seed {seed}, n={n})", show_all_spills=True), clear_figure=True)
    with col_info:
        st.subheader("Synthetic coefficients")
        st.dataframe(pd.DataFrame(sc.coefficient_table()), hide_index=True, width="stretch")
        st.caption("; ".join(DISTRICT_LABELS.values()))
    st.stop()

st.caption(source_note)


def method_rows():
    q_mf = qrow["most_frequent"]
    q_best = qrow["best_observed"]
    rows = [("Exact enumeration", inst["exact"], f"{inst['exact']['evaluations']} plans"),
            ("QAOA — most frequent sample (displayed)", q_mf, f"{qrow['n_evaluations']} circuit evals + {qrow['shots']} shots"),
            ("QAOA — best observed sample*", q_best, "classical evaluation of samples"),
            ("Greedy (marginal ΔE)", inst["greedy"], f"{inst['greedy']['evaluations']} marginals"),
            (f"Annealing (seed {inst['annealing'][0]['seed']})", inst["annealing"][0], f"{inst['annealing'][0]['evaluations']} evals")]
    out = []
    for name, r, budget in rows:
        out.append({
            "method": name,
            "plan": r["label"],
            "E (lower better)": round(r["E"], 4),
            "diversion proxy R": round(r["R"], 4),
            "synthetic harm H": round(r["H"], 4),
            "fairness penalty F": round(r["F"], 5),
            "threshold": "meets τ" if r["meets_threshold"] else "BELOW τ",
            "gap to exact": round(r["additive_gap"], 4),
            "rank": f"{r['rank']}/{r['n_feasible']}",
            "budget": budget,
        })
    return pd.DataFrame(out)


with col_info:
    n_meet, n_feas = v["n_meet_threshold"], v["n_feasible"]
    if v["no_plan_meets_threshold"]:
        st.error(f"No feasible plan reaches the synthetic threshold τ = {tau:.2f} (0 of {n_feas}).")
    else:
        st.write(f"**Synthetic threshold τ = {tau:.2f}:** {n_meet} of {n_feas} feasible plans reach it "
                 f"(diversion proxy R ≥ τ). It is checked after solving; plans below it are marked **BELOW τ**.")
    st.dataframe(method_rows(), hide_index=True, width="stretch")
    st.caption("*Best observed = lowest E among sampled plans, found by evaluating every sample classically. With many "
               "shots over few plans this approaches exhaustive search, so it is not evidence of quantum performance. "
               "The demo therefore displays the most frequent sample as the QAOA plan.")

with col_map:
    choice = st.radio("Show plan on map", ["Exact", "QAOA", "Greedy", "Annealing"], horizontal=True)
    plan_of = {"Exact": inst["exact"], "QAOA": qrow["most_frequent"], "Greedy": inst["greedy"], "Annealing": inst["annealing"][0]}
    r = plan_of[choice]
    badge = "meets τ" if r["meets_threshold"] else "BELOW τ"
    st.pyplot(map_figure(sc, r["plan"], f"{choice}: {r['label']}  —  R={r['R']:.3f} ({badge}), H={r['H']:.2f}"),
              clear_figure=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Pareto view (exact, all feasible plans)")
    overlays = [
        {"R": inst["exact"]["R"], "H": inst["exact"]["H"], "marker": "o", "s": 140, "color": METHOD_COLORS["exact"], "label": "exact weighted optimum"},
        {"R": qrow["most_frequent"]["R"], "H": qrow["most_frequent"]["H"], "marker": "*", "s": 200, "color": METHOD_COLORS["qaoa"], "label": "QAOA most frequent"},
    ]
    st.pyplot(pareto_figure(par["R"], par["H"], par["F"], par["nd3_mask"], par["nd2_mask"], overlays, highlight=tau),
              clear_figure=True)
    st.caption(f"{par['n_nondominated_3obj']} plans are nondominated over (R↑, H↓, F↓); {par['n_frontier_2obj']} lie on the "
               f"2-objective (R, H) frontier shown as a step line. A weighted sum only reaches *supported* points — "
               f"{par['n_unsupported_3obj_grid']} nondominated plans were never selected on a dense weight grid.")
with c2:
    st.subheader("District burden (additive proxy)")
    st.pyplot(burden_figure({"exact": inst["exact"]["burdens"], "qaoa": qrow["most_frequent"]["burdens"],
                             "greedy": inst["greedy"]["burdens"], "annealing": inst["annealing"][0]["burdens"]}),
              clear_figure=True)
    st.caption("b_d = Σ_i a_di x_i / P_d. Fairness F = Σ_{d<e}(b_d − b_e)². F is zero when all districts carry the *same* "
               "burden, even a high one — read it together with H.")

st.subheader("Quantum circuit panel")
cm = qrow["circuit"]
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("qubits", cm["n_qubits"])
m2.metric("parameters", cm["n_parameters"])
m3.metric("logical depth", cm["logical_depth"])
m4.metric("transpiled depth", cm["transpiled_depth"])
m5.metric("CX (prep + layers)", f"{cm['prep_transpiled_cx']} + {cm['layers_transpiled_cx']}")
m6.metric("feasible samples", f"{qrow['feasible_rate']:.1%}")
m1.metric("P(opt) exact", f"{qrow['exact_p_opt']:.3f}")
m2.metric("P(opt) uniform", f"{qrow['uniform_p_opt']:.3f}")
m3.metric("P(opt) sampled", f"{qrow['empirical_p_opt']:.3f}")
m4.metric("⟨E⟩ optimized", f"{qrow['expected_E_opt']:+.3f}", f"{qrow['expected_E_opt'] - qrow['expected_E_initial']:+.3f} vs initial",
          delta_color="inverse")
m5.metric("⟨E⟩ uniform plan", f"{qrow['uniform_feasible_mean_E']:+.3f}")
m6.metric("max plan probability", f"{qrow['max_plan_probability']:.3f}")
q1, q2 = st.columns([1.3, 1])
with q1:
    opt_labels = {"{" + ", ".join(f"S{i}" for i in pl) + "}" for pl in inst["exact"]["optimal_plans"]}
    st.pyplot(counts_figure([(t["label"], t["count"]) for t in qrow["top_samples"]], next(iter(opt_labels))), clear_figure=True)
with q2:
    st.markdown(
        f"- Initial state: **{cm['init_state']}**; depth p = {qrow['config']['p']}; ring XY mixer (RXX·RYY, parity order)\n"
        f"- COBYLA evaluations: {qrow['n_evaluations']} (budget {qrow['maxiter']}); exact ⟨E⟩ from the statevector\n"
        f"- Shots: {qrow['shots']} (StatevectorSampler, seed {qrow['sampler_seed']}); distinct plans sampled: {qrow['distinct_bitstrings']}\n"
        f"- Probability outside the k-subspace: {qrow['leakage_outside_k_subspace']:.1e} (ideal simulation)\n"
        f"- Timing: setup {qrow['setup_s']:.2f} s, optimization {qrow['optimize_s']:.2f} s, sampling {qrow['sample_s']:.2f} s "
        f"(exact enumeration: {inst['exact']['runtime_s'] * 1e3:.1f} ms)\n"
        f"- P(seeing the optimum within m shots): "
        + ", ".join(f"m={m_}: {v_:.2f}" for m_, v_ in qrow["p_see_opt_within_m_shots"].items())
    )
    if qrow["max_plan_probability"] > 0.9:
        st.warning("The circuit has collapsed onto essentially one plan (a relabelled basis state). This is a known "
                   "failure mode of the basis-state start with shallow ring mixing; see the report.")
    st.caption("No quantum speedup or advantage is claimed: this instance is solved exactly by enumeration in milliseconds.")

with st.expander("Synthetic coefficients and validation checks"):
    st.dataframe(pd.DataFrame(sc.coefficient_table()), hide_index=True, width="stretch")
    st.json({k_: v[k_] for k_ in ("all_R_positive", "all_H_positive", "all_F_nonnegative", "city_unexposed",
                                    "objectives_compete", "qubo_max_abs_err_feasible", "n_meet_threshold", "hard_failures")})
    st.json(inst["normalizers"])
plt.close("all")
