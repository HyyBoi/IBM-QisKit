"""Matplotlib figures shared by the Streamlit demo and the report.

All figures are explicitly labelled SYNTHETIC. They draw the same Scenario object the
optimizer uses.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon  # noqa: E402

from .scenario import DISTRICT_NAMES, Scenario  # noqa: E402

# Categorical slots 1-4 of the reference data-viz palette, in fixed order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DISTRICT_COLORS = {"A": "#E3EEFA", "B": "#FCE9E0", "C": "#E0F4EC"}  # light tints of slots 1-3
DISTRICT_EDGE = {"A": SLOT[0], "B": SLOT[1], "C": SLOT[2]}
METHOD_COLORS = {"exact": SLOT[0], "qaoa": SLOT[1], "greedy": SLOT[2], "annealing": SLOT[3]}
SEQ_CMAP = "Blues"  # sequential, single hue (light = low)


def draw_map(ax, sc: Scenario, selected=None, title: str | None = None, show_all_spills: bool = False):
    """Draw the synthetic map. `selected` is an iterable of site indices."""
    selected = set(int(i) for i in (selected or []))
    for d, poly in sc.district_polygons.items():
        ax.add_patch(Polygon(poly, closed=True, facecolor=DISTRICT_COLORS[d], edgecolor=DISTRICT_EDGE[d], lw=1.2, alpha=0.9, zorder=0))
    label_pos = {"A": (0.04, 0.95), "B": (0.04, 0.05), "C": (0.78, 0.95)}
    for d, (x, y) in label_pos.items():
        ax.text(x, y, f"District {d}", fontsize=9, color=DISTRICT_EDGE[d], weight="bold", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # river
    hw = sc.config.river_half_width
    ax.fill_betweenx(sc.river_y, sc.river_x - hw * 1.4, sc.river_x + hw * 1.4, color="#4A90D9", alpha=0.85, zorder=2, lw=0)
    ax.text(sc.river_x[8] + 0.035, 0.975, "flow ↓", fontsize=7, color="#2C5F99", zorder=3)
    # illustrative spill areas (1 and 2 kernel-sigma contours of the same ellipse used for e_iv)
    for i in range(sc.n):
        if i in selected or show_all_spills:
            sel = i in selected
            for m, alpha in ((2.0, 0.18 if sel else 0.06), (1.0, 0.35 if sel else 0.10)):
                ax.add_patch(
                    Ellipse(
                        sc.spill_center[i],
                        width=2 * m * sc.spill_axes[i, 0],
                        height=2 * m * sc.spill_axes[i, 1],
                        angle=np.degrees(sc.spill_angle[i]),
                        facecolor="#3B7DD8" if sel else "#888888",
                        edgecolor="#1F4E8C" if sel else "none",
                        lw=0.8,
                        alpha=alpha,
                        zorder=3,
                    )
                )
    # city
    ax.add_patch(Circle(sc.city_xy, sc.config.city_radius, facecolor="#555555", edgecolor="black", alpha=0.85, zorder=4))
    ax.text(sc.city_xy[0], sc.city_xy[1], "CITY", color="white", ha="center", va="center", fontsize=8, weight="bold", zorder=5)
    # villages (marker size ~ synthetic population)
    sizes = 25 + 25 * sc.village_pop
    ax.scatter(sc.village_xy[:, 0], sc.village_xy[:, 1], s=sizes, marker="^", c="#8C6D46", edgecolors="black", linewidths=0.5, zorder=5)
    for v, (x, y) in enumerate(sc.village_xy):
        ax.text(x + 0.012, y - 0.022, f"V{v + 1:02d}", fontsize=6, color="#4A3B28", zorder=5)
    # candidate sites
    for i, (x, y) in enumerate(sc.site_xy):
        sel = i in selected
        ax.scatter([x], [y], s=110 if sel else 55, marker="X" if sel else "o",
                   c="#D62728" if sel else "white", edgecolors="black", linewidths=0.9, zorder=6)
        dx = 0.035 * sc.site_bank[i]
        ax.text(x + dx, y + 0.005, f"S{i}", fontsize=8, ha="center", va="center", weight="bold" if sel else "normal",
                zorder=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.012, "SYNTHETIC — NOT REAL GEOGRAPHY", ha="center", va="bottom", fontsize=7, color="#666666", style="italic", zorder=8)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def map_figure(sc: Scenario, selected=None, title=None, show_all_spills=False, figsize=(5.2, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_map(ax, sc, selected, title, show_all_spills)
    fig.tight_layout()
    return fig


def method_maps_figure(sc: Scenario, plans: dict, figsize=(13, 3.9)):
    """One small map per method. plans: name -> (site ids, subtitle)."""
    fig, axes = plt.subplots(1, len(plans), figsize=figsize)
    axes = np.atleast_1d(axes)
    for ax, (name, (ids, sub)) in zip(axes, plans.items()):
        draw_map(ax, sc, ids, f"{name}\n{sub}")
    fig.tight_layout()
    return fig


def burden_figure(burden_by_method: dict, figsize=(5.2, 3.0)):
    names = list(burden_by_method)
    D = len(DISTRICT_NAMES)
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=figsize)
    for j, name in enumerate(names):
        vals = burden_by_method[name]
        ax.bar(np.arange(D) + j * width, vals, width, label=name, color=METHOD_COLORS.get(name.lower().split()[0], None))
    ax.set_xticks(np.arange(D) + width * (len(names) - 1) / 2)
    ax.set_xticklabels([f"District {d}" for d in DISTRICT_NAMES])
    ax.set_ylabel("population-normalized burden b_d\n(synthetic index)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def pareto_figure(R, H, F, nd3_mask, nd2_mask, overlays=None, highlight=None, figsize=(6.4, 4.6)):
    """Relief vs harm for every feasible plan, coloured by fairness penalty.

    nd3_mask: exact 3-objective nondominated set; nd2_mask: (R,H) 2-objective frontier.
    overlays: list of dicts {R, H, label, marker, color} (e.g., QAOA weight-sweep plans).
    """
    fig, ax = plt.subplots(figsize=figsize)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(plt.get_cmap(SEQ_CMAP)(np.linspace(1.0, 0.28, 256)))  # low F = dark; never near-white
    sc_ = ax.scatter(H, R, c=F, cmap=cmap, s=24, alpha=0.9, edgecolors="none", label="all feasible plans")
    cb = fig.colorbar(sc_, ax=ax)
    cb.set_label("fairness penalty F (lower = more even)")
    ax.scatter(H[nd3_mask], R[nd3_mask], s=60, facecolors="none", edgecolors="black", linewidths=1.0,
               label="exact 3-objective nondominated")
    idx = np.flatnonzero(nd2_mask)
    order = idx[np.argsort(H[idx])]
    ax.step(H[order], R[order], where="post", color="#52514e", lw=1.5, ls=":", label="2-objective (R, H) frontier")
    if overlays:
        for o in overlays:
            ax.scatter([o["H"]], [o["R"]], marker=o.get("marker", "*"), s=o.get("s", 170), c=o.get("color", METHOD_COLORS["qaoa"]),
                       edgecolors="black", linewidths=0.6, zorder=6, label=o.get("label"))
    if highlight is not None:
        ax.axhline(highlight, color="#52514e", ls="--", lw=1, label="synthetic threshold τ")
    ax.set_xlabel("synthetic harm index H (minimize)")
    ax.set_ylabel("diversion proxy R (maximize)")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="lower right", frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def counts_figure(top_items, optimal_label=None, figsize=(6.0, 3.0)):
    """Bar chart of the most frequent sampled plans. top_items: list of (label, count)."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = [t[0] for t in top_items]
    vals = [t[1] for t in top_items]
    colors = [METHOD_COLORS["qaoa"] if lab == optimal_label else "#B8B6AE" for lab in labels]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("shots")
    ax.set_title("Most frequent sampled plans (orange = exact optimum)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def convergence_figure(histories: dict, figsize=(5.5, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    for name, hist in histories.items():
        ax.plot(np.arange(1, len(hist) + 1), hist, lw=1.2, label=name)
    ax.set_xlabel("objective evaluation")
    ax.set_ylabel("exact ⟨E⟩ (statevector)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def architecture_figure(figsize=(11, 5.6)):
    """Architecture diagram of the FloodCut pipeline."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.25, 5.6)
    ax.axis("off")

    def box(x, y, w, h, lines, fc, ec="#333333", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h - 0.16, lines[0], ha="center", va="top", fontsize=fs + 0.5, weight="bold")
        if len(lines) > 1:
            ax.text(x + w / 2, y + h - 0.45, "\n".join(lines[1:]), ha="center", va="top", fontsize=fs - 0.8, linespacing=1.3)

    def arrow(x1, y1, x2, y2, text=None, dx=0.0, dy=0.08, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.1, connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, text, ha="center", fontsize=7.5, color="#444444", style="italic",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    C_DATA, C_MATH, C_Q, C_CL, C_OUT = "#EAF2FB", "#FFF4E0", "#F3ECFB", "#E6F4EA", "#F2F2F2"
    # row 1: data -> math
    box(0.2, 4.0, 2.3, 1.25, ["Synthetic scenario", "seeded river, city, villages", "3 districts, 10–12 sites",
                              "illustrative spill ellipses"], C_DATA)
    box(3.0, 4.0, 2.4, 1.25, ["Coefficients", "r_i, o_ij (diversion proxy)", "h_i, s_ij (harm index)",
                              "a_di / P_d (district burden)"], C_DATA)
    box(5.9, 4.0, 2.5, 1.25, ["Objective → QUBO", "E = wH·Hn + wF·Fn − wR·Rn", "fixed normalizers",
                              "c0 + q·x + Σ Q_ij x_i x_j"], C_MATH)
    box(8.8, 4.0, 2.0, 1.25, ["Validation", "direct = QUBO = Ising", "on every plan (1e−10)",
                              "scenario sanity checks"], C_MATH)
    arrow(2.5, 4.62, 3.0, 4.62)
    arrow(5.4, 4.62, 5.9, 4.62)
    arrow(8.4, 4.62, 8.8, 4.62)
    # row 2: quantum path (left) and classical path (right); both feed the evaluation
    box(0.2, 1.95, 3.0, 1.45, ["Fixed-k QAOA circuit (Qiskit)", "init: Dicke |D(n,k)⟩ (or basis + U_M(β0))",
                               "[U_C(γ_l) U_M(β_l)] × p", "U_C: RZ + RZZ;  U_M: ring RXX·RYY"], C_Q)
    box(3.6, 1.95, 2.6, 1.45, ["Optimize, then sample", "exact ⟨E⟩ via Statevector", "COBYLA, bounded budget",
                               "StatevectorSampler shots"], C_Q)
    box(6.6, 1.95, 2.0, 1.45, ["Classical baselines", "exact enumeration", "greedy (marginal ΔE)",
                               "annealing (swap moves)"], C_CL)
    box(9.0, 1.95, 1.8, 1.45, ["Evaluation", "R, H, F, E, threshold τ", "gaps, ranks, P(opt)", "exact Pareto set"], C_CL)
    arrow(6.4, 4.0, 1.7, 3.4, "Ising map  x = (1 − Z)/2", dx=0.2, dy=0.05)
    arrow(3.2, 2.67, 3.6, 2.67)
    arrow(7.4, 4.0, 7.6, 3.4)
    arrow(8.6, 2.67, 9.0, 2.67)
    arrow(4.9, 1.95, 9.5, 1.95, "decoded samples (little-endian)", dy=-0.36, rad=0.14)
    # row 3: outputs (the demo and the report both read the saved JSON; the demo can also run live)
    box(7.9, 0.35, 2.7, 0.9, ["results/*.json", "provenance, fingerprints, raw counts"], C_OUT)
    box(4.4, 0.35, 2.8, 0.9, ["Streamlit demo", "live run  |  replay (verified match)"], C_OUT)
    box(0.9, 0.35, 2.8, 0.9, ["DOCX report", "figures + tables from saved results"], C_OUT)
    arrow(9.9, 1.95, 9.25, 1.25)
    arrow(7.9, 0.8, 7.2, 0.8)
    arrow(8.6, 0.35, 2.3, 0.35, rad=-0.12)
    ax.text(5.5, 5.55, "FloodCut architecture — one shared solver package used by CLI, benchmark, demo and report",
            ha="center", va="top", fontsize=10, weight="bold")
    fig.tight_layout()
    return fig
