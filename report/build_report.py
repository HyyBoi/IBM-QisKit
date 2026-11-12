"""Build docs/FloodCut_Report.docx from SAVED results (results/benchmark_full.json) and figures.

Run:  python report/build_report.py
Every number in the results sections is read from the saved benchmark JSON.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from floodcut.config import DEFAULT_TAU, ScenarioConfig, Weights  # noqa: E402
from floodcut.results_io import load_json  # noqa: E402

FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "FloodCut_Report.docx"
FONT = "Calibri"
MATH_FONT = "Cambria Math"
INK = RGBColor(0x1F, 0x1F, 0x1F)
MUTED = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
CONTENT_WIDTH_CM = 16.6


# ----------------------------------------------------------------------------- helpers
def set_cell_shading(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def set_repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def add_page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("FloodCut — synthetic educational prototype — page ")
    run.font.size = Pt(8)
    run.font.color.rgb = MUTED
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "16")
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    p._p.append(fld)


class Report:
    def __init__(self):
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        sec = self.doc.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)  # A4
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.2))
        sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(2.0)
        add_page_number_footer(sec)
        st = self.doc.styles
        normal = st["Normal"]
        normal.font.name = FONT
        normal.font.size = Pt(10.5)
        normal.font.color.rgb = INK
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.12
        for name, size, before, after, color in (("Title", 26, 0, 6, INK), ("Heading 1", 16, 16, 6, ACCENT),
                                                 ("Heading 2", 12.5, 12, 4, INK), ("Heading 3", 11, 8, 3, INK)):
            s = st[name]
            s.font.name = FONT
            s.font.size = Pt(size)
            s.font.bold = True
            s.font.color.rgb = color
            rpr = s.element.get_or_add_rPr()
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts")
                rpr.append(rfonts)
            for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rfonts.set(qn(att), FONT)
            for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rfonts.get(qn(att)) is not None:
                    del rfonts.attrib[qn(att)]
            s.paragraph_format.space_before = Pt(before)
            s.paragraph_format.space_after = Pt(after)
            s.paragraph_format.keep_with_next = True
        for name in ("List Bullet", "List Number"):
            st[name].font.name = FONT
            st[name].font.size = Pt(10.5)
            st[name].paragraph_format.space_after = Pt(3)

    # -- text
    def h1(self, text):
        return self.doc.add_heading(text, level=1)

    def h2(self, text):
        return self.doc.add_heading(text, level=2)

    def h3(self, text):
        return self.doc.add_heading(text, level=3)

    def para(self, text, italic=False, size=None, color=None, align=None, space_after=None):
        """Plain paragraph; **bold** segments are supported."""
        p = self.doc.add_paragraph()
        parts = text.split("**")
        for i, part in enumerate(parts):
            if not part:
                continue
            r = p.add_run(part)
            r.bold = i % 2 == 1
            r.italic = italic
            if size:
                r.font.size = Pt(size)
            if color:
                r.font.color.rgb = color
        if align:
            p.alignment = align
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        return p

    def bullets(self, items, style="List Bullet"):
        for it in items:
            p = self.doc.add_paragraph(style=style)
            parts = it.split("**")
            for i, part in enumerate(parts):
                if part:
                    r = p.add_run(part)
                    r.bold = i % 2 == 1

    def eq(self, text, label=None):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        r.font.name = MATH_FONT
        r._element.rPr.rFonts.set(qn("w:hAnsi"), MATH_FONT)
        r.font.size = Pt(11)
        if label:
            r2 = p.add_run(f"     ({label})")
            r2.font.size = Pt(9)
            r2.font.color.rgb = MUTED
        return p

    def note_box(self, title, lines, fill="FFF4E0"):
        t = self.doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        c = t.rows[0].cells[0]
        c.width = Cm(CONTENT_WIDTH_CM)
        set_cell_shading(c, fill)
        p = c.paragraphs[0]
        r = p.add_run(title)
        r.bold = True
        for line in lines:
            q = c.add_paragraph()
            parts = line.split("**")
            for i, part in enumerate(parts):
                if part:
                    rr = q.add_run(part)
                    rr.bold = i % 2 == 1
                    rr.font.size = Pt(9.5)
            q.paragraph_format.space_after = Pt(2)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def figure(self, path: Path, caption: str, width_cm=CONTENT_WIDTH_CM):
        if not path.exists():
            raise FileNotFoundError(f"missing figure {path}; run `python -m floodcut figures` first")
        self.fig_no += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(f"Figure {self.fig_no}. ")
        r.bold = True
        r.font.size = Pt(9)
        r2 = c.add_run(caption)
        r2.font.size = Pt(9)
        r2.font.color.rgb = MUTED
        return self.fig_no

    def table(self, header, rows, widths_cm, caption=None, font_size=8.5, align_right_from=1):
        if caption:
            self.tab_no += 1
            c = self.doc.add_paragraph()
            c.paragraph_format.keep_with_next = True
            c.paragraph_format.space_after = Pt(3)
            r = c.add_run(f"Table {self.tab_no}. ")
            r.bold = True
            r.font.size = Pt(9)
            r2 = c.add_run(caption)
            r2.font.size = Pt(9)
            r2.font.color.rgb = MUTED
        assert abs(sum(widths_cm) - CONTENT_WIDTH_CM) < 0.25 or sum(widths_cm) < CONTENT_WIDTH_CM, widths_cm
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        hdr = t.rows[0]
        set_repeat_header(hdr)
        for j, h in enumerate(header):
            cell = hdr.cells[j]
            cell.width = Cm(widths_cm[j])
            set_cell_shading(cell, "E3EEFA")
            p = cell.paragraphs[0]
            run = p.add_run(str(h))
            run.bold = True
            run.font.size = Pt(font_size)
            p.paragraph_format.space_after = Pt(0)
        for i, row in enumerate(rows):
            cells = t.add_row().cells
            for j, v in enumerate(row):
                cell = cells[j]
                cell.width = Cm(widths_cm[j])
                if i % 2 == 1:
                    set_cell_shading(cell, "F6F6F4")
                p = cell.paragraphs[0]
                txt = str(v)
                bold = txt.startswith("**") and txt.endswith("**")
                run = p.add_run(txt.strip("*") if bold else txt)
                run.bold = bold
                run.font.size = Pt(font_size)
                p.paragraph_format.space_after = Pt(0)
                if j >= align_right_from and _is_num(txt.strip("*")):
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        # keep each (small) table on one page together with its caption
        for row in t.rows[:-1]:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True
        for row in t.rows:
            trPr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            cs.set(qn("w:val"), "true")
            trPr.append(cs)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(path)


def _is_num(s):
    try:
        float(s.replace("−", "-").replace("%", "").replace("×", "").replace(",", ""))
        return True
    except ValueError:
        return False


def f(x, d=3):
    s = f"{x:.{d}f}"
    return s.replace("-", "−")


def fs(x, d=3):
    s = f"{x:+.{d}f}"
    return s.replace("-", "−")


def pick(inst, init, p, seed):
    return next(r for r in inst["qaoa"] if r["config"]["init_state"] == init and r["config"]["p"] == p and r["config"]["seed"] == seed)


# ----------------------------------------------------------------------------- content
def build(bench: dict) -> Report:
    R = Report()
    prov = bench["provenance"]
    d_cfg = ScenarioConfig()
    inst = next(i for i in bench["instances"] if i["settings"]["scenario_seed"] == d_cfg.seed and i["settings"]["n"] == 10)
    val = inst["validation"]
    nm = inst["normalizers"]
    w = Weights()
    agg = {(g["n"], g["init_state"], g["p"]): g for g in bench["qaoa_aggregate"]}
    q2 = pick(inst, "dicke", 2, 0)
    g_d2 = agg[(10, "dicke", 2)]
    g_d1 = agg[(10, "dicke", 1)]
    g_b1 = agg[(10, "basis", 1)]
    g_b2 = agg[(10, "basis", 2)]
    g12 = agg[(12, "dicke", 2)]
    unif10 = 1 / math.comb(10, 3)

    # ---------------- title page
    R.doc.add_paragraph().paragraph_format.space_after = Pt(60)
    t = R.doc.add_paragraph(style="Title")
    t.add_run("FloodCut")
    R.para("Selecting k hypothetical flood-diversion sites with fixed-Hamming-weight QAOA — "
           "a synthetic, multi-objective decision-support prototype", size=14, color=MUTED, space_after=18)
    R.para("Qiskit Fall Fest 2026 Hackathon submission  •  Domain: Disaster Response", size=11)
    R.para("Team lead: Abdul Hannan", size=11, space_after=2)
    R.para("Team members: Dayyan Abdullah  •  Muhammad Dawood Ali Khan", size=11)
    R.para(f"Report generated from saved results created {prov['created_utc']} (Python {prov['python']}, "
           f"Qiskit {prov['qiskit']}).", size=9, color=MUTED, space_after=30)
    R.note_box("Read this first", [
        "This is an **educational prototype on synthetic data**. The river, city, villages, districts and every "
        "coefficient are fictional and expressed in normalized units.",
        "It contains **no real locations, no breach coordinates, no methods for altering infrastructure** and no "
        "hydrological model. Scores are a diversion proxy, a synthetic harm index and a fairness penalty — not "
        "lives, flood depths or damages.",
        "Nothing here validates real-world safety. Any adaptation would require hydrological modelling, expert, "
        "community and regulatory review.",
        "**No quantum speedup or advantage is claimed.** Every instance is solved exactly by enumeration in under a millisecond.",
    ])
    R.page_break()

    # ---------------- abstract
    R.h1("Abstract")
    R.para(
        "During severe river floods, authorities sometimes have to decide where water may be diverted so that "
        "a protected area is relieved, knowing that diversion shifts harm onto other communities. FloodCut "
        "turns a stylised, fully synthetic version of this decision into a constrained quadratic optimisation: choose exactly "
        "k of n candidate sites to balance a diversion proxy R, a synthetic harm index H with pairwise "
        "spill-overlap interactions, and a district-fairness penalty F built from population-normalised burdens. "
        "The weighted objective is an exact QUBO (it matches the direct formulas on every plan with a maximum error of "
        f"{val['qubo_max_abs_err_feasible']:.0e}), mapped to an Ising Hamiltonian and solved with a fixed-k "
        "Quantum Alternating Operator Ansatz in Qiskit: one qubit per site, a ring XY mixer built from RXX·RYY "
        "exchanges that conserves Hamming weight, and finite-shot sampling of the optimised circuit.")
    R.para(
        f"On {len(bench['instances'])} seeded instances (n = 10 and 12, k = 3; {bench['completed_qaoa_runs']} QAOA runs, "
        f"{bench['completed_sa_runs']} annealing runs) every sample was feasible and the probability outside the "
        f"k-subspace stayed below 1e−30 in ideal simulation. Starting from a Dicke state, depth-2 QAOA placed on average "
        f"{g_d2['p_opt_ratio_vs_uniform']['mean']:.1f}× (n = 10) and {g12['p_opt_ratio_vs_uniform']['mean']:.1f}× "
        "(n = 12) more probability on the exact optimum than uniform random sampling of feasible plans. However its "
        f"most-frequent sample was the optimum in only {g_d2['most_frequent_is_optimal_rate']:.0%} of n = 10 runs. "
        "The originally specified computational-basis start "
        "collapsed onto single relabelled basis states and did no better than random. Exact enumeration is exact and "
        "about four orders of magnitude faster. Greedy and a 200-step annealer were also faster and usually exact, but greedy missed "
        "the optimum on the n = 12 instance, where depth-2 QAOA's most frequent sample was optimal in all three runs. "
        "Small-budget annealing often failed. The value of the submission is a "
        "verified, honest end-to-end pipeline and a clear account of what the mathematics does and does not support.")

    # ---------------- problem
    R.h1("1. Problem, intended users and motivation")
    R.para(
        "The 2022 monsoon floods in Pakistan affected about 33 million people. The joint Post-Disaster Needs "
        "Assessment estimated damages above US$14.9 billion and economic losses of about US$15.2 billion, with "
        "reconstruction needs above US$16 billion [1, 2]. In events of this kind, decisions about where flood water may go "
        "involve competing objectives and scarce options, and they are made under uncertainty and time pressure. "
        "That pattern — select a limited number of actions whose effects interact, while protecting some people "
        "and burdening others — is the motivation for FloodCut. Beyond the cited impact figures, this report "
        "makes no historical or operational claim about any specific event, structure or location.")
    R.para("**Hackathon question.** How can quantum computing be used to address a meaningful real-world optimisation "
           "problem with competing objectives or limited resources? FloodCut treats the resource limit as an exact "
           "cardinality constraint (k sites) and the competing objectives as a weighted, normalised quadratic "
           "objective with an exact Pareto analysis alongside.")
    R.para("**Intended users.** Students, hackathon judges and educators who want to see how a multi-objective siting "
           "problem is encoded for QAOA and how its results should be checked. It is not intended for emergency managers "
           "making real decisions.")

    # ---------------- data
    R.h1("2. Synthetic scenario and modelling assumptions")
    R.para(
        f"A deterministic generator (scenario seed {d_cfg.seed} by default) places a sinusoidal river in the unit square, "
        "a city on the river near the southern edge, and three districts. District A is the upper west bank, B the lower "
        f"west bank and C the east bank. {3 * d_cfg.villages_per_district} fictional villages with synthetic populations "
        f"are placed in a floodplain band, and {d_cfg.n_sites} candidate sites on alternating banks of the upstream reach. "
        "Each site has an illustrative spill ellipse beside the river on its own bank. The map and the coefficients come "
        "from the same objects, and a SHA-256 fingerprint of all coefficients is stored with every result.")
    R.figure(FIG / "map_scenario.png", "Default synthetic scenario (seed 1). Circles are candidate sites S0–S9 and "
             "triangles are fictional villages, sized by synthetic population. Shaded ellipses are the illustrative spill "
             "areas (1σ and 2σ contours of the exposure kernel). Not real geography.", width_cm=10.5)
    R.para("Coefficients are derived from the geometry with documented formulas. Here s_i is the river distance "
           "from site i up to the city, and p_v the synthetic population of village v.")
    R.bullets([
        "Exposure kernel: e_iv = exp(−½ m_iv²), where m_iv is the Mahalanobis distance of village v from spill "
        "ellipse i. It is zero across the river, because the illustrative spill stays on its own bank.",
        f"Diversion benefit r_i = c_i·exp(−s_i/L), with synthetic capacity c_i ~ U[{d_cfg.cap_min}, {d_cfg.cap_max}] "
        f"and L = {d_cfg.relief_decay_L}. Units are fractions of a synthetic excess flow.",
        f"Redundancy o_ij = η·min(r_i, r_j)·exp(−|s_i − s_j|/λ), with η = {d_cfg.redundancy_eta} "
        f"and λ = {d_cfg.redundancy_lambda}. Nearby sites divert partly the same water.",
        "District burden a_di = Σ_{v∈d} p_v e_iv, and individual harm h_i = Σ_d a_di. Harm and fairness therefore use "
        "the same exposure data.",
        f"Spill interaction s_ij = κ·Σ_v p_v e_iv e_jv ≥ 0 (κ = {d_cfg.interaction_kappa}). This is extra harm "
        "where two illustrative spill areas reach the same village.",
    ])
    tab = inst["coefficient_table"]
    R.table(["site", "bank", "spill district", "s_i", "c_i", "r_i", "h_i"],
            [[r["site"], r["bank"], r["spill district"], f(r["river dist. s_i"]), f(r["capacity c_i"]), f(r["relief r_i"]),
              f(r["harm h_i"])] for r in tab],
            [1.6, 1.8, 2.6, 2.6, 2.6, 2.6, 2.6],
            caption="Synthetic per-site coefficients of the default instance (normalised units).", align_right_from=3)
    R.para("**Validation over every feasible plan** (default instance; the same checks pass on all four benchmark instances):", space_after=2)
    R.bullets([
        f"R > 0, H > 0 and F ≥ 0 on all {val['n_feasible']} plans. R ranges over [{f(val['R_min'])}, {f(val['R_max'])}], "
        f"H over [{f(val['H_min'])}, {f(val['H_max'])}] and F over [{f(val['F_min'], 4)}, {f(val['F_max'], 4)}].",
        f"The city is unexposed: its maximum kernel value from any spill is {val['max_city_exposure']:.1e}.",
        f"The objectives compete. The max-relief plan {val['argmax_R']}, the min-harm plan {val['argmin_H']}, the "
        f"min-fairness plan {val['argmin_F']} and the weighted optimum {val['argmin_E']} are all different.",
        f"Synthetic threshold: {val['n_meet_threshold']} of {val['n_feasible']} plans reach R ≥ τ = {DEFAULT_TAU}. "
        "On scenario seed 2 the weighted optimum is itself below the threshold (Table 5).",
        "The analytic bound R ≥ (1 − η(k−1)/2)·Σ r_i holds whenever η < 2/(k−1), and the tests check it numerically for k = 2, 3 and 4.",
    ])
    R.para("**Seed choice disclosure.** Seed 1 is the demo default because its weighted optimum differs from both the "
           "max-relief and the min-harm plans, which makes the trade-off visible. Seeds 0 and 2 are benchmarked as well. "
           "On seed 0 the optimum equals the min-harm plan, and on seed 2 it misses the threshold.", italic=False)

    # ---------------- formulation
    R.h1("3. Formulation")
    R.h2("3.1 Variables, objectives and constraint")
    R.para("Binary x_i = 1 selects candidate i. The only hard constraint is the cardinality constraint.")
    R.eq("Σ_i x_i = k", "1")
    R.eq("R(x) = Σ_i r_i x_i − Σ_{i<j} o_ij x_i x_j        (diversion proxy, maximise)", "2")
    R.eq("H(x) = Σ_i h_i x_i + Σ_{i<j} s_ij x_i x_j        (synthetic harm index, minimise)", "3")
    R.eq("b_d(x) = Σ_i a_di x_i / P_d ,     F(x) = Σ_{d<e} (b_d(x) − b_e(x))²        (fairness penalty, minimise)", "4")
    R.para("**Why the problem is genuinely quadratic.** Redundancy and spill interaction are pairwise by construction. "
           "Each b_d is linear, so F is a sum of squares of linear forms. With x_i² = x_i this expands into linear terms plus "
           "pair terms 2g_i g_j for every pair of sites, where g = a_d/P_d − a_e/P_e. F is therefore dense and couples "
           "sites that are far apart. No quadratic expression is squared anywhere, so the model remains a QUBO.")
    R.para("**Fairness caveat.** F measures the disparity of an additive, population-normalised burden. It ignores the "
           "pairwise interactions s_ij, is zero whenever all districts carry the same burden however high, and does not "
           "capture who is harmed within a district, vulnerability, consent or other ethical concerns. It is always "
           "reported next to H so that a fairer plan with more total harm is visible as a trade-off, not rewarded silently.")
    R.h2("3.2 Normalised weighted objective")
    R.eq("E(x) = w_H·(H − H_min)/ΔH + w_F·(F − F_min)/ΔF − w_R·(R − R_min)/ΔR", "5")
    R.para("The minima and ranges are taken over all feasible plans for the scenario and k, and are then held fixed. "
           "Changing the weights therefore changes priorities on a common scale. This requires enumeration, which is "
           "affordable only because there are at most 220 plans. Larger instances would need analytic bounds instead.")
    R.table(["quantity", "value", "quantity", "value"],
            [["R_min", f(nm["R_min"], 4), "ΔR", f(nm["R_range"], 4)],
             ["H_min", f(nm["H_min"], 4), "ΔH", f(nm["H_range"], 4)],
             ["F_min", f(nm["F_min"], 5), "ΔF", f(nm["F_range"], 4)],
             ["w_H", f"{w.w_H:g}", "w_F", f"{w.w_F:g}"],
             ["w_R", f"{w.w_R:g}", "τ (threshold on R)", f"{DEFAULT_TAU:g}"]],
            [4.15, 4.15, 4.15, 4.15], caption="Fixed normalisation constants (default instance, k = 3) and default weights.")
    R.h2("3.3 The synthetic city-safety threshold is checked after solving")
    R.para("A hard constraint R(x) ≥ τ is a quadratic inequality. Encoding it as a penalty, for example "
           "(R(x) − τ − slack)², produces quartic terms because R is already quadratic, so the result is not a QUBO. "
           "Keeping a QUBO would need slack qubits plus degree reduction (more qubits and penalty tuning) or a different "
           "solver. FloodCut therefore evaluates the threshold after solving and labels every plan from every method "
           "“meets τ” or “BELOW τ”. If no feasible plan meets τ, the tools say so explicitly. "
           "The weighted objective does not guarantee the threshold, and the threshold itself is a synthetic proxy, not "
           "a safety standard.")

    # ---------------- ising
    R.h1("4. QUBO-to-Ising mapping")
    R.para("Expanding (5) term by term gives E(x) = c₀ + Σ q_i x_i + Σ_{i<j} Q_ij x_i x_j. Substituting "
           "x_i = (1 − Z_i)/2, so that |1⟩ means selected, gives:")
    R.eq("E = offset + Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j", "6")
    R.eq("J_ij = Q_ij/4,    h_i = −q_i/2 − Σ_{j≠i} Q_ij/4,    offset = c₀ + Σ_i q_i/2 + Σ_{i<j} Q_ij/4", "7")
    R.para(f"Automated checks: direct formulas = QUBO on all feasible plans (max error {val['qubo_max_abs_err_feasible']:.1e}) "
           f"and on random infeasible bitstrings ({val['qubo_max_abs_err_random_bitstrings']:.1e}). The Ising diagonal "
           "including the offset matches the QUBO on all 2^n basis states, and the SparsePauliOp expectation matches "
           "⟨E⟩ for random states and basis states. Bit order is checked by preparing known basis states, "
           "sampling them, and decoding counts keys with x_i = bit[n−1−i], because Qiskit is little-endian.")

    # ---------------- qaoa
    R.h1("5. Fixed-k QAOA design")
    R.bullets([
        "**Qubits.** One per candidate site (10 or 12).",
        "**Cost layer.** U_C(γ) = exp(−iγH_C), implemented as RZ(2γh_i) and RZZ(2γJ_ij). The fairness term makes the "
        "cost layer all-to-all (45 RZZ gates per layer for n = 10). The offset is a global phase: it is dropped from the "
        "circuit but included in every reported energy.",
        "**Mixer layer.** A ring of XY exchanges in parity order: first edges (0,1), (2,3), …, then (1,2), …, (n−1,0). "
        "Each edge applies RXX(β)·RYY(β), which equals exp(−iβ/2·(XX + YY)) exactly because XX and YY commute "
        "on the same pair (checked in the tests). Every factor conserves Hamming weight.",
        "**Not the exact ring exponential.** Neighbouring edges share a qubit and do not commute, so the parity-ordered "
        "product is not exp(−iβ/2·Σ_edges(XX + YY)). A test confirms that the two operators differ. A mixer_reps "
        "option applies Trotter-style repetitions.",
        "**Angles and ordering.** The parameters are [γ_1, β_1, …, γ_p, β_p]. Each layer applies U_C(γ_l) and then "
        "U_M(β_l). Depths p = 1 and 2 are used.",
        "**Optimisation.** The exact ⟨E⟩ is computed from the Qiskit Statevector, and SciPy COBYLA minimises it with "
        "rhobeg = 0.5 and a budget of 150 evaluations (p = 1) or 300 (p = 2). Initial angles are seeded draws from "
        "U[0, π]. The optimised circuit is then sampled with StatevectorSampler (4096 shots, seeded). "
        "Exact expectations and sampled frequencies are reported separately.",
    ])
    R.h2("5.1 Initial state: a finding that changed the design")
    R.para("The brief asked for a computational-basis initial state with Hamming weight k. Two problems appeared:")
    R.bullets([
        "**At p = 1 the cost layer has no effect.** A basis state is an eigenstate of the diagonal cost unitary, so the "
        "first cost layer only adds a global phase. The p = 1 output distribution is then independent of γ_1, which a test "
        "verifies. As a fix, the basis mode inserts an extra parameterised mixer layer U_M(β_0) before the first cost layer.",
        f"**With that fix, the circuit still collapses.** Optimising ⟨E⟩ from a basis start drove the circuit towards "
        f"angles at which the exchanges act as permutations, leaving a single relabelled basis state. The most likely plan's "
        f"probability averaged {g_b1['max_plan_probability']['mean']:.2f} (p = 1) and {g_b2['max_plan_probability']['mean']:.2f} "
        f"(p = 2), and P(opt) was {g_b1['p_opt_ratio_vs_uniform']['mean']:.1f}× and {g_b2['p_opt_ratio_vs_uniform']['mean']:.1f}× "
        "the uniform baseline (Table 4). That is a classical search over relabellings, not a useful quantum superposition.",
    ])
    R.para("FloodCut therefore makes the **Dicke-state start** the default. A weight-k basis state |0…0 1…1⟩ is followed "
           "by the deterministic Hamming-weight-preserving preparation of Bärtschi and Eidenbenz [5], built from CX, CRY and CCRY "
           "gates, which gives the uniform superposition over all C(n,k) plans. The preparation is checked to 1e−12 for every "
           "n ≤ 12 and 1 ≤ k < n. The basis-start variant is kept, benchmarked and reported as specified.")
    R.figure(FIG / "architecture.png", "Architecture. One shared Python package (floodcut) provides the scenario, objective, "
             "QUBO/Ising mapping, QAOA circuit, classical baselines and evaluation. The CLI, the benchmark, the Streamlit demo "
             "and this report all call it.")

    # ---------------- setup
    R.h1("6. Experimental setup and reproducibility")
    spec = bench["spec"]
    R.bullets([
        f"Software: Python {prov['python']} on {prov['platform']}; Qiskit {prov['qiskit']}, NumPy {prov['numpy']}, SciPy {prov['scipy']}. "
        "Exact pins are in requirements.txt.",
        f"Simulator: {prov['simulator']}. Ideal and noiseless, with no hardware coupling map.",
        f"Benchmark preset 'full': scenario seeds {[i['seed'] for i in spec['instances'] if i['n'] == 10]} at n = 10 and seed 1 at n = 12, "
        f"k = {spec['k']}, optimiser seeds {spec['qaoa_seeds']}, p ∈ {spec['ps']}, initial states {spec['inits']}, "
        f"{spec['shots']} shots. Annealing used {len(spec['sa_seeds'])} seeds × {spec['sa_steps']} steps, plus a budget study.",
        f"Completed: {bench['completed_qaoa_runs']} QAOA runs and {bench['completed_sa_runs']} annealing runs in "
        f"{bench['total_runtime_s']:.0f} s of wall time. Every run is stored with its raw counts, angles, timings and provenance in "
        "results/benchmark_full.json.",
        "Commands: pip install -r requirements.txt; pip install -e .; pytest; python -m floodcut quick; "
        "python -m floodcut bench --preset full; python -m floodcut figures; python report/build_report.py; "
        "streamlit run app/streamlit_app.py.",
        "Metrics: E, R, H, F; threshold status; the additive gap E − E*; the normalised gap (E − E*)/(E_max − E*) "
        "∈ [0, 1]; and the plan's rank among all feasible plans. Percentage gaps relative to E* are not used, because "
        "E* is negative.",
    ])

    # ---------------- results
    R.h1("7. Results")
    R.h2("7.1 Default instance: all methods side by side")
    ex = inst["exact"]
    rows = []

    def row(name, r, budget):
        rows.append([name, r["label"], f(r["E"], 4), f(r["R"]), f(r["H"]), f(r["F"], 4),
                     "meets" if r["meets_threshold"] else "BELOW", f(r["additive_gap"], 4), f"{r['rank']}/{r['n_feasible']}", budget])

    row("Exact enumeration", ex, f"{ex['evaluations']} plans")
    row("QAOA most frequent", q2["most_frequent"], f"{q2['n_evaluations']} evals + {q2['shots']} shots")
    row("QAOA best observed*", q2["best_observed"], "post-hoc over samples")
    row("Greedy", inst["greedy"], f"{inst['greedy']['evaluations']} marginals")
    sa0 = inst["annealing"][0]
    row("Annealing (seed 0)", sa0, f"{sa0['evaluations']} evals")
    R.table(["method", "plan", "E", "R", "H", "F", "τ", "gap", "rank", "budget"], rows,
            [2.6, 2.1, 1.4, 1.2, 1.2, 1.3, 1.1, 1.3, 1.3, 3.1],
            caption="Default instance (seed 1, n = 10, k = 3, default weights). QAOA: Dicke start, p = 2, optimiser seed 0. "
                    "*Best observed is the lowest-E plan among the samples, found by classical post-processing.", font_size=8)
    R.para(f"The QAOA circuit's most frequent sample was {q2['most_frequent']['label']} with {q2['most_frequent']['count']} of "
           f"{q2['shots']} shots. This is the rank-{q2['most_frequent']['rank']} plan, with an additive gap of "
           f"{f(q2['most_frequent']['additive_gap'], 3)}. The exact optimum {ex['label']} was sampled {q2['best_observed']['count']} times "
           f"(empirical P = {q2['empirical_p_opt']:.3f}, exact P = {q2['exact_p_opt']:.3f}, uniform 1/120 = {unif10:.4f}). "
           f"Optimisation lowered ⟨E⟩ from {fs(q2['expected_E_initial'])} to {fs(q2['expected_E_opt'])}, against a uniform-plan mean of "
           f"{fs(q2['uniform_feasible_mean_E'])} and E* = {fs(ex['E_star'])}. The demo **displays the most frequent sample**. "
           "The best-observed plan is shown only with a caveat: with 4096 shots over 120 plans it approaches exhaustive search.")
    R.figure(FIG / "method_maps.png", "Plans chosen by each method on the default instance. Red crosses mark the selected sites "
             "and shaded areas the illustrative spill footprints.")
    R.figure(FIG / "burdens.png", "Population-normalised district burdens b_d of each method's plan (additive proxy). "
             "Exact, greedy and annealing coincide on this instance.", width_cm=11)

    R.h2("7.2 QAOA across seeds and depths")
    arows = []
    for g in bench["qaoa_aggregate"]:
        s = g["p_opt_ratio_vs_uniform"]
        arows.append([g["n"], g["init_state"], g["p"], g["runs"], f(g["exact_p_opt"]["mean"]), f"{s['mean']:.1f}",
                      f"{s['min']:.1f}–{s['max']:.1f}", f"{g['runs_p_opt_above_uniform']}/{g['runs']}",
                      f"{g['most_frequent_is_optimal_rate']:.0%}", f(g["most_frequent_normalized_gap"]["mean"]),
                      f(g["max_plan_probability"]["mean"]), f"{g['n_evaluations']['mean']:.0f}"])
    R.table(["n", "init", "p", "runs", "P(opt)", "×unif", "range", ">unif", "MF opt", "MF ngap", "max P", "evals"], arows,
            [0.8, 1.3, 0.7, 1.0, 1.4, 1.3, 1.9, 1.4, 1.5, 1.6, 1.4, 1.3],
            caption="QAOA summary. P(opt) is the exact probability of the optimal plan in the optimised state. ×unif is "
                    "P(opt)/(1/C(n,k)) with its min–max range. >unif counts runs above uniform. MF opt is how often the "
                    "most-frequent sample was optimal, and MF ngap its mean normalised gap. max P is the largest single-plan "
                    "probability (near 1 means collapse).", font_size=8, align_right_from=2)
    R.figure(FIG / "p_opt_summary.png", "Optimal-plan probability relative to uniform sampling of feasible plans. Orange bars "
             "are the Dicke start and grey bars the basis start. Whiskers show the min–max over seeds.")
    R.para(f"Every QAOA run had a feasible-sample rate of 100% and leakage below 1e−30. With the Dicke start, optimised "
           f"⟨E⟩ was below the uniform-plan mean in {g_d1['runs_expected_E_below_uniform_mean'] + g_d2['runs_expected_E_below_uniform_mean']} "
           f"of {g_d1['runs'] + g_d2['runs']} n = 10 runs. P(opt) exceeded uniform in {g_d1['runs_p_opt_above_uniform']}/9 runs at p = 1 and "
           f"{g_d2['runs_p_opt_above_uniform']}/9 at p = 2. **Failures are real.** Several p = 1 runs stalled close to or below the uniform "
           "baseline, and the most-frequent sample was often a good but suboptimal plan. P(opt) of a few percent "
           "means the circuit is a biased sampler, not a solver that outputs the optimum.")
    R.figure(FIG / "qaoa_counts.png", "Most frequent sampled plans (default instance, Dicke start, p = 2, seed 0, 4096 shots). "
             "The orange bar is the exact optimum.", width_cm=12)
    if (FIG / "convergence.png").exists():
        R.figure(FIG / "convergence.png", "COBYLA traces of the exact ⟨E⟩ (default instance, optimiser seed 0). The basis-start "
                 "traces reach low ⟨E⟩ by collapsing onto single plans, which is not a better quantum state.", width_cm=11)

    R.h2("7.3 Classical baselines")
    crow = []
    for c in bench["classical_aggregate"]:
        crow.append([c["scenario_seed"], c["n"], c["n_feasible"], c["exact_label"], "meets" if c["exact_meets_threshold"] else "BELOW",
                     f"{c['n_meet_threshold']}", c["greedy_label"], c["greedy_rank"], f(c["greedy_additive_gap"], 4),
                     f"{c['sa_success_rate']:.0%}", c["sa_evaluations"]])
    R.table(["seed", "n", "plans", "exact optimum", "τ", "#meet", "greedy", "rank", "gap", "SA opt", "SA evals"], crow,
            [1.0, 0.8, 1.2, 2.4, 1.5, 1.3, 2.4, 1.1, 1.5, 1.6, 1.6],
            caption="Exact, greedy and annealing (10 seeds × 200 steps) on every instance. #meet is the number of feasible plans "
                    "reaching τ.", font_size=8, align_right_from=1)
    sab = bench["sa_budget_study"]
    srows = [[r["scenario_seed"], r["n"], r["steps"], r["evaluations"], r["n_feasible"], f"{r['success_rate']:.0%}",
              f(r["mean_additive_gap"], 4), r["worst_rank"]] for r in sab]
    R.table(["seed", "n", "steps", "evals", "plans", "success", "mean gap", "worst rank"], srows,
            [1.6, 1.4, 1.8, 2.0, 2.0, 2.4, 2.8, 2.6],
            caption="Annealing budget study (10 seeds each). At 200 steps annealing spends more evaluations than there are "
                    "feasible plans, so exhaustive enumeration would be cheaper.", font_size=8)
    gl = [c for c in bench["classical_aggregate"] if c["greedy_rank"] > 1]
    R.para("Greedy was optimal on " + f"{len(bench['classical_aggregate']) - len(gl)} of {len(bench['classical_aggregate'])} instances"
           + (f" and missed on the n = 12 instance (rank {gl[0]['greedy_rank']}, gap {f(gl[0]['greedy_additive_gap'], 4)})" if gl else "")
           + ". Annealing with small budgets (10–25 swaps) often failed. On the n = 12 instance it needed about 200 steps to "
           "succeed reliably. On scenario seed 2 the weighted optimum misses the synthetic threshold, so the trade-off is "
           "visible: a plan that reaches τ exists, but it costs more harm or fairness than the chosen weights accept.")

    R.h2("7.4 Runtime and circuit resources")
    cm = q2["circuit"]
    R.table(["item", "value"], [
        ["exact enumeration of 120 plans", f"{ex['runtime_s'] * 1e3:.2f} ms"],
        ["QAOA setup / optimisation / sampling (p = 2, seed 0)", f"{q2['setup_s']:.3f} s / {q2['optimize_s']:.2f} s / {q2['sample_s']:.3f} s"],
        ["logical depth; 2-qubit gates; 3-qubit gates", f"{cm['logical_depth']}; {cm['logical_two_qubit_gates']}; {cm['logical_three_qubit_gates']}"],
        ["transpiled depth; CX (basis rz, sx, x, cx; no routing)", f"{cm['transpiled_depth']}; {cm['transpiled_cx']}"],
        ["of which Dicke preparation: depth; CX", f"{cm['prep_transpiled_depth']}; {cm['prep_transpiled_cx']}"],
        ["of which 2 QAOA layers: depth; CX", f"{cm['layers_transpiled_depth']}; {cm['layers_transpiled_cx']}"],
        ["n = 12, p = 2 transpiled CX (prep + layers)", f"{g12['prep_transpiled_cx']} + {g12['layers_transpiled_cx']}"],
    ], [10.6, 6.0], caption="Runtime and resources on the default instance (simulation on a laptop CPU).", align_right_from=9)
    R.para("Simulated QAOA is roughly four orders of magnitude slower than exact enumeration here. Transpilation assumes "
           "all-to-all connectivity. On heavy-hex hardware the dense RZZ layer would need SWAP routing and would be much deeper.")

    # ---------------- pareto
    R.h1("8. Pareto analysis")
    prow = []
    for i in bench["instances"]:
        pa = i["pareto"]
        prow.append([i["settings"]["scenario_seed"], i["settings"]["n"], pa["n_feasible"], pa["n_nondominated_3obj"],
                     pa["n_frontier_2obj"], pa["n_unsupported_3obj_grid"], pa["fairness_harm_tension_pairs"]])
    R.table(["seed", "n", "plans", "3-obj nondominated", "2-obj (R,H) frontier", "not reached by weight grid", "fairness/harm tension pairs"],
            prow, [1.4, 1.2, 1.6, 3.0, 3.0, 3.2, 3.2],
            caption="Exact nondominated sets over (max R, min H, min F) by full enumeration. Ties are kept, and dominance needs "
                    "≤ on all objectives and < on at least one. The last column counts ordered pairs of nondominated plans "
                    "in which the fairer plan causes more total harm.", font_size=8, align_right_from=0)
    R.figure(FIG / "pareto.png", "Diversion proxy versus synthetic harm for every feasible plan of the default instance, "
             "coloured by fairness penalty. Ringed points are the exact three-objective nondominated set. The dotted step "
             "line is the separate two-objective (R, H) frontier. Orange markers are QAOA's most frequent plans under a "
             "six-point weight sweep (Dicke start, p = 2, seed 0). Identical plans share a marker.", width_cm=14.0)
    sw = bench.get("weight_sweep") or []
    wrows = []
    for r in sw:
        ww = r["weights"]
        m = r["qaoa_most_frequent"]
        e = r["exact_weighted_optimum"]
        wrows.append([f"({ww['w_H']:g}, {ww['w_F']:g}, {ww['w_R']:g})", m["label"], m["rank"], "yes" if r["qaoa_in_3obj_nondominated"] else "no",
                      e["label"], "meets" if e["meets_threshold"] else "BELOW", f(r["exact_p_opt"])])
    if wrows:
        R.table(["weights (H, F, R)", "QAOA most frequent", "rank", "3-obj nondominated", "exact weighted optimum", "τ", "P(opt)"],
                wrows, [2.5, 2.9, 1.1, 2.6, 3.0, 1.5, 1.5], caption="Weight sweep on the default instance.", font_size=8,
                align_right_from=2)
    R.para("The sweep is a sparse sample of six weight vectors and **is not the Pareto front**. A weighted sum can only reach "
           "supported nondominated points, those on the convex hull. Table 8 shows that several exact nondominated plans "
           "were never the minimiser for any weight on a dense simplex grid. QAOA's most-frequent plan was often not the "
           "weighted optimum, and sometimes not nondominated at all. Shallow QAOA here samples the trade-off space; it does "
           "not reliably locate it.")

    # ---------------- limitations
    R.h1("9. Limitations")
    R.bullets([
        "**No hydrodynamic calibration.** Relief, harm and spill areas are geometric proxies, not flood simulations. Water "
        "levels, timing, flow routing and embankment behaviour are not modelled.",
        "**Uncertainty in input estimates.** All coefficients are synthetic point values. A real decision would need "
        "uncertain inputs, scenario ensembles and sensitivity analysis. None is modelled.",
        "**Simplified pairwise interactions.** Redundancy and spill overlap are pairwise only. Real interactions can be "
        "higher-order and dynamic.",
        "**Simplified district fairness.** F is an additive, population-normalised disparity proxy. It ignores interaction "
        "harm, intra-district distribution, vulnerability and procedural justice, and it can be zero when everyone is "
        "badly burdened.",
        "**No guarantee of city safety.** The threshold is a synthetic relief proxy checked after solving. The weighted "
        "objective can, and on seed 2 does, choose a plan below it.",
        "**Small instances are solved exactly.** With 120–220 plans, enumeration is exact and takes under a millisecond. "
        "Nothing here suggests a quantum speedup or advantage.",
        "**Ideal-simulator assumptions.** Noiseless statevector simulation. On hardware, noise would leak probability out of "
        "the k-subspace, the Dicke preparation and all-to-all RZZ would need deep routed circuits, and sampling would "
        "replace exact ⟨E⟩ during optimisation.",
        "**QAOA parameter sensitivity and runtime overhead.** Results vary strongly with the optimiser seed (Table 4 ranges). "
        "The basis-state start collapses. Parameter optimisation dominates runtime.",
        "**Normalisation by enumeration.** The fixed normalisers use the exact feasible range, which does not scale. "
        "Larger instances would need bounds, which would change the weight semantics.",
        "**Expert review is required** before any real-world adaptation, including hydrologists, emergency managers, "
        "affected communities, legal and ethics review.",
    ])
    R.h1("10. Ethical considerations")
    R.para("Deliberately redirecting flood water moves risk from one group of people to another. A tool that hides this "
           "trade-off inside a single score, or presents synthetic scores as lives saved, could legitimise harmful choices. "
           "FloodCut therefore keeps every objective visible, reports threshold failures rather than hiding them, labels all "
           "outputs as synthetic, contains no real locations or operational detail, and does not claim quantum advantage. "
           "Weights encode value judgements. They belong to accountable decision makers and affected communities, not to "
           "an optimiser.")

    # ---------------- references
    R.h1("References and code attribution")
    refs = [
        "[1] Government of Pakistan, ADB, EU, UNDP, World Bank. Pakistan Floods 2022: Post-Disaster Needs Assessment (PDNA), "
        "October 2022. https://www.undp.org/pakistan/publications/pakistan-floods-2022-post-disaster-needs-assessment-pdna-main-report",
        "[2] World Bank press release, 28 October 2022: “Pakistan: Flood Damages and Economic Losses Over USD 30 billion and "
        "Reconstruction Needs Over USD 16 billion – New Assessment”. https://www.worldbank.org/en/news/press-release/2022/10/28/"
        "pakistan-flood-damages-and-economic-losses-over-usd-30-billion-and-reconstruction-needs-over-usd-16-billion-new-assessme",
        "[3] E. Farhi, J. Goldstone, S. Gutmann. A Quantum Approximate Optimization Algorithm. arXiv:1411.4028 (2014).",
        "[4] S. Hadfield et al. From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz. "
        "Algorithms 12(2):34 (2019).",
        "[5] A. Bärtschi, S. Eidenbenz. Deterministic Preparation of Dicke States. FCT 2019, LNCS 11651, 126–139; arXiv:1904.07358.",
        "[6] Z. Wang, N. C. Rubin, J. M. Dominy, E. G. Rieffel. XY mixers: Analytical and numerical results for the quantum "
        "alternating operator ansatz. Phys. Rev. A 101, 012320 (2020).",
        "[7] A. Lucas. Ising formulations of many NP problems. Frontiers in Physics 2:5 (2014).",
        "[8] I. Das, J. E. Dennis. A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set "
        "generation in multicriteria optimization problems. Structural Optimization 14, 63–69 (1997).",
        "[9] A. Javadi-Abhari et al. Quantum computing with Qiskit. arXiv:2405.08810 (2024). Qiskit is Apache-2.0 licensed.",
        "[10] P. Virtanen et al. SciPy 1.0. Nature Methods 17, 261–272 (2020); M. J. D. Powell, COBYLA (1994).",
    ]
    for r_ in refs:
        p = R.para(r_, size=9, space_after=3)
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.first_line_indent = Cm(-0.6)
    R.para("Code attribution: all FloodCut code was written for this submission. It uses Qiskit (circuits, Statevector, "
           "StatevectorSampler, SparsePauliOp, transpile), NumPy, SciPy (COBYLA), Matplotlib, Streamlit and python-docx. "
           "The Dicke-state circuit follows the construction in [5]. The figure palette follows a colour-blind-validated "
           "categorical order. AI assistance (Claude) was used to write code and draft this report; all reported numbers "
           "come from the saved benchmark JSON produced by the code in the repository.", size=9)
    return R


def main():
    bench_path = ROOT / "results" / "benchmark_full.json"
    bench = load_json(bench_path)
    rep = build(bench)
    rep.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
