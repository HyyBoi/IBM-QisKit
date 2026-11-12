"""Deterministic, seeded SYNTHETIC flood-diversion scenario.

Everything here is fictional and expressed in normalized units on the unit square.
The same geometry objects are used both to derive the optimization coefficients and
to draw the map, so the visualization and the optimizer cannot disagree.

Coefficient definitions (see docs / report for discussion):

* exposure kernel  e_iv = exp(-0.5 * m_iv^2), with m_iv the Mahalanobis distance of
  village v from the centre of site i's illustrative spill ellipse; e_iv = 0 when the
  village is on the opposite bank (illustrative spill does not cross the river).
* relief           r_i  = c_i * exp(-s_i / L)
* redundancy       o_ij = eta * min(r_i, r_j) * exp(-|s_i - s_j| / lambda)
* district burden  a_di = sum_{v in d} p_v e_iv
* individual harm  h_i  = sum_d a_di
* spill interaction s_ij = kappa * sum_v p_v e_iv e_jv
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from .config import ScenarioConfig

DISTRICT_NAMES = ("A", "B", "C")
DISTRICT_LABELS = {
    "A": "District A (west bank, upper)",
    "B": "District B (west bank, lower)",
    "C": "District C (east bank)",
}


@dataclass
class Scenario:
    config: ScenarioConfig
    # geometry
    river_y: np.ndarray  # (M,) sample points along the centreline, north (1) to south (0)
    river_x: np.ndarray
    river_phase: float
    city_xy: np.ndarray  # (2,)
    district_polygons: dict  # name -> (P, 2) array
    village_xy: np.ndarray  # (V, 2)
    village_pop: np.ndarray  # (V,)
    village_district: np.ndarray  # (V,) int index into DISTRICT_NAMES
    village_bank: np.ndarray  # (V,) -1 west, +1 east
    site_xy: np.ndarray  # (n, 2)
    site_bank: np.ndarray  # (n,) -1 west, +1 east
    site_s: np.ndarray  # (n,) river distance upstream of the city
    site_capacity: np.ndarray  # (n,)
    spill_center: np.ndarray  # (n, 2)
    spill_axes: np.ndarray  # (n, 2) semi-axes (along, across)
    spill_angle: np.ndarray  # (n,) angle of the along-river axis (radians)
    # coefficients
    exposure: np.ndarray  # (n, V) e_iv
    r: np.ndarray  # (n,)
    o: np.ndarray  # (n, n) symmetric, zero diagonal
    h: np.ndarray  # (n,)
    s: np.ndarray  # (n, n) symmetric, zero diagonal
    a: np.ndarray  # (D, n) district burden a_di
    P: np.ndarray  # (D,) district populations
    city_exposure: np.ndarray  # (n,) kernel value of each spill at the city centre

    @property
    def n(self) -> int:
        return int(self.r.shape[0])

    @property
    def n_districts(self) -> int:
        return int(self.P.shape[0])

    def site_district(self) -> list[str]:
        """District whose polygon contains each site's spill centre (for display)."""
        out = []
        for c in self.spill_center:
            name = "?"
            for d, poly in self.district_polygons.items():
                if Path(poly).contains_point(tuple(c)):
                    name = d
                    break
            out.append(name)
        return out

    def fingerprint(self) -> str:
        """SHA-256 of the configuration and every coefficient array (rounded)."""
        payload = {
            "config": self.config.to_dict(),
            "r": np.round(self.r, 12).tolist(),
            "o": np.round(self.o, 12).tolist(),
            "h": np.round(self.h, 12).tolist(),
            "s": np.round(self.s, 12).tolist(),
            "a": np.round(self.a, 12).tolist(),
            "P": np.round(self.P, 12).tolist(),
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def coefficient_table(self) -> list[dict]:
        rows = []
        dist = self.site_district()
        for i in range(self.n):
            rows.append(
                {
                    "site": f"S{i}",
                    "bank": "west" if self.site_bank[i] < 0 else "east",
                    "spill district": dist[i],
                    "river dist. s_i": round(float(self.site_s[i]), 4),
                    "capacity c_i": round(float(self.site_capacity[i]), 4),
                    "relief r_i": round(float(self.r[i]), 4),
                    "harm h_i": round(float(self.h[i]), 4),
                }
            )
        return rows


# ----------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------

def _river_x(y: np.ndarray, cfg: ScenarioConfig, phase: float) -> np.ndarray:
    return 0.5 + cfg.river_amp * np.sin(2 * np.pi * cfg.river_freq * y + phase)


def _river_dxdy(y: np.ndarray, cfg: ScenarioConfig, phase: float) -> np.ndarray:
    return cfg.river_amp * 2 * np.pi * cfg.river_freq * np.cos(2 * np.pi * cfg.river_freq * y + phase)


def _arc_length_from_city(y: np.ndarray, cfg: ScenarioConfig, phase: float) -> np.ndarray:
    """River distance from the city (at cfg.city_y) up to each y (y > city_y)."""
    grid = np.linspace(cfg.city_y, 1.0, 4001)
    speed = np.sqrt(1.0 + _river_dxdy(grid, cfg, phase) ** 2)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(grid))])
    return np.interp(y, grid, cum)


def _unit_frames(y: np.ndarray, cfg: ScenarioConfig, phase: float):
    """Unit tangent (pointing north, i.e. upstream) and unit east-pointing normal."""
    dx = _river_dxdy(y, cfg, phase)
    norm = np.sqrt(1.0 + dx**2)
    tangent = np.stack([dx / norm, 1.0 / norm], axis=-1)
    normal_east = np.stack([1.0 / norm, -dx / norm], axis=-1)
    return tangent, normal_east


def _district_polygons(cfg: ScenarioConfig, phase: float) -> dict:
    ys_top = np.linspace(1.0, cfg.district_split_y, 120)
    ys_bot = np.linspace(cfg.district_split_y, 0.0, 120)
    ys_all_up = np.linspace(0.0, 1.0, 240)
    poly_a = np.vstack(
        [[0.0, 1.0], np.column_stack([_river_x(ys_top, cfg, phase), ys_top]), [0.0, cfg.district_split_y]]
    )
    poly_b = np.vstack(
        [[0.0, cfg.district_split_y], np.column_stack([_river_x(ys_bot, cfg, phase), ys_bot]), [0.0, 0.0]]
    )
    poly_c = np.vstack(
        [[1.0, 0.0], [1.0, 1.0], np.column_stack([_river_x(ys_all_up[::-1], cfg, phase), ys_all_up[::-1]])]
    )
    return {"A": poly_a, "B": poly_b, "C": poly_c}


def _exposure(points: np.ndarray, center: np.ndarray, axes: np.ndarray, angle: float) -> np.ndarray:
    d = points - center
    t = np.array([np.cos(angle), np.sin(angle)])
    nrm = np.array([-np.sin(angle), np.cos(angle)])
    u = d @ t / axes[0]
    w = d @ nrm / axes[1]
    return np.exp(-0.5 * (u**2 + w**2))


# ----------------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------------

def generate_scenario(cfg: ScenarioConfig | None = None) -> Scenario:
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)
    phase = float(rng.uniform(0, 2 * np.pi))

    river_y = np.linspace(1.0, 0.0, 400)
    river_x = _river_x(river_y, cfg, phase)
    city_xy = np.array([float(_river_x(np.array(cfg.city_y), cfg, phase)), cfg.city_y])
    polys = _district_polygons(cfg, phase)
    paths = {d: Path(p) for d, p in polys.items()}

    # --- villages: a fixed quota per district, rejection-sampled -------------------
    vxy, vdist = [], []
    for di, d in enumerate(DISTRICT_NAMES):
        placed = 0
        attempts = 0
        while placed < cfg.villages_per_district:
            attempts += 1
            if attempts > 20000:
                raise RuntimeError("village placement failed; adjust ScenarioConfig")
            y = rng.uniform(0.24, 0.97)
            side = -1.0 if d in ("A", "B") else 1.0
            off = rng.uniform(cfg.village_offset_min, cfg.village_offset_max)
            x = float(_river_x(np.array(y), cfg, phase)) + side * off
            pt = np.array([x, y])
            if not (0.03 < x < 0.97):
                continue
            if not paths[d].contains_point(tuple(pt)):
                continue
            if np.linalg.norm(pt - city_xy) < cfg.city_radius + 0.06:
                continue
            if any(np.linalg.norm(pt - q) < cfg.village_min_spacing for q in vxy):
                continue
            vxy.append(pt)
            vdist.append(di)
            placed += 1
    village_xy = np.array(vxy)
    village_district = np.array(vdist, dtype=int)
    village_pop = rng.uniform(cfg.pop_min, cfg.pop_max, size=len(vxy))
    village_bank = np.where(village_xy[:, 0] < _river_x(village_xy[:, 1], cfg, phase), -1, 1)

    # --- candidate sites: stratified jittered y positions (always succeeds) ---------
    span = cfg.site_y_max - cfg.site_y_min
    strata = span / cfg.n_sites
    jitter = rng.uniform(cfg.site_jitter_lo, cfg.site_jitter_hi, cfg.n_sites)
    site_y = cfg.site_y_max - (np.arange(cfg.n_sites) + jitter) * strata  # north (upstream) first
    banks = np.array([-1 if i % 2 == 0 else 1 for i in range(cfg.n_sites)])
    # randomize the alternation start so seeds differ in bank pattern
    if rng.uniform() < 0.5:
        banks = -banks
    tangent, normal_e = _unit_frames(site_y, cfg, phase)
    centre_line = np.column_stack([_river_x(site_y, cfg, phase), site_y])
    site_xy = centre_line + (banks[:, None] * cfg.river_half_width) * normal_e
    spill_center = centre_line + (banks[:, None] * cfg.spill_offset) * normal_e
    spill_axes = np.column_stack(
        [
            rng.uniform(cfg.spill_along_min, cfg.spill_along_max, cfg.n_sites),
            rng.uniform(cfg.spill_across_min, cfg.spill_across_max, cfg.n_sites),
        ]
    )
    spill_angle = np.arctan2(tangent[:, 1], tangent[:, 0])
    site_s = _arc_length_from_city(site_y, cfg, phase)
    capacity = rng.uniform(cfg.cap_min, cfg.cap_max, cfg.n_sites)

    # --- coefficients ----------------------------------------------------------------
    n, V = cfg.n_sites, len(village_xy)
    exposure = np.zeros((n, V))
    city_exp = np.zeros(n)
    for i in range(n):
        e = _exposure(village_xy, spill_center[i], spill_axes[i], spill_angle[i])
        e[village_bank != banks[i]] = 0.0  # spill stays on its own bank
        exposure[i] = e
        city_exp[i] = _exposure(city_xy[None, :], spill_center[i], spill_axes[i], spill_angle[i])[0]

    r = capacity * np.exp(-site_s / cfg.relief_decay_L)
    o = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            val = cfg.redundancy_eta * min(r[i], r[j]) * np.exp(-abs(site_s[i] - site_s[j]) / cfg.redundancy_lambda)
            o[i, j] = o[j, i] = val

    D = len(DISTRICT_NAMES)
    a = np.zeros((D, n))
    P = np.zeros(D)
    for d in range(D):
        mask = village_district == d
        P[d] = village_pop[mask].sum()
        a[d] = exposure[:, mask] @ village_pop[mask]
    h = a.sum(axis=0)
    weighted = exposure * village_pop[None, :]
    s = cfg.interaction_kappa * (weighted @ exposure.T)
    np.fill_diagonal(s, 0.0)

    return Scenario(
        config=cfg,
        river_y=river_y,
        river_x=river_x,
        river_phase=phase,
        city_xy=city_xy,
        district_polygons=polys,
        village_xy=village_xy,
        village_pop=village_pop,
        village_district=village_district,
        village_bank=village_bank,
        site_xy=site_xy,
        site_bank=banks,
        site_s=site_s,
        site_capacity=capacity,
        spill_center=spill_center,
        spill_axes=spill_axes,
        spill_angle=spill_angle,
        exposure=exposure,
        r=r,
        o=o,
        h=h,
        s=s,
        a=a,
        P=P,
        city_exposure=city_exp,
    )
