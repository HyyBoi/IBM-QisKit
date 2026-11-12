"""Shared configuration dataclasses and documented defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ScenarioConfig:
    """Parameters of the synthetic scenario generator (all units normalized, unit square)."""

    # Default demo seed 1: chosen (and disclosed) because its default-weight optimum differs
    # from both the max-relief and the min-harm plans, so the trade-off is visible.
    seed: int = 1
    n_sites: int = 10
    villages_per_district: int = 5
    # River centreline x(y) = 0.5 + amp * sin(2*pi*freq*y + phase); phase is seeded.
    river_amp: float = 0.08
    river_freq: float = 1.1
    river_half_width: float = 0.018
    # City (protected area) on the river near the southern edge.
    city_y: float = 0.12
    city_radius: float = 0.07
    # District split on the west bank (A = west-upper, B = west-lower, C = east bank).
    district_split_y: float = 0.62
    # Candidate breach sites lie on the upstream reach between these y-values.
    site_y_min: float = 0.42
    site_y_max: float = 0.94
    # Sites are placed in n equal strata of the reach, jittered within [lo, hi] of each stratum.
    site_jitter_lo: float = 0.2
    site_jitter_hi: float = 0.8
    # Synthetic diversion capacity c_i ~ U[cap_min, cap_max] (fraction of synthetic excess flow).
    cap_min: float = 0.20
    cap_max: float = 0.36
    # Relief attenuation length L in r_i = c_i * exp(-s_i / L), s_i = river distance upstream of city.
    relief_decay_L: float = 1.5
    # Redundancy o_ij = eta * min(r_i, r_j) * exp(-|s_i - s_j| / lambda).
    redundancy_eta: float = 0.6
    redundancy_lambda: float = 0.15
    # Illustrative spill ellipse: offset from bank and semi-axes (along, across the river).
    spill_offset: float = 0.09
    spill_along_min: float = 0.05
    spill_along_max: float = 0.08
    spill_across_min: float = 0.04
    spill_across_max: float = 0.065
    # Spill-interaction strength s_ij = kappa * sum_v p_v e_iv e_jv.
    interaction_kappa: float = 1.0
    # Synthetic village populations p_v ~ U[pop_min, pop_max] (synthetic population units).
    # Villages lie in a floodplain band at this distance from the river centreline.
    village_offset_min: float = 0.06
    village_offset_max: float = 0.22
    village_min_spacing: float = 0.055
    pop_min: float = 0.5
    pop_max: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Weights:
    """Weights of the normalized objective E = wH*Hn + wF*Fn - wR*Rn."""

    w_H: float = 1.0
    w_F: float = 0.5
    w_R: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_K = 3
# Synthetic city-safety threshold: relief proxy R(x) must reach this fraction of the
# synthetic excess flow. Evaluated AFTER solving; it is not part of the objective.
DEFAULT_TAU = 0.5


@dataclass(frozen=True)
class QAOAConfig:
    p: int = 1
    maxiter: int | None = None  # None -> 150 for p=1, 300 for p>=2
    shots: int = 4096
    seed: int = 0
    mixer_reps: int = 1
    # Initial state: "dicke" = weight-k basis state followed by an exact, weight-preserving
    # Dicke-state preparation (uniform superposition over all k-subsets; default);
    # "basis" = the weight-k computational basis state itself (as originally specified).
    init_state: str = "dicke"
    # Only used with init_state="basis": prepend U_M(beta_0) before the first cost layer,
    # otherwise the first cost layer acts on an eigenstate and is a pure global phase.
    pre_mixer: bool = True
    rhobeg: float = 0.5

    def resolved_maxiter(self) -> int:
        if self.maxiter is not None:
            return int(self.maxiter)
        return 150 if self.p == 1 else 300

    def to_dict(self) -> dict:
        d = asdict(self)
        d["maxiter"] = self.resolved_maxiter()
        return d


@dataclass(frozen=True)
class SAConfig:
    steps: int = 200
    seed: int = 0
    calibration_swaps: int = 20
    t_end_ratio: float = 1e-3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunSettings:
    """Everything needed to reproduce one problem instance."""

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    k: int = DEFAULT_K
    weights: Weights = field(default_factory=Weights)
    tau: float = DEFAULT_TAU
