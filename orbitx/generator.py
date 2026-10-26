"""Encounter generator: samples realistic conjunctions from processed ESA Kelvins events (or TLE screening).

A `Conjunction` is a fixed fly-by described by: true miss vector at TCA (RTN), relative velocity,
hard-body radius and a sigma-vs-time-to-TCA curve. The satellite's own burns change the miss through
CW dynamics (see dynamics.py); the debris trajectory is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .collision import pc_fast
from .constants import DEFAULT_HBR_M, SECONDS_PER_DAY
from .encounter import bplane_basis, project_covariance, rtn_covariance
from .uncertainty import CovarianceProfile, event_sigma

MISS_COLS = ["miss_r", "miss_t", "miss_n"]
VREL_COLS = ["vrel_r", "vrel_t", "vrel_n"]


@dataclass
class Conjunction:
    tca_s: float                    # time of closest approach, seconds since episode start
    miss_rtn: np.ndarray            # TRUE miss (debris - satellite) at TCA [m], RTN
    vrel_rtn: np.ndarray            # relative velocity [m/s], RTN
    hbr: float                      # combined hard-body radius [m]
    ev_tau_days: np.ndarray         # (K,) ascending
    ev_log10_sigma: np.ndarray      # (K, 3)
    profile: CovarianceProfile
    event_id: object = None
    source: str = "kelvins"
    kelvins_risk: float = float("nan")
    hard: bool = False
    E: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.miss_rtn = np.asarray(self.miss_rtn, dtype=float)
        self.vrel_rtn = np.asarray(self.vrel_rtn, dtype=float)
        self.E = bplane_basis(self.vrel_rtn)

    # ---- uncertainty ------------------------------------------------------------------------
    def sigma_rtn(self, tau_s: float) -> np.ndarray:
        """Combined 1-sigma position error [m] (R, T, N) when tau_s seconds remain before TCA."""
        return event_sigma(tau_s / SECONDS_PER_DAY, self.ev_tau_days, self.ev_log10_sigma, self.profile)

    def sigma_final_rtn(self) -> np.ndarray:
        """Sigma of the event's last real CDM: the best knowledge, used to score the TRUE Pc."""
        return 10.0 ** self.ev_log10_sigma[0]

    # ---- risk ---------------------------------------------------------------------------------
    def true_pc(self, delta_r_sat=None) -> float:
        """Pc of the TRUE geometry (after the satellite is displaced by delta_r_sat) with final-CDM sigma."""
        m = self.miss_rtn if delta_r_sat is None else self.miss_rtn - np.asarray(delta_r_sat, dtype=float)
        c2 = project_covariance(rtn_covariance(self.sigma_final_rtn()), self.E)
        return pc_fast(self.E.T @ m, c2, self.hbr)

    def sample_estimate(self, tau_s: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """A noisy 'CDM': miss estimate (RTN) = truth + N(0, sigma(tau)), plus the sigma that applies."""
        sig = self.sigma_rtn(tau_s)
        return self.miss_rtn + rng.normal(0.0, sig), sig

    def estimated_pc(self, miss_hat_rtn, sig_rtn, delta_r_sat=None) -> float:
        """Pc the AGENT can compute from an estimate and its reported sigma."""
        m = np.asarray(miss_hat_rtn, dtype=float)
        if delta_r_sat is not None:
            m = m - np.asarray(delta_r_sat, dtype=float)
        c2 = project_covariance(rtn_covariance(sig_rtn), self.E)
        return pc_fast(self.E.T @ m, c2, self.hbr)


class EncounterGenerator:
    """Samples conjunctions (bootstrap of real events) with optional high-risk oversampling."""

    def __init__(
        self,
        events: pd.DataFrame,
        series: Dict[object, Tuple[np.ndarray, np.ndarray]],
        profile: CovarianceProfile,
        source: str = "kelvins",
        hbr_range: Tuple[float, float] = (DEFAULT_HBR_M / 2.0, DEFAULT_HBR_M * 2.0),
        p_hard: float = 0.5,
        hard_k_max: float = 3.0,
        seed: Optional[int] = None,
    ):
        if len(events) == 0:
            raise ValueError("no events to sample from")
        self.source = source
        self.hbr_range = hbr_range
        self.p_hard = p_hard
        self.hard_k_max = hard_k_max
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self.series = series
        self._pool_ids = list(series.keys())
        self._ids = events["event_id"].tolist()
        self._miss = events[MISS_COLS].to_numpy(dtype=float)
        self._vrel = events[VREL_COLS].to_numpy(dtype=float)
        self._risk = events["final_risk"].to_numpy(dtype=float) if "final_risk" in events else np.full(len(events), np.nan)

    @classmethod
    def from_processed(cls, processed_dir, split: str = "train", **kwargs) -> "EncounterGenerator":
        """split in {'train','val','test'} (Kelvins events) or 'tle' (SGP4-screened events)."""
        processed = Path(processed_dir)
        ser = pd.read_parquet(processed / "cdm_sigma_series.parquet")
        profile = CovarianceProfile.load(processed / "cov_profile.npz")
        series = {}
        for eid, g in ser.groupby("event_id"):
            g = g.sort_values("time_to_tca").drop_duplicates("time_to_tca")
            series[eid] = (g["time_to_tca"].to_numpy(), np.log10(g[["sig_r", "sig_t", "sig_n"]].to_numpy()))
        if split == "tle":
            events = pd.read_parquet(processed / "tle_conjunctions.parquet")
            events = events.assign(event_id=events["event_id"].astype(str))
            kwargs.setdefault("source", "tle")
        else:
            events = pd.read_parquet(processed / "events.parquet")
            events = events[events["split"] == split]
        return cls(events, series, profile, **kwargs)

    def sample_conjunction(self, tca_s: float, rng: Optional[np.random.Generator] = None) -> Conjunction:
        rng = rng or self.rng
        i = int(rng.integers(len(self._ids)))
        eid = self._ids[i]
        miss, vrel = self._miss[i].copy(), self._vrel[i].copy()
        if eid in self.series:
            tau, logsig = self.series[eid]
        else:                                           # TLE events have no CDMs: borrow a real curve
            tau, logsig = self.series[self._pool_ids[int(rng.integers(len(self._pool_ids)))]]
        basis = bplane_basis(vrel)
        hard = False
        if rng.random() < self.p_hard:                  # synthetic high-risk encounter
            c2 = project_covariance(rtn_covariance(10.0 ** logsig[0]), basis)
            sigma_eff = float(np.linalg.det(c2)) ** 0.25
            k = rng.uniform(0.0, self.hard_k_max)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            miss = basis @ (k * sigma_eff * np.array([np.cos(phi), np.sin(phi)]))
            hard = True
        else:
            miss = basis @ (basis.T @ miss)             # enforce perpendicular-to-v_rel miss
        return Conjunction(
            tca_s=float(tca_s), miss_rtn=miss, vrel_rtn=vrel,
            hbr=float(rng.uniform(*self.hbr_range)),
            ev_tau_days=tau, ev_log10_sigma=logsig, profile=self.profile,
            event_id=eid, source=self.source, kelvins_risk=float(self._risk[i]), hard=hard,
        )

    def sample_episode(
        self,
        horizon_s: float = 72 * 3600.0,
        n_conj: Optional[int] = None,
        min_tca_s: Optional[float] = None,
        min_sep_s: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Conjunction]:
        """1..3 conjunctions with staggered TCAs in [min_tca_s, horizon_s - 1h]."""
        rng = rng or self.rng
        n = int(n_conj) if n_conj is not None else int(rng.integers(1, 4))

        max_tca = max(horizon_s - 3600.0, 0.5 * horizon_s)
        if min_tca_s is None:
            min_tca = min(12.0 * 3600.0, 0.2 * max_tca)
        else:
            min_tca = min(min_tca_s, 0.8 * max_tca) if min_tca_s >= max_tca else min_tca_s

        if min_sep_s is None:
            sep = min(6.0 * 3600.0, (max_tca - min_tca) / (n + 1))
        else:
            sep = min_sep_s if (max_tca - min_tca) > n * min_sep_s else (max_tca - min_tca) / (n + 1)

        tcas: List[float] = []
        for _ in range(500):
            if len(tcas) == n:
                break
            t = float(rng.uniform(min_tca, max_tca))
            if all(abs(t - u) >= sep for u in tcas):
                tcas.append(t)
        return [self.sample_conjunction(t, rng) for t in sorted(tcas)]

