"""Position-uncertainty model: how the combined 1-sigma position error shrinks as TCA approaches.

Calibrated from real ESA CDMs (Kelvins). Two layers:
  * CovarianceProfile : global median sigma(tau) per RTN axis (used to extrapolate / for TLE events)
  * event_sigma       : an individual event's own sigma-vs-time-to-TCA curve (interpolated in log space)

tau is always "days until TCA" here, matching the Kelvins `time_to_tca` column.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TAU_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])


@dataclass
class CovarianceProfile:
    tau_days: np.ndarray       # (K,) bin centres, ascending
    log10_sigma: np.ndarray    # (K, 3) median log10 combined sigma [m] for R, T, N

    def sigma(self, tau_days: float) -> np.ndarray:
        """Median combined sigma [m] (3,) at tau_days (clamped to the calibrated range)."""
        return np.array([10.0 ** np.interp(tau_days, self.tau_days, self.log10_sigma[:, i]) for i in range(3)])

    def ratio(self, tau_days: float, tau_ref: float) -> np.ndarray:
        """sigma(tau) / sigma(tau_ref), per axis."""
        return self.sigma(tau_days) / self.sigma(tau_ref)

    def save(self, path) -> None:
        np.savez(path, tau_days=self.tau_days, log10_sigma=self.log10_sigma)

    @classmethod
    def load(cls, path) -> "CovarianceProfile":
        d = np.load(Path(path))
        return cls(d["tau_days"], d["log10_sigma"])

    @classmethod
    def fit(cls, series: pd.DataFrame, edges: np.ndarray = DEFAULT_TAU_EDGES, min_rows: int = 50) -> "CovarianceProfile":
        """Fit from a table with columns time_to_tca, sig_r, sig_t, sig_n (combined sigmas, metres)."""
        taus, logs = [], []
        tau = series["time_to_tca"].to_numpy()
        sig = np.log10(series[["sig_r", "sig_t", "sig_n"]].to_numpy())
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (tau >= lo) & (tau < hi)
            if mask.sum() < min_rows:
                continue
            taus.append(float(np.median(tau[mask])))
            logs.append(np.median(sig[mask], axis=0))
        if len(taus) < 3:
            raise ValueError("not enough CDM rows to fit a covariance profile")
        return cls(np.array(taus), np.array(logs))


def event_sigma(tau_days: float, ev_tau: np.ndarray, ev_log10_sigma: np.ndarray, profile: CovarianceProfile) -> np.ndarray:
    """Combined sigma [m] (3,) of one event at tau_days.

    ev_tau: (K,) ascending days-to-TCA of the event's CDMs; ev_log10_sigma: (K, 3).
    Inside the observed range -> interpolate the event's own curve.
    Outside -> scale the nearest observed value by the global median profile ratio.
    """
    tau = max(float(tau_days), 0.0)
    if len(ev_tau) == 0:
        return profile.sigma(tau)
    lo, hi = float(ev_tau[0]), float(ev_tau[-1])
    if tau < lo:
        return 10.0 ** ev_log10_sigma[0] * profile.ratio(tau, lo)
    if tau > hi:
        return 10.0 ** ev_log10_sigma[-1] * profile.ratio(tau, hi)
    return np.array([10.0 ** np.interp(tau, ev_tau, ev_log10_sigma[:, i]) for i in range(3)])
