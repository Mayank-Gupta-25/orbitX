"""Phase 2a: turn the raw ESA Kelvins CDM table into compact artefacts for OrbitX.

Outputs (in --out, default data/processed/):
    events.parquet          one row per event: final-CDM geometry, final risk, train/val/test split
    cdm_sigma_series.parquet per-event combined sigma vs time-to-TCA (all CDMs)
    cov_profile.npz         global median sigma(tau) profile
    kelvins_report.json     data statistics + validation of our Pc engine against ESA's `risk`

Usage:  python data/kelvins_stats.py [--raw PATH_TO_train_data.csv] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitx.collision import log10_pc, pc_fast                                    # noqa: E402
from orbitx.encounter import bplane_basis, project_covariance, rtn_covariance      # noqa: E402
from orbitx.uncertainty import CovarianceProfile                                   # noqa: E402

WANTED = [
    "event_id", "mission_id", "time_to_tca", "risk", "miss_distance", "relative_speed",
    "relative_position_r", "relative_position_t", "relative_position_n",
    "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n", "c_object_type",
]
NUMERIC = [c for c in WANTED if c not in ("mission_id", "c_object_type")]
HIGH_RISK = -6.0          # log10(Pc) above which an event counts as "high risk"
MIN_REL_SPEED = 100.0     # [m/s] slower encounters violate the straight-line fly-by assumption


def find_train_csv(raw_dir: Path) -> Path:
    hits = sorted(raw_dir.rglob("train_data.csv"))
    if not hits:
        raise FileNotFoundError(f"train_data.csv not found under {raw_dir}")
    return hits[0]


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda c: c in WANTED)


def assign_splits(events: pd.DataFrame, seed: int = 42, fractions=(0.70, 0.15, 0.15)) -> np.ndarray:
    """Train/val/test split by event, stratified on high-risk so every split has risky events."""
    rng = np.random.default_rng(seed)
    split = np.empty(len(events), dtype=object)
    hard = events["hard"].to_numpy()
    for flag in (True, False):
        idx = np.flatnonzero(hard == flag)
        rng.shuffle(idx)
        n_tr, n_va = int(round(fractions[0] * len(idx))), int(round(fractions[1] * len(idx)))
        split[idx[:n_tr]], split[idx[n_tr:n_tr + n_va]], split[idx[n_tr + n_va:]] = "train", "val", "test"
    return split


def validate_pc(events: pd.DataFrame, hbrs=(5.0, 10.0, 20.0, 40.0)) -> dict:
    """Compare our Pc engine with ESA's own `risk` (log10 Pc) at each event's final CDM."""
    out = {}
    logs = {h: [] for h in hbrs}
    for row in events.itertuples(index=False):
        vrel = np.array([row.vrel_r, row.vrel_t, row.vrel_n])
        basis = bplane_basis(vrel)
        m2 = basis.T @ np.array([row.miss_r, row.miss_t, row.miss_n])
        c2 = project_covariance(rtn_covariance([row.sig_r, row.sig_t, row.sig_n]), basis)
        for h in hbrs:
            logs[h].append(log10_pc(pc_fast(m2, c2, h)))
    risk = events["final_risk"].to_numpy()
    informative = risk > -20.0                     # Kelvins floors tiny risks at -30
    for h in hbrs:
        ours = np.array(logs[h])
        rho = spearmanr(ours[informative], risk[informative]).correlation if informative.sum() > 10 else float("nan")
        out[f"hbr_{h:g}m"] = {
            "spearman": float(rho),
            "median_abs_dlog10": float(np.median(np.abs(ours[informative] - risk[informative]))),
            "median_signed_dlog10": float(np.median(ours[informative] - risk[informative])),
        }
    out["n_informative_events"] = int(informative.sum())
    return out


def build_artifacts(raw: pd.DataFrame, seed: int = 42) -> dict:
    """Pure function (testable): raw CDM table -> {events, series, profile, report}."""
    report = {"raw_rows": int(len(raw)), "raw_events": int(raw["event_id"].nunique())}
    df = raw.dropna(subset=NUMERIC).copy()
    df = df[(df["time_to_tca"] >= 0)]
    for p in ("t", "c"):
        df = df[(df[f"{p}_sigma_r"] > 0) & (df[f"{p}_sigma_t"] > 0) & (df[f"{p}_sigma_n"] > 0)]
    report["rows_after_cleaning"] = int(len(df))
    for ax in "rtn":
        df[f"sig_{ax}"] = np.sqrt(df[f"t_sigma_{ax}"] ** 2 + df[f"c_sigma_{ax}"] ** 2)

    series = df[["event_id", "time_to_tca", "sig_r", "sig_t", "sig_n"]].sort_values(["event_id", "time_to_tca"])
    profile = CovarianceProfile.fit(series)

    last = df.loc[df.groupby("event_id")["time_to_tca"].idxmin()]
    n_cdm = df.groupby("event_id").size()
    first_tau = df.groupby("event_id")["time_to_tca"].max()
    events = pd.DataFrame({
        "event_id": last["event_id"].to_numpy(),
        "mission_id": last["mission_id"].to_numpy() if "mission_id" in last else -1,
        "n_cdm": n_cdm.loc[last["event_id"]].to_numpy(),
        "first_tau": first_tau.loc[last["event_id"]].to_numpy(),
        "final_tau": last["time_to_tca"].to_numpy(),
        "final_risk": last["risk"].to_numpy(),
        "miss_distance": last["miss_distance"].to_numpy(),
        "relative_speed": last["relative_speed"].to_numpy(),
        "miss_r": last["relative_position_r"].to_numpy(),
        "miss_t": last["relative_position_t"].to_numpy(),
        "miss_n": last["relative_position_n"].to_numpy(),
        "vrel_r": last["relative_velocity_r"].to_numpy(),
        "vrel_t": last["relative_velocity_t"].to_numpy(),
        "vrel_n": last["relative_velocity_n"].to_numpy(),
        "sig_r": last["sig_r"].to_numpy(), "sig_t": last["sig_t"].to_numpy(), "sig_n": last["sig_n"].to_numpy(),
        "c_object_type": last["c_object_type"].to_numpy() if "c_object_type" in last else "UNKNOWN",
    })
    n_before = len(events)
    events = events[events["relative_speed"] >= MIN_REL_SPEED].reset_index(drop=True)
    report["events_dropped_low_relative_speed"] = int(n_before - len(events))
    events["hard"] = events["final_risk"] > HIGH_RISK
    events["split"] = assign_splits(events, seed)
    series = series[series["event_id"].isin(events["event_id"])]

    vrel = events[["vrel_r", "vrel_t", "vrel_n"]].to_numpy()
    miss = events[["miss_r", "miss_t", "miss_n"]].to_numpy()
    cosang = np.abs(np.einsum("ij,ij->i", vrel, miss)) / (np.linalg.norm(vrel, axis=1) * np.linalg.norm(miss, axis=1) + 1e-12)
    report.update({
        "events": int(len(events)),
        "high_risk_events(risk>-6)": int(events["hard"].sum()),
        "split_counts": events["split"].value_counts().to_dict(),
        "miss_distance_m_quantiles(5,25,50,75,95)": [float(x) for x in np.percentile(events["miss_distance"], [5, 25, 50, 75, 95])],
        "relative_speed_m_s_quantiles(5,50,95)": [float(x) for x in np.percentile(events["relative_speed"], [5, 50, 95])],
        "final_sigma_m_median(R,T,N)": [float(events[c].median()) for c in ("sig_r", "sig_t", "sig_n")],
        "median_|cos(angle(miss, v_rel))|": float(np.median(cosang)),
        "median_|miss_distance-|relative_position||_m": float(np.median(np.abs(events["miss_distance"] - np.linalg.norm(miss, axis=1)))),
        "profile_tau_days": [float(x) for x in profile.tau_days],
        "profile_median_sigma_m(R,T,N)": [[float(10 ** v) for v in row] for row in profile.log10_sigma],
        "pc_validation_vs_ESA_risk": validate_pc(events),
    })
    return {"events": events, "series": series, "profile": profile, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=None, help="path to train_data.csv (auto-detected under data/raw/kelvins)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_path = args.raw or find_train_csv(ROOT / "data" / "raw" / "kelvins")
    print(f"Reading {raw_path}")
    art = build_artifacts(load_raw(raw_path), seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    art["events"].to_parquet(args.out / "events.parquet", index=False)
    art["series"].to_parquet(args.out / "cdm_sigma_series.parquet", index=False)
    art["profile"].save(args.out / "cov_profile.npz")
    (args.out / "kelvins_report.json").write_text(json.dumps(art["report"], indent=2, default=str))
    print(json.dumps(art["report"], indent=2, default=str))
    print(f"\nWrote events.parquet, cdm_sigma_series.parquet, cov_profile.npz, kelvins_report.json -> {args.out}")


if __name__ == "__main__":
    main()
