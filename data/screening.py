"""Phase 2c: SGP4 conjunction screening of real CelesTrak debris against real target satellites.

Produces an INDEPENDENT test set of real fly-by geometries (relative velocity, approach angle, miss
vector in the target's RTN frame) from public orbit data, used to test generalisation (experiment E5).

Method (per target satellite):
  1. altitude pre-filter: keep debris whose perigee/apogee band overlaps the target's;
  2. coarse screening: propagate on a `--step`-second grid, find local minima of the range;
  3. refinement: re-propagate +-1 step around each minimum on a 0.1 s grid -> TCA, miss vector, v_rel;
  4. express miss vector and relative velocity in the target's RTN frame.

TLE/GP data is only km-accurate, so these are GEOMETRY samples (not precise Pc values). Covariances are
borrowed from real Kelvins CDM curves by the generator.

Usage:
    python data/screening.py --targets 8 --days 7
    python data/screening.py --target-name CARTOSAT --targets 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitx.constants import MU_EARTH, R_EARTH   # noqa: E402

DEBRIS_GROUPS = ["cosmos-2251-debris", "iridium-33-debris", "fengyun-1c-debris", "cosmos-1408-debris"]
V_REL_MAX_KM_S = 16.0


def latest_file(raw_dir: Path, group: str) -> Path:
    files = sorted(raw_dir.glob(f"{group}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no {group}_*.csv in {raw_dir} (run data/fetch_celestrak.py)")
    return files[-1]


def load_catalog(path: Path) -> list:
    """Parse an OMM/CSV file into a list of dicts: name, catnr, sat (Satrec), epoch_jd, perigee_km, apogee_km, ecc."""
    out = []
    with open(path, newline="") as f:
        for row in omm.parse_csv(f):
            sat = Satrec()
            try:
                omm.initialize(sat, row)
            except Exception:
                continue
            if sat.error != 0:
                continue
            n = float(row["MEAN_MOTION"]) * 2.0 * np.pi / 86400.0
            a = (MU_EARTH / n ** 2) ** (1.0 / 3.0)
            ecc = float(row["ECCENTRICITY"])
            out.append({
                "name": row["OBJECT_NAME"], "catnr": int(row["NORAD_CAT_ID"]), "sat": sat,
                "epoch_jd": sat.jdsatepoch + sat.jdsatepochF,
                "perigee_km": (a * (1 - ecc) - R_EARTH) / 1e3, "apogee_km": (a * (1 + ecc) - R_EARTH) / 1e3,
                "ecc": ecc, "incl_deg": float(row["INCLINATION"]),
            })
    return out


def select_targets(active: list, start_jd: float, n: int, seed: int, alt_min: float, alt_max: float,
                   name_contains: str | None, max_age_days: float = 3.0) -> list:
    pool = []
    for o in active:
        nm = o["name"].upper()
        if "DEB" in nm or "R/B" in nm:
            continue
        if name_contains and name_contains.upper() not in nm:
            continue
        if o["ecc"] > 0.02 or not (alt_min <= o["perigee_km"] <= alt_max):
            continue
        if start_jd - o["epoch_jd"] > max_age_days:
            continue
        pool.append(o)
    if name_contains:
        return pool[:n]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))[:n]
    return [pool[i] for i in idx]


def rtn_components(r_t: np.ndarray, v_t: np.ndarray, vec: np.ndarray) -> np.ndarray:
    r_hat = r_t / np.linalg.norm(r_t)
    n_hat = np.cross(r_t, v_t)
    n_hat /= np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    return np.array([vec @ r_hat, vec @ t_hat, vec @ n_hat])


def refine(tgt, deb, jd0: float, fr_c: float, step_s: float):
    """Fine search +-step around a coarse minimum. Returns (miss_km, tca_fr, r_t, v_t, r_d, v_d) or None."""
    fr = fr_c + np.linspace(-step_s, step_s, 401) / 86400.0
    jd = np.full(fr.shape, jd0)
    e1, rt, vt = tgt.sgp4_array(jd, fr)
    e2, rd, vd = deb.sgp4_array(jd, fr)
    if np.any(e1 != 0) or np.any(e2 != 0):
        return None
    d = np.linalg.norm(rd - rt, axis=1)
    k = int(np.argmin(d))
    return float(d[k]), float(fr[k]), rt[k], vt[k], rd[k], vd[k]


def screen_target(target: dict, debris: list, jd0: float, fr0: float, days: float, step_s: float,
                  max_miss_km: float, group: str, chunk: int = 64) -> list:
    n_t = int(days * 86400.0 / step_s) + 1
    fr = fr0 + np.arange(n_t) * step_s / 86400.0
    jd = np.full(n_t, jd0)
    err, r_t, v_t = target["sat"].sgp4_array(jd, fr)
    if np.any(err != 0):
        return []
    coarse_km = 0.5 * V_REL_MAX_KM_S * step_s + max_miss_km
    rows = []
    for lo in range(0, len(debris), chunk):
        block = debris[lo:lo + chunk]
        e, r_d, _ = SatrecArray([b["sat"] for b in block]).sgp4(jd, fr)
        d = np.linalg.norm(r_d - r_t[None], axis=2)
        d[(e != 0) | ~np.isfinite(d)] = np.inf
        inner = d[:, 1:-1]
        mask = (inner < d[:, :-2]) & (inner <= d[:, 2:]) & (inner < coarse_km)
        for s_idx, t_idx in zip(*np.nonzero(mask)):
            res = refine(target["sat"], block[s_idx]["sat"], jd0, fr[t_idx + 1], step_s)
            if res is None:
                continue
            miss_km, fr_tca, rt, vt, rd, vd = res
            if miss_km > max_miss_km:
                continue
            rel_v = (vd - vt) * 1e3
            speed = float(np.linalg.norm(rel_v))
            if speed < 100.0:
                continue
            miss = rtn_components(rt, vt, (rd - rt) * 1e3)
            vrel = rtn_components(rt, vt, rel_v)
            tca_off = (fr_tca - fr0) * 86400.0
            rows.append({
                "event_id": f"{target['catnr']}-{block[s_idx]['catnr']}-{int(tca_off)}",
                "group": group, "target_name": target["name"], "target_catnr": target["catnr"],
                "debris_name": block[s_idx]["name"], "debris_catnr": block[s_idx]["catnr"],
                "tca_offset_s": tca_off, "miss_distance": miss_km * 1e3, "relative_speed": speed,
                "miss_r": miss[0], "miss_t": miss[1], "miss_n": miss[2],
                "vrel_r": vrel[0], "vrel_t": vrel[1], "vrel_n": vrel[2],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "celestrak")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--groups", nargs="+", default=DEBRIS_GROUPS)
    ap.add_argument("--targets-group", type=str, default="resource", help="CelesTrak group the target satellites come from")
    ap.add_argument("--targets", type=int, default=8)
    ap.add_argument("--target-name", type=str, default=None, help="substring of the target satellite name")
    ap.add_argument("--alt-min", type=float, default=600.0, help="target perigee altitude band [km]")
    ap.add_argument("--alt-max", type=float, default=900.0)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--step", type=float, default=20.0, help="coarse time step [s]")
    ap.add_argument("--max-miss-km", type=float, default=20.0)
    ap.add_argument("--max-age-days", type=float, default=7.0, help="drop debris TLEs older than this at start time")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    active = load_catalog(latest_file(args.raw_dir, args.targets_group))
    catalogs = {g: load_catalog(latest_file(args.raw_dir, g)) for g in args.groups}
    start_jd = max(max(o["epoch_jd"] for o in cat) for cat in [active, *catalogs.values()] if cat)
    jd0, fr0 = float(np.floor(start_jd)), float(start_jd - np.floor(start_jd))
    targets = select_targets(active, start_jd, args.targets, args.seed, args.alt_min, args.alt_max, args.target_name)
    print(f"start JD {start_jd:.4f} | {len(active)} candidate targets | targets: {[t['name'] for t in targets]}")
    if not targets:
        sys.exit("no target satellites matched; widen --alt-min/--alt-max or change --target-name")

    all_rows = []
    for group, cat in catalogs.items():
        fresh = [o for o in cat if start_jd - o["epoch_jd"] <= args.max_age_days and o["ecc"] < 0.5]
        print(f"[{group}] {len(cat)} objects, {len(fresh)} with fresh elements")
        for tgt in targets:
            lo, hi = tgt["perigee_km"] - 30.0, tgt["apogee_km"] + 30.0
            cand = [o for o in fresh if o["perigee_km"] <= hi and o["apogee_km"] >= lo and o["catnr"] != tgt["catnr"]]
            t0 = time.time()
            rows = screen_target(tgt, cand, jd0, fr0, args.days, args.step, args.max_miss_km, group)
            all_rows += rows
            print(f"   {tgt['name']:<24} alt {tgt['perigee_km']:.0f} km | {len(cand):>4} candidates | "
                  f"{len(rows):>3} conjunctions < {args.max_miss_km:.0f} km | {time.time() - t0:.0f}s")

    df = pd.DataFrame(all_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "tle_conjunctions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nTotal {len(df)} conjunctions -> {out_path}")
    if len(df):
        print(df.groupby("group")["miss_distance"].describe()[["count", "min", "50%", "max"]].round(0))
        print("conjunctions with miss < 2 km:", int((df["miss_distance"] < 2000).sum()))


if __name__ == "__main__":
    main()
