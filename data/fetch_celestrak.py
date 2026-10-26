"""Phase 2b: rate-limit-safe CelesTrak downloader (OMM/CSV format).

CelesTrak only refreshes GP data every 2 hours and firewalls IPs that make repeated failing requests
(>50 HTTP errors in 2 h). This script therefore:
  * skips any group that already has a downloaded file (data is FROZEN for reproducibility),
  * with --refresh only re-downloads files older than 2 hours,
  * stops IMMEDIATELY on the first HTTP error (never retries, never loops).

Usage:
    python data/fetch_celestrak.py                       # the default debris groups + active
    python data/fetch_celestrak.py --groups fengyun-1c-debris
    python data/fetch_celestrak.py --refresh
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = ["cosmos-2251-debris", "iridium-33-debris", "fengyun-1c-debris", "cosmos-1408-debris", "resource"]
URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=CSV"
MIN_AGE_HOURS = 2.0


def existing_files(raw_dir: Path, group: str):
    return sorted(raw_dir.glob(f"{group}_*.csv"))


def download(group: str, out_path: Path) -> int:
    req = urllib.request.Request(URL.format(group=group), headers={"User-Agent": "OrbitX-student-project"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    if "OBJECT_NAME" not in body.splitlines()[0]:
        raise RuntimeError(f"unexpected response for {group}: {body[:200]!r}")
    out_path.write_text(body, encoding="utf-8")
    return max(len(body.splitlines()) - 1, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "celestrak")
    ap.add_argument("--refresh", action="store_true", help="re-download groups whose newest file is > 2 h old")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")

    for group in args.groups:
        files = existing_files(args.out, group)
        if files:
            age_h = (time.time() - files[-1].stat().st_mtime) / 3600.0
            if not args.refresh or age_h < MIN_AGE_HOURS:
                print(f"SKIP  {group:<22} already have {files[-1].name} (age {age_h:.1f} h)")
                continue
        out_path = args.out / f"{group}_{stamp}.csv"
        try:
            n = download(group, out_path)
        except urllib.error.HTTPError as e:
            print(f"STOP  {group}: HTTP {e.code} {e.reason}. Open the URL in a browser to read CelesTrak's message. "
                  f"Not retrying (retries can get your IP blocked).")
            return 1
        except Exception as e:                                        # network down, bad content, ...
            print(f"STOP  {group}: {e}")
            return 1
        print(f"OK    {group:<22} {n:>6} objects -> {out_path}")
        time.sleep(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
