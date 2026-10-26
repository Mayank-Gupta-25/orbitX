"""Phase 2b: download + verify + extract the ESA Kelvins Collision Avoidance dataset from Zenodo.

Zenodo record 10.5281/zenodo.4463683 (CC-BY-4.0). Skips work that is already done.

Usage:  python data/download_kelvins.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/api/records/4463683/files/Collision%20Avoidance%20Challenge%20-%20Dataset.zip/content"
MD5 = "d19dc8875229f2f6893253c38adddc87"
SIZE = 221_128_642


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    dest = ROOT / "data" / "raw" / "kelvins"
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("train_data.csv")):
        print("train_data.csv already present -> nothing to do")
        return 0
    zpath = dest / "kelvins_dataset.zip"
    if not zpath.exists() or zpath.stat().st_size != SIZE:
        print(f"Downloading {SIZE / 1e6:.0f} MB from Zenodo ...")
        urllib.request.urlretrieve(URL, zpath)
    if md5(zpath) != MD5:
        zpath.unlink()
        print("MD5 mismatch - corrupted download deleted; run again.")
        return 1
    print("MD5 OK. Extracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    for inner in list(dest.rglob("*.zip")):
        if inner != zpath:
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.parent)
    print("Done:", next(dest.rglob("train_data.csv")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
