"""Phase 3 Benchmark: Evaluate baselines B0 to B5 on OrbitEnv-v0.

Computes and prints:
  * Mean episode return
  * Mean fuel used (m/s)
  * Risk violation rate (true Pc > 1e-4)
  * Total encounters and violations
  * Max slot drift (m)

Usage:
    python scripts/evaluate_baselines.py --episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitx.baselines import (  # noqa: E402
    CaraThresholdRule,
    CoastPolicy,
    GreedyMaxSeparation,
    OraclePolicy,
    RandomPolicy,
    TunedThresholdRule,
    evaluate_policy,
)
from orbitx.env import OrbitEnv  # noqa: E402
from orbitx.generator import EncounterGenerator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes per baseline")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "tle"])
    args = ap.parse_args()

    processed_dir = ROOT / "data" / "processed"
    if not (processed_dir / "events.parquet").exists():
        sys.exit(f"Processed data not found at {processed_dir}. Run data/kelvins_stats.py first.")

    print(f"Loading EncounterGenerator (split='{args.split}') from {processed_dir}...")
    generator = EncounterGenerator.from_processed(processed_dir, split=args.split, seed=args.seed)
    env = OrbitEnv(generator=generator)

    policies = {
        "B0 (Coast / Do Nothing)": CoastPolicy(),
        "B1 (Random)": RandomPolicy(env.action_space, seed=args.seed),
        "B2 (CARA Threshold Rule)": CaraThresholdRule(),
        "B3 (Tuned Threshold Rule)": TunedThresholdRule(),
        "B4 (Greedy Separation)": GreedyMaxSeparation(),
        "B5 (Oracle Upper Bound)": OraclePolicy(env=env),
    }

    results = []
    print(f"\nEvaluating {len(policies)} baselines across {args.episodes} episodes each...\n")

    for name, policy in policies.items():
        print(f"--> Running {name}...")
        metrics = evaluate_policy(policy, env, n_episodes=args.episodes, seed=args.seed)
        results.append({
            "Baseline": name,
            "Mean Return": f"{metrics['mean_return']:.1f}",
            "Mean Fuel (m/s)": f"{metrics['mean_fuel_m_s']:.4f}",
            "Violation Rate": f"{metrics['violation_rate']*100:.1f}%",
            "Violations": f"{metrics['total_violations']}/{metrics['total_encounters']}",
            "Max Drift (m)": f"{metrics['max_drift_m']:.1f}",
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("🛰️ OrbitX: Phase 3 Baseline Benchmark Results")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
