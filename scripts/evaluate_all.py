"""Unified evaluation benchmark for OrbitX:
Compares Baselines (B0..B5) and trained RL agents (PPO, DQN, SAC) on identical episodes.

Evaluates on:
  1. In-distribution Test Split (ESA Kelvins CDMs)
  2. Out-of-distribution Generalisation Split (CelesTrak SGP4 TLEs)

Usage:
    python scripts/evaluate_all.py --episodes 30
    python scripts/evaluate_all.py --episodes 50 --split tle
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

# Fix for PyTorch 2.4+ on Windows: reading zip streams directly from ZipExtFile
# triggers internal miniz C++ read errors. Buffering into BytesIO enables full seeking.
_orig_torch_load = torch.load


def _safe_torch_load(f, *args, **kwargs):
    if hasattr(f, "read") and not isinstance(f, (io.BytesIO, io.BufferedReader)):
        try:
            content = f.read()
            f = io.BytesIO(content)
        except Exception:
            pass
    return _orig_torch_load(f, *args, **kwargs)


torch.load = _safe_torch_load

from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


class Sb3PolicyWrapper:
    """Wraps an SB3 model (and optional VecNormalize) for the evaluate_policy runner."""

    def __init__(self, model, vec_norm: Optional[VecNormalize] = None):
        self.model = model
        self.vec_norm = vec_norm

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        if self.vec_norm is not None:
            norm_obs = self.vec_norm.normalize_obs(observation)
            return self.model.predict(norm_obs, deterministic=deterministic)
        return self.model.predict(observation, deterministic=deterministic)

    def reset(self):
        pass


def run_benchmark(generator: EncounterGenerator, episodes: int, seed: int) -> pd.DataFrame:
    env_discrete = OrbitEnv(generator=generator, discrete_actions=True)
    env_continuous = OrbitEnv(generator=generator, discrete_actions=False)

    candidates = {
        "B0 (Coast / Do Nothing)": (CoastPolicy(), env_discrete),
        "B1 (Random)": (RandomPolicy(env_discrete.action_space, seed=seed), env_discrete),
        "B2 (CARA Threshold Rule)": (CaraThresholdRule(), env_discrete),
        "B3 (Tuned Threshold Rule)": (TunedThresholdRule(), env_discrete),
        "B4 (Greedy Separation)": (GreedyMaxSeparation(), env_discrete),
        "B5 (Oracle Upper Bound)": (OraclePolicy(env=env_discrete), env_discrete),
    }

    # Add trained RL models if available
    models_dir = ROOT / "models"

    # PPO
    ppo_path = models_dir / "ppo" / "best_model.zip"
    if not ppo_path.exists():
        ppo_path = models_dir / "ppo" / "final_model.zip"
    if ppo_path.exists():
        ppo_model = PPO.load(str(ppo_path))
        norm_path = models_dir / "ppo" / "vec_normalize.pkl"
        vec_norm = None
        if norm_path.exists():
            dummy = DummyVecEnv([lambda: env_discrete])
            vec_norm = VecNormalize.load(str(norm_path), dummy)
            vec_norm.training = False
            vec_norm.norm_reward = False
        candidates["RL: PPO (Trained)"] = (Sb3PolicyWrapper(ppo_model, vec_norm), env_discrete)

    # DQN
    dqn_path = models_dir / "dqn" / "best_model.zip"
    if not dqn_path.exists():
        dqn_path = models_dir / "dqn" / "final_model.zip"
    if dqn_path.exists():
        dqn_model = DQN.load(str(dqn_path))
        candidates["RL: DQN (Trained)"] = (Sb3PolicyWrapper(dqn_model), env_discrete)

    # SAC
    sac_path = models_dir / "sac" / "best_model.zip"
    if not sac_path.exists():
        sac_path = models_dir / "sac" / "final_model.zip"
    if sac_path.exists():
        sac_model = SAC.load(str(sac_path))
        candidates["RL: SAC (Trained)"] = (Sb3PolicyWrapper(sac_model), env_continuous)

    rows = []
    for name, (policy, env) in candidates.items():
        print(f"   Evaluating {name:<26} ...")
        metrics = evaluate_policy(policy, env, n_episodes=episodes, seed=seed)
        rows.append({
            "Method": name,
            "Mean Return": round(metrics["mean_return"], 1),
            "Mean Fuel (m/s)": round(metrics["mean_fuel_m_s"], 4),
            "Violation Rate": f"{metrics['violation_rate']*100:.1f}%",
            "Violations / Total": f"{metrics['total_violations']} / {metrics['total_encounters']}",
            "Max Drift (m)": round(metrics["max_drift_m"], 1),
        })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=30, help="Number of test episodes per agent")
    ap.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    ap.add_argument("--split", type=str, default="both", choices=["test", "tle", "both"], help="Test split")
    args = ap.parse_args()

    processed_dir = ROOT / "data" / "processed"
    if not (processed_dir / "events.parquet").exists():
        sys.exit("Processed data not found. Run data/kelvins_stats.py first.")

    splits_to_run = ["test", "tle"] if args.split == "both" else [args.split]

    all_results = {}
    for split_name in splits_to_run:
        label = "In-Distribution (ESA Kelvins Test Split)" if split_name == "test" else "Generalisation (CelesTrak SGP4 TLE Split)"
        print("\n" + "=" * 90)
        print(f"🛰️  OrbitX Benchmark: {label}")
        print("=" * 90)

        gen = EncounterGenerator.from_processed(processed_dir, split=split_name, seed=args.seed)
        df = run_benchmark(gen, episodes=args.episodes, seed=args.seed)

        print("\n" + df.to_string(index=False) + "\n")
        all_results[split_name] = df.to_dict(orient="records")

    # Save benchmark results
    out_json = processed_dir / "benchmark_comparison.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"Full benchmark results saved to: {out_json}\n")


if __name__ == "__main__":
    main()
