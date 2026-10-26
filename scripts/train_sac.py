"""Train a Soft Actor-Critic (SAC) agent on continuous OrbitEnv-v0.

Features:
  - Continuous 3D delta_v burn control
  - Off-policy maximum entropy reinforcement learning
  - Automatic entropy coefficient tuning
  - Soft target network updates

Usage:
    python scripts/train_sac.py --timesteps 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitx.env import OrbitEnv  # noqa: E402
from orbitx.generator import EncounterGenerator  # noqa: E402


def make_orbit_env(processed_dir: Path, split: str = "train", seed: int = 0):
    def _init():
        gen = EncounterGenerator.from_processed(processed_dir, split=split, seed=seed)
        return OrbitEnv(generator=gen, discrete_actions=False)
    return _init


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--eval-freq", type=int, default=20_000, help="Timesteps between evaluation runs")
    ap.add_argument("--n-eval-episodes", type=int, default=15, help="Number of evaluation episodes")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    ap.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "models" / "sac")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "sac")
    args = ap.parse_args()

    processed_dir = ROOT / "data" / "processed"
    if not (processed_dir / "events.parquet").exists():
        sys.exit(f"Processed data not found under {processed_dir}. Run data/kelvins_stats.py first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f"🛰️  OrbitX: Training SAC Agent (Continuous Actions)")
    print(f"========================================================")
    print(f"Total timesteps: {args.timesteps:,} | Buffer size: {args.buffer_size:,}")
    print(f"Discount factor (gamma): {args.gamma} | Learning rate: {args.lr}")
    print(f"Models directory: {args.output_dir}")
    print(f"Logs directory:   {args.log_dir}\n")

    train_env = DummyVecEnv([make_orbit_env(processed_dir, split="train", seed=args.seed)])
    eval_env = DummyVecEnv([make_orbit_env(processed_dir, split="val", seed=args.seed + 1000)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.output_dir),
        log_path=str(args.log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=40_000,
        save_path=str(args.output_dir / "checkpoints"),
        name_prefix="sac_orbitx",
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=5_000,
        batch_size=128,
        tau=0.005,
        gamma=args.gamma,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback])

    final_path = args.output_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\nSAC Training Complete! Final model saved to: {final_path}\n")


if __name__ == "__main__":
    main()
