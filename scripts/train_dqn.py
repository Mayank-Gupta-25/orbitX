"""Train a Deep Q-Network (DQN) agent on discrete OrbitEnv-v0.

Features:
  - Double Q-learning with target network
  - Replay buffer and epsilon-greedy exploration schedule
  - Periodic evaluation on validation set
  - Model checkpointing

Usage:
    python scripts/train_dqn.py --timesteps 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitx.env import OrbitEnv  # noqa: E402
from orbitx.generator import EncounterGenerator  # noqa: E402


def make_orbit_env(processed_dir: Path, split: str = "train", seed: int = 0):
    def _init():
        gen = EncounterGenerator.from_processed(processed_dir, split=split, seed=seed)
        return OrbitEnv(generator=gen, discrete_actions=True)
    return _init


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--eval-freq", type=int, default=20_000, help="Timesteps between evaluation runs")
    ap.add_argument("--n-eval-episodes", type=int, default=15, help="Number of evaluation episodes")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    ap.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "models" / "dqn")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "dqn")
    args = ap.parse_args()

    processed_dir = ROOT / "data" / "processed"
    if not (processed_dir / "events.parquet").exists():
        sys.exit(f"Processed data not found under {processed_dir}. Run data/kelvins_stats.py first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f"🛰️  OrbitX: Training DQN Agent (Discrete(13))")
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
        name_prefix="dqn_orbitx",
    )

    policy_kwargs = dict(net_arch=[256, 256])
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=5_000,
        batch_size=128,
        tau=1.0,
        target_update_interval=2_000,
        exploration_fraction=0.25,
        exploration_final_eps=0.02,
        gamma=args.gamma,
        train_freq=4,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback])

    final_path = args.output_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\nDQN Training Complete! Final model saved to: {final_path}\n")


if __name__ == "__main__":
    main()
