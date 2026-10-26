"""Train a Proximal Policy Optimization (PPO) agent on OrbitEnv-v0.

Features:
  - Vectorised environments with DummyVecEnv
  - Observation and reward normalisation with VecNormalize
  - Evaluation callback on the held-out validation set
  - Checkpointing best model and VecNormalize statistics
  - TensorBoard logging

Usage:
    python scripts/train_ppo.py --timesteps 300000 --n-envs 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitx.env import OrbitEnv  # noqa: E402
from orbitx.generator import EncounterGenerator  # noqa: E402


def make_orbit_env(processed_dir: Path, split: str = "train", seed: int = 0, discrete: bool = True):
    def _init():
        gen = EncounterGenerator.from_processed(processed_dir, split=split, seed=seed)
        env = OrbitEnv(generator=gen, discrete_actions=discrete)
        return env
    return _init


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=300_000, help="Total training timesteps")
    ap.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--continuous", action="store_true", help="Use continuous action space instead of discrete")
    ap.add_argument("--eval-freq", type=int, default=25_000, help="Timesteps between evaluation runs")
    ap.add_argument("--n-eval-episodes", type=int, default=20, help="Number of evaluation episodes")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--gamma", type=float, default=0.995, help="Discount factor for long horizons")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "models" / "ppo")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "ppo")
    args = ap.parse_args()

    processed_dir = ROOT / "data" / "processed"
    if not (processed_dir / "events.parquet").exists():
        sys.exit(f"Processed data not found under {processed_dir}. Run data/kelvins_stats.py first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    discrete = not args.continuous
    action_type = "continuous" if args.continuous else "discrete"
    print(f"\n========================================================")
    print(f"🛰️  OrbitX: Training PPO Agent ({action_type} actions)")
    print(f"========================================================")
    print(f"Total timesteps: {args.timesteps:,} | Parallel envs: {args.n_envs}")
    print(f"Discount factor (gamma): {args.gamma} | Learning rate: {args.lr}")
    print(f"Models directory: {args.output_dir}")
    print(f"Logs directory:   {args.log_dir}\n")

    # 1. Training Environment with VecNormalize
    env_fns = [
        make_orbit_env(processed_dir, split="train", seed=args.seed + i, discrete=discrete)
        for i in range(args.n_envs)
    ]
    train_vec_env = DummyVecEnv(env_fns)
    train_vec_env = VecNormalize(
        train_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=100.0, gamma=args.gamma
    )

    # 2. Evaluation Environment (Validation Split)
    eval_fn = [make_orbit_env(processed_dir, split="val", seed=args.seed + 1000, discrete=discrete)]
    eval_vec_env = DummyVecEnv(eval_fn)
    eval_vec_env = VecNormalize(
        eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False
    )

    # 3. Callbacks
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=str(args.output_dir),
        log_path=str(args.log_dir),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(args.output_dir / "checkpoints"),
        name_prefix="ppo_orbitx",
    )

    # 4. Neural Network Policy & Model
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    model = PPO(
        "MlpPolicy",
        train_vec_env,
        learning_rate=args.lr,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        verbose=1,
    )

    # 5. Train
    model.learn(total_timesteps=args.timesteps, callback=[eval_callback, checkpoint_callback])

    # 6. Save final model and VecNormalize statistics
    final_model_path = args.output_dir / "final_model.zip"
    norm_path = args.output_dir / "vec_normalize.pkl"
    model.save(str(final_model_path))
    train_vec_env.save(str(norm_path))

    print(f"\nTraining Complete!")
    print(f"Final model saved to:      {final_model_path}")
    print(f"Best model saved to:       {args.output_dir / 'best_model.zip'}")
    print(f"VecNormalize stats saved: {norm_path}\n")


if __name__ == "__main__":
    main()
