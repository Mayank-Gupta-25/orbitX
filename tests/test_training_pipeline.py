from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from orbitx.env import OrbitEnv
from orbitx.generator import EncounterGenerator


@pytest.fixture(scope="module")
def env_fn():
    proc = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not (proc / "events.parquet").exists():
        pytest.skip("processed data not found in data/processed")
    gen = EncounterGenerator.from_processed(proc, split="train", seed=42)

    def _init():
        return OrbitEnv(generator=gen, horizon_s=6.0 * 3600.0, dt_s=720.0)

    return _init


def test_vec_normalize_integration(env_fn, tmp_path):
    vec_env = DummyVecEnv([env_fn])
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    obs = norm_env.reset()
    assert obs.shape == (1, 38)
    assert np.all(np.isfinite(obs))

    act = [0]
    next_obs, reward, dones, infos = norm_env.step(act)
    assert next_obs.shape == (1, 38)
    assert np.all(np.isfinite(next_obs))

    # Save and reload VecNormalize
    save_file = tmp_path / "vec_norm.pkl"
    norm_env.save(str(save_file))
    loaded_norm = VecNormalize.load(str(save_file), DummyVecEnv([env_fn]))
    assert loaded_norm.norm_obs


def test_ppo_short_training_run(env_fn):
    vec_env = DummyVecEnv([env_fn])
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model = PPO(
        "MlpPolicy",
        norm_env,
        n_steps=32,
        batch_size=16,
        n_epochs=2,
        learning_rate=1e-3,
        verbose=0,
    )
    # Train for a small number of steps to verify pipeline integrity
    model.learn(total_timesteps=64)

    obs = norm_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (1,)
    assert 0 <= action[0] < 13


def test_dqn_short_training_run(env_fn):
    vec_env = DummyVecEnv([env_fn])
    model = DQN(
        "MlpPolicy",
        vec_env,
        buffer_size=1000,
        learning_starts=10,
        batch_size=16,
        learning_rate=1e-3,
        verbose=0,
    )
    model.learn(total_timesteps=32)

    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert 0 <= action[0] < 13

