import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from orbitx.constants import PC_ACCEPTABLE, PC_THRESHOLD
from orbitx.env import OrbitEnv
from orbitx.generator import EncounterGenerator


@pytest.fixture(scope="module")
def real_generator():
    from pathlib import Path
    proc = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not (proc / "events.parquet").exists():
        pytest.skip("processed data not found in scratch")
    return EncounterGenerator.from_processed(proc, split="train", seed=42)


def test_env_conforms_to_gymnasium_spec(real_generator):
    env = OrbitEnv(generator=real_generator)
    # Gymnasium's strict environment checker validates shapes, dtypes, spaces, and resets
    check_env(env)


def test_env_observation_dimensions_and_types(real_generator):
    env = OrbitEnv(generator=real_generator)
    obs, info = env.reset(seed=123)
    assert obs.shape == (38,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert 0.0 <= obs[-2] <= 1.0     # fuel fraction
    assert 0.0 <= obs[-1] <= 1.0     # time fraction


def test_env_discrete_step(real_generator):
    env = OrbitEnv(generator=real_generator)
    obs, _ = env.reset(seed=1)
    # Take coast action
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert next_obs.shape == (38,)
    assert isinstance(reward, float)
    assert not terminated
    assert info["fuel_used_m_s"] == 0.0


def test_env_burn_consumes_fuel(real_generator):
    env = OrbitEnv(generator=real_generator, dv_budget=0.1)
    obs, _ = env.reset(seed=2)
    # Action 9 is large along-track burn (+T at 0.05 m/s)
    obs, r1, _, _, info1 = env.step(9)
    assert pytest.approx(info1["fuel_used_m_s"], abs=1e-5) == 0.05
    assert pytest.approx(info1["fuel_remaining"], abs=1e-5) == 0.05

    # Another large burn uses the remaining 0.05 m/s
    obs, r2, _, _, info2 = env.step(9)
    assert pytest.approx(info2["fuel_remaining"], abs=1e-5) == 0.0

    # Further burn should do nothing because fuel is exhausted
    obs, r3, _, _, info3 = env.step(9)
    assert info3["step_fuel"] == 0.0
    assert info3["fuel_remaining"] == 0.0


def test_env_continuous_action_mode(real_generator):
    env = OrbitEnv(generator=real_generator, discrete_actions=False)
    assert isinstance(env.action_space, gym.spaces.Box)
    obs, _ = env.reset(seed=3)
    # Continuous small burn along radial
    action = np.array([0.01, 0.0, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (38,)
    assert pytest.approx(info["fuel_used_m_s"], abs=1e-4) == 0.01


def test_env_runs_to_full_horizon(real_generator):
    env = OrbitEnv(generator=real_generator, horizon_s=3600.0 * 2, dt_s=720.0)  # 10 steps
    obs, _ = env.reset(seed=4)
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(0)
        done = terminated or truncated
        steps += 1
    assert steps == 10
    assert terminated
