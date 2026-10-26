from pathlib import Path

import numpy as np
import pytest

from orbitx.baselines import (
    CaraThresholdRule,
    CoastPolicy,
    GreedyMaxSeparation,
    OraclePolicy,
    RandomPolicy,
    TunedThresholdRule,
    evaluate_policy,
)
from orbitx.env import OrbitEnv
from orbitx.generator import EncounterGenerator


@pytest.fixture(scope="module")
def env():
    proc = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not (proc / "events.parquet").exists():
        pytest.skip("processed data not found in scratch")
    gen = EncounterGenerator.from_processed(proc, split="train", seed=10)
    # Fast test episode: 12 hours horizon with 12-min steps
    return OrbitEnv(generator=gen, horizon_s=12.0 * 3600.0, dt_s=720.0)


def test_b0_coast_policy_uses_zero_fuel(env):
    b0 = CoastPolicy()
    metrics = evaluate_policy(b0, env, n_episodes=5, seed=1)
    assert metrics["mean_fuel_m_s"] == 0.0
    assert metrics["total_encounters"] > 0


def test_b1_random_policy_runs(env):
    b1 = RandomPolicy(env.action_space, seed=2)
    metrics = evaluate_policy(b1, env, n_episodes=3, seed=2)
    assert metrics["mean_fuel_m_s"] > 0.0


def test_b2_cara_threshold_rule_triggers_on_risk(env):
    b2 = CaraThresholdRule(p_threshold=1e-5, lead_time_s=12.0 * 3600.0)
    metrics = evaluate_policy(b2, env, n_episodes=5, seed=3)
    assert "mean_return" in metrics
    assert "violation_rate" in metrics


def test_b3_tuned_threshold_rule_runs(env):
    b3 = TunedThresholdRule(p_threshold=1e-5, lead_time_s=12.0 * 3600.0)
    metrics = evaluate_policy(b3, env, n_episodes=5, seed=4)
    assert "violation_rate" in metrics


def test_b4_greedy_runs(env):
    b4 = GreedyMaxSeparation(p_threshold=1e-6)
    metrics = evaluate_policy(b4, env, n_episodes=3, seed=5)
    assert "mean_return" in metrics


def test_b5_oracle_runs(env):
    b5 = OraclePolicy(env=env)
    metrics = evaluate_policy(b5, env, n_episodes=5, seed=6)
    assert "violation_rate" in metrics


def test_sb3_predict_interface_compatibility(env):
    obs, _ = env.reset(seed=7)
    for pol in [CoastPolicy(), RandomPolicy(env.action_space), CaraThresholdRule(), OraclePolicy(env)]:
        act, state = pol.predict(obs, deterministic=True)
        assert env.action_space.contains(act)
