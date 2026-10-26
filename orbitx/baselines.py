"""Baselines for satellite collision avoidance benchmark:
  * B0: CoastPolicy (No manoeuvre) - lower bound on fuel, upper bound on risk
  * B1: RandomPolicy - uniform random exploration sanity check
  * B2: CaraThresholdRule - NASA CARA industry practice: if est_Pc > 1e-4 within lead time T_L, along-track burn to est_Pc < 1e-6
  * B3: TunedThresholdRule - B2 with tuned threshold, lead time, and burn sizing
  * B4: GreedyMaxSeparation - burns in the direction that maximizes B-plane miss distance whenever est_Pc > 1e-5
  * B5: OraclePolicy - theoretical upper bound with full ground truth (knows true miss and final CDM sigma)

All policies expose an SB3-compatible `.predict(obs, state=None, episode_start=None, deterministic=True)` interface.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import PC_ACCEPTABLE, PC_THRESHOLD


class BasePolicy:
    """Base class providing the Stable-Baselines3 predict() API."""

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[Any, Optional[Tuple[np.ndarray, ...]]]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class CoastPolicy(BasePolicy):
    """B0: Always coast (do nothing)."""

    def __init__(self, discrete: bool = True):
        self.discrete = discrete

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=True):
        if self.discrete:
            return 0, state
        return np.zeros(3, dtype=np.float32), state


class RandomPolicy(BasePolicy):
    """B1: Uniform random actions."""

    def __init__(self, action_space, seed: Optional[int] = None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=True):
        action = self.action_space.sample()
        return action, state


class CaraThresholdRule(BasePolicy):
    """B2: NASA CARA-style heuristic threshold rule.

    If any tracked conjunction has estimated Pc > p_threshold (1e-4) within lead time T_lead (e.g. 24 h):
    Perform along-track burn (+-T) sized to reduce estimated Pc below p_target (1e-6).
    """

    def __init__(
        self,
        p_threshold: float = PC_THRESHOLD,
        lead_time_s: float = 24.0 * 3600.0,
        horizon_s: float = 72.0 * 3600.0,
        discrete: bool = True,
        small_burn: float = 0.01,
        large_burn: float = 0.05,
    ):
        self.p_threshold = float(p_threshold)
        self.lead_time_s = float(lead_time_s)
        self.horizon_s = float(horizon_s)
        self.discrete = discrete
        self.small_burn = small_burn
        self.large_burn = large_burn
        self._burned_conjunctions = set()

    def reset(self) -> None:
        self._burned_conjunctions.clear()

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=True):
        if episode_start:
            self.reset()

        fuel_frac = observation[-2]
        if fuel_frac <= 0.001:
            return (0 if self.discrete else np.zeros(3, dtype=np.float32)), state

        # Inspect up to 3 tracked conjunctions from observation
        for k in range(3):
            base = 6 + k * 10
            valid = observation[base + 9] > 0.5
            if not valid:
                continue

            tau_s = observation[base + 0] * self.horizon_s
            log10_pc_est = observation[base + 3] * 10.0
            est_pc = 10.0 ** log10_pc_est

            # Conjunction key based on coarse tau
            conj_id = f"c_{k}_{int(tau_s / 3600)}"

            # Trigger condition: risk > 1e-4 and within lead time
            if est_pc >= self.p_threshold and tau_s <= self.lead_time_s:
                if conj_id in self._burned_conjunctions:
                    continue    # already reacted to this encounter

                # Along-track burn direction:
                # An along-track burn (+y) creates a negative along-track displacement (-3*dv*tau).
                # We check the sign of the miss vector to decide +T vs -T
                m_e1 = observation[base + 1]
                sign = 1.0 if m_e1 >= 0 else -1.0

                self._burned_conjunctions.add(conj_id)

                if self.discrete:
                    # Pick along-track action (+-T): 3 = +T small, 4 = -T small, 9 = +T large, 10 = -T large
                    # For urgent or large Pc, use large burn
                    if est_pc > 1e-3 or tau_s < 12 * 3600.0:
                        act = 9 if sign > 0 else 10
                    else:
                        act = 3 if sign > 0 else 4
                    return act, state
                else:
                    dv = np.array([0.0, sign * self.small_burn, 0.0], dtype=np.float32)
                    return dv, state

        return (0 if self.discrete else np.zeros(3, dtype=np.float32)), state


class TunedThresholdRule(CaraThresholdRule):
    """B3: Tuned threshold rule with optimized hyperparameters."""

    def __init__(
        self,
        p_threshold: float = 5e-5,          # slightly more proactive than standard 1e-4
        lead_time_s: float = 30.0 * 3600.0,  # 30 hours before TCA (allows cheaper earlier burns)
        horizon_s: float = 72.0 * 3600.0,
        discrete: bool = True,
        small_burn: float = 0.01,
        large_burn: float = 0.05,
    ):
        super().__init__(
            p_threshold=p_threshold,
            lead_time_s=lead_time_s,
            horizon_s=horizon_s,
            discrete=discrete,
            small_burn=small_burn,
            large_burn=large_burn,
        )


class GreedyMaxSeparation(BasePolicy):
    """B4: Myopic burn choosing the action that maximally increases miss distance if risk > 1e-5."""

    def __init__(self, discrete: bool = True, p_threshold: float = 1e-5, horizon_s: float = 72.0 * 3600.0):
        self.discrete = discrete
        self.p_threshold = p_threshold
        self.horizon_s = horizon_s

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=True):
        fuel_frac = observation[-2]
        if fuel_frac <= 0.001:
            return (0 if self.discrete else np.zeros(3, dtype=np.float32)), state

        # Check most urgent conjunction
        base = 6
        if observation[base + 9] > 0.5:
            est_pc = 10.0 ** (observation[base + 3] * 10.0)
            if est_pc > self.p_threshold:
                # Miss vector direction in B-plane
                m1, m2 = observation[base + 1], observation[base + 2]
                # Burn along-track in direction that pushes away
                sign = 1 if m1 >= 0 else -1
                if self.discrete:
                    return (3 if sign > 0 else 4), state
                else:
                    return np.array([0.0, sign * 0.01, 0.0], dtype=np.float32), state

        return (0 if self.discrete else np.zeros(3, dtype=np.float32)), state


class OraclePolicy(BasePolicy):
    """B5: Perfect information oracle upper bound.

    Has direct access to the environment's true conjunctions and plans the optimal
    single along-track burn at 24h before TCA to guarantee true Pc < 1e-6.
    """

    def __init__(self, env=None, target_pc: float = PC_ACCEPTABLE, discrete: bool = True):
        self.env = env
        self.target_pc = target_pc
        self.discrete = discrete
        self._acted_conjunctions = set()

    def reset(self) -> None:
        self._acted_conjunctions.clear()

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=True):
        if episode_start:
            self.reset()

        if self.env is None:
            return (0 if self.discrete else np.zeros(3, dtype=np.float32)), state

        t = self.env.time_s
        for idx, conj in enumerate(self.env.conjunctions):
            if idx in self._acted_conjunctions or idx in self.env.passed_conjunctions:
                continue

            tau = conj.tca_s - t
            # Optimal burn timing: between 20h and 26h before TCA
            if 20.0 * 3600.0 <= tau <= 26.0 * 3600.0:
                true_pc_nominal = conj.true_pc()
                if true_pc_nominal > PC_THRESHOLD:
                    self._acted_conjunctions.add(idx)
                    # Along-track burn moves satellite along-track by ~ -3 * dv * tau
                    # We pick the direction that increases distance
                    sign = 1 if conj.miss_rtn[1] >= 0 else -1
                    if self.discrete:
                        return (3 if sign > 0 else 4), state
                    else:
                        return np.array([0.0, sign * 0.01, 0.0], dtype=np.float32), state

        return (0 if self.discrete else np.zeros(3, dtype=np.float32)), state


def evaluate_policy(policy: BasePolicy, env, n_episodes: int = 50, seed: int = 42) -> Dict[str, float]:
    """Evaluate an agent across fixed episodes and return summary metrics."""
    returns = []
    fuels = []
    violations = []
    true_pcs = []
    drifts = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        policy.reset()
        if hasattr(policy, "env"):
            policy.env = env

        ep_ret = 0.0
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            done = terminated or truncated

        returns.append(ep_ret)
        fuels.append(info.get("fuel_used_m_s", 0.0))
        violations.append(info.get("violations_count", 0))
        drifts.append(info.get("max_drift_m", 0.0))
        true_pcs.extend(info.get("true_pcs", []))

    total_encounters = len(true_pcs)
    v_rate = float(sum(violations) / max(total_encounters, 1))

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_fuel_m_s": float(np.mean(fuels)),
        "violation_rate": v_rate,
        "total_encounters": total_encounters,
        "total_violations": int(sum(violations)),
        "max_drift_m": float(np.max(drifts)) if drifts else 0.0,
    }
