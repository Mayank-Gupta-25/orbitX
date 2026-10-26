"""Gymnasium environment for autonomous satellite collision avoidance (OrbitEnv-v0).

MDP formulation:
  - Observation space (38 dims, continuous):
      * 6 own state: delta_r [m] / 1000, delta_v [m/s] / 1.0 (RTN deviation from slot)
      * 30 conjunction features (3 tracked conjunctions x 10 features, sorted by tau, zero-padded):
          [tau/T_max, miss_hat_e1/1000, miss_hat_e2/1000, log10(est_Pc)/10, log10(s1)/3, log10(s2)/3,
           v_rel/10000, pred_miss_e1/1000, pred_miss_e2/1000, valid_flag]
      * 2 resources: remaining fuel fraction, remaining episode time fraction
  - Action space:
      * Discrete(13): 0 = Coast; 1..6 = small (+-R, +-T, +-N at 0.01 m/s); 7..12 = large (+-R, +-T, +-N at 0.05 m/s)
      * Continuous: Box(-0.05, 0.05, (3,)) with deadzone (< 0.002 m/s -> coast)
  - Reward function:
      * Running risk penalty: -w1 * max(0, log10(est_Pc) - log10(P_acc)) * dt / (tau + dt)
      * Running fuel penalty: -w2 * ||dv|| / dv_ref
      * Running slot drift penalty: -w3 * ||delta_r|| / L_ref
      * Terminal TCA event reward: -W_col * true_Pc - W_viol * 1[true_Pc > 1e-4] + B_safe * 1[true_Pc < P_acc]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .collision import log10_pc, pc_fast
from .constants import (
    DEFAULT_ALTITUDE_M,
    PC_ACCEPTABLE,
    PC_FLOOR,
    PC_THRESHOLD,
    SECONDS_PER_DAY,
)
from .dynamics import apply_impulse, burn_displacement, cw_stm, mean_motion
from .encounter import project_covariance, rtn_covariance
from .generator import Conjunction, EncounterGenerator


class OrbitEnv(gym.Env):
    """Autonomous satellite collision avoidance environment in LEO."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(
        self,
        generator: Optional[EncounterGenerator] = None,
        discrete_actions: bool = True,
        dt_s: float = 720.0,                    # 12 minutes (~1/8 LEO orbit)
        horizon_s: float = 72.0 * 3600.0,       # 72 hours (3 days)
        dv_budget: float = 1.0,                 # 1.0 m/s total fuel budget
        small_burn_m_s: float = 0.01,           # small impulse magnitude
        large_burn_m_s: float = 0.05,           # large impulse magnitude
        altitude_m: float = DEFAULT_ALTITUDE_M,
        k_max_tracked: int = 3,                 # maximum number of conjunctions in observation
        cdm_interval_s: float = 8.0 * 3600.0,   # CDMs arrive every 8 hours
        w_risk: float = 1.0,
        w_fuel: float = 5.0,
        w_drift: float = 0.1,
        w_col: float = 10000.0,
        w_viol: float = 50.0,
        b_safe: float = 5.0,
        p_acc: float = PC_ACCEPTABLE,
        p_threshold: float = PC_THRESHOLD,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.generator = generator
        self.discrete_actions = discrete_actions
        self.dt_s = float(dt_s)
        self.horizon_s = float(horizon_s)
        self.dv_budget = float(dv_budget)
        self.small_burn = float(small_burn_m_s)
        self.large_burn = float(large_burn_m_s)
        self.altitude_m = float(altitude_m)
        self.n = mean_motion(self.altitude_m)
        self.k_max = int(k_max_tracked)
        self.cdm_interval_s = float(cdm_interval_s)
        self.render_mode = render_mode

        # Reward weights
        self.w_risk = float(w_risk)
        self.w_fuel = float(w_fuel)
        self.w_drift = float(w_drift)
        self.w_col = float(w_col)
        self.w_viol = float(w_viol)
        self.b_safe = float(b_safe)
        self.p_acc = float(p_acc)
        self.p_threshold = float(p_threshold)

        # Precompute 1-step STM
        self.stm_dt = cw_stm(self.dt_s, self.n)

        # Action space
        if self.discrete_actions:
            self.action_space = spaces.Discrete(13)
            # Map index to 3-vector (m/s)
            self._action_table = np.zeros((13, 3), dtype=np.float32)
            # 1..6: small burns (+-R, +-T, +-N)
            dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
            for i, d in enumerate(dirs):
                self._action_table[1 + i] = np.array(d, dtype=np.float32) * self.small_burn
                self._action_table[7 + i] = np.array(d, dtype=np.float32) * self.large_burn
        else:
            self.action_space = spaces.Box(
                low=-self.large_burn, high=self.large_burn, shape=(3,), dtype=np.float32
            )

        # Observation space: 6 (own state) + 10 * k_max (conjunctions) + 2 (resources) = 38
        obs_dim = 6 + 10 * self.k_max + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State attributes
        self.time_s: float = 0.0
        self.sat_state: np.ndarray = np.zeros(6, dtype=np.float64)   # [x, y, z, vx, vy, vz] RTN
        self.fuel_remaining: float = self.dv_budget
        self.conjunctions: List[Conjunction] = []
        self.active_cdms: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}  # idx -> (miss_hat, sig, last_update_t)
        self.passed_conjunctions: set = set()
        self.episode_stats: Dict[str, Any] = {}

    def _get_burn_vector(self, action: Any) -> np.ndarray:
        """Convert action into 3D delta_v vector."""
        if self.discrete_actions:
            a = int(action)
            if not 0 <= a < 13:
                raise ValueError(f"discrete action {a} out of bounds [0, 12]")
            dv = self._action_table[a]
        else:
            dv = np.asarray(action, dtype=np.float32)
            if np.linalg.norm(dv) < 0.002:       # deadzone: burns < 2 mm/s are treated as coasting
                dv = np.zeros(3, dtype=np.float32)
            else:
                dv = np.clip(dv, -self.large_burn, self.large_burn)
        return dv.astype(np.float64)

    def _update_cdm_estimates(self, force: bool = False) -> None:
        """Update noisy CDM measurements if cdm_interval_s has elapsed."""
        for i, conj in enumerate(self.conjunctions):
            if i in self.passed_conjunctions:
                continue
            tau = conj.tca_s - self.time_s
            if tau <= 0.0:
                continue
            need_update = force or (i not in self.active_cdms)
            if not need_update:
                _, _, last_t = self.active_cdms[i]
                if self.time_s - last_t >= self.cdm_interval_s:
                    need_update = True
            if need_update:
                m_hat, sig = conj.sample_estimate(max(tau, 0.0), self.np_random)
                self.active_cdms[i] = (m_hat, sig, self.time_s)

    def _get_obs(self) -> np.ndarray:
        """Construct the 38-dim observation vector."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        # 1. Own state (scaled: positions in km, velocities in m/s)
        obs[0:3] = self.sat_state[0:3] / 1000.0
        obs[3:6] = self.sat_state[3:6] / 1.0

        # 2. Conjunctions (sorted by time-to-TCA, active only)
        active_indices = [
            i for i, c in enumerate(self.conjunctions)
            if i not in self.passed_conjunctions and c.tca_s >= self.time_s
        ]
        active_indices.sort(key=lambda idx: self.conjunctions[idx].tca_s - self.time_s)

        slot = 0
        for idx in active_indices[:self.k_max]:
            conj = self.conjunctions[idx]
            tau_s = max(conj.tca_s - self.time_s, 0.0)
            m_hat, sig, _ = self.active_cdms[idx]

            # Predicted satellite position at TCA if coasting from now
            dr_tca = burn_displacement(self.sat_state[3:6], tau_s, self.n) + cw_stm(tau_s, self.n)[:3, :3] @ self.sat_state[:3]

            # Project estimated miss to B-plane
            m2 = conj.E.T @ m_hat
            c2 = project_covariance(rtn_covariance(sig), conj.E)
            est_pc = pc_fast(m2, c2, conj.hbr)

            # Principal sigmas in B-plane
            lam = np.linalg.eigvalsh(c2)
            s1, s2 = np.sqrt(max(lam[1], 1e-6)), np.sqrt(max(lam[0], 1e-6))

            # Predicted miss in B-plane if coasting (debris miss - sat position at TCA)
            m_coast2 = conj.E.T @ (m_hat - dr_tca)

            base = 6 + slot * 10
            obs[base + 0] = float(np.clip(tau_s / self.horizon_s, 0.0, 1.0))
            obs[base + 1] = float(m2[0] / 1000.0)
            obs[base + 2] = float(m2[1] / 1000.0)
            obs[base + 3] = float(np.clip(log10_pc(est_pc) / 10.0, -3.0, 0.0))
            obs[base + 4] = float(np.clip(np.log10(s1) / 3.0, 0.0, 3.0))
            obs[base + 5] = float(np.clip(np.log10(s2) / 3.0, 0.0, 3.0))
            obs[base + 6] = float(np.linalg.norm(conj.vrel_rtn) / 10000.0)
            obs[base + 7] = float(m_coast2[0] / 1000.0)
            obs[base + 8] = float(m_coast2[1] / 1000.0)
            obs[base + 9] = 1.0         # valid tracking flag
            slot += 1

        # 3. Resources
        obs[-2] = float(np.clip(self.fuel_remaining / self.dv_budget, 0.0, 1.0))
        obs[-1] = float(np.clip(1.0 - (self.time_s / self.horizon_s), 0.0, 1.0))
        return obs

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.time_s = 0.0
        self.sat_state = np.zeros(6, dtype=np.float64)
        self.fuel_remaining = self.dv_budget
        self.passed_conjunctions.clear()
        self.active_cdms.clear()

        # Sample conjunctions for this episode
        if self.generator is not None:
            n_c = options.get("n_conj", None) if options else None
            self.conjunctions = self.generator.sample_episode(
                horizon_s=self.horizon_s, n_conj=n_c, rng=self.np_random
            )
        else:
            self.conjunctions = []

        self.episode_stats = {
            "fuel_used_m_s": 0.0,
            "manoeuvres_count": 0,
            "true_pcs": [],
            "violations_count": 0,
            "safe_encounters": 0,
            "max_drift_m": 0.0,
            "conjunctions_count": len(self.conjunctions),
        }

        self._update_cdm_estimates(force=True)
        obs = self._get_obs()
        info = {k: v for k, v in self.episode_stats.items()}
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_time = self.time_s
        next_time = prev_time + self.dt_s

        # 1. Parse and apply burn if fuel is available
        dv = self._get_burn_vector(action)
        dv_norm = float(np.linalg.norm(dv))

        if dv_norm > 0.0:
            if self.fuel_remaining >= dv_norm:
                self.sat_state = apply_impulse(self.sat_state, dv)
                self.fuel_remaining -= dv_norm
                self.episode_stats["fuel_used_m_s"] += dv_norm
                self.episode_stats["manoeuvres_count"] += 1
            else:
                # partial burn with all remaining fuel
                if self.fuel_remaining > 1e-6:
                    scaled_dv = dv * (self.fuel_remaining / dv_norm)
                    self.sat_state = apply_impulse(self.sat_state, scaled_dv)
                    self.episode_stats["fuel_used_m_s"] += self.fuel_remaining
                    self.episode_stats["manoeuvres_count"] += 1
                    dv_norm = self.fuel_remaining
                    self.fuel_remaining = 0.0
                else:
                    dv_norm = 0.0

        # 2. Propagate satellite deviation to next time step
        self.sat_state = self.stm_dt @ self.sat_state
        self.time_s = next_time

        drift_m = float(np.linalg.norm(self.sat_state[:3]))
        if drift_m > self.episode_stats["max_drift_m"]:
            self.episode_stats["max_drift_m"] = drift_m

        # 3. Update active CDMs
        self._update_cdm_estimates()

        # 4. Compute reward
        reward = 0.0

        # Running fuel penalty
        reward -= self.w_fuel * (dv_norm / self.large_burn)

        # Running orbit deviation penalty (drift from slot)
        reward -= self.w_drift * (drift_m / 1000.0)

        # Running risk penalty for active tracked conjunctions
        for idx, (m_hat, sig, _) in self.active_cdms.items():
            if idx in self.passed_conjunctions:
                continue
            conj = self.conjunctions[idx]
            tau = conj.tca_s - self.time_s
            if tau > 0:
                c2 = project_covariance(rtn_covariance(sig), conj.E)
                est_pc = pc_fast(conj.E.T @ m_hat, c2, conj.hbr)
                log_est = log10_pc(est_pc)
                log_acc = np.log10(self.p_acc)
                if log_est > log_acc:
                    urgency = self.dt_s / (tau + self.dt_s)
                    reward -= self.w_risk * (log_est - log_acc) * urgency

        # Conjunction passage event reward (evaluated at true TCA)
        for i, conj in enumerate(self.conjunctions):
            if i not in self.passed_conjunctions and (prev_time < conj.tca_s <= next_time):
                self.passed_conjunctions.add(i)
                # Compute TRUE collision probability at encounter
                # Satellite displacement at TCA
                tau_to_tca = conj.tca_s - prev_time
                dr_at_tca = cw_stm(tau_to_tca, self.n)[:3, :3] @ self.sat_state[:3]
                true_pc = conj.true_pc(delta_r_sat=dr_at_tca)

                self.episode_stats["true_pcs"].append(true_pc)
                # Terminal collision terms
                reward -= self.w_col * true_pc
                if true_pc > self.p_threshold:
                    reward -= self.w_viol
                    self.episode_stats["violations_count"] += 1
                elif true_pc < self.p_acc:
                    reward += self.b_safe
                    self.episode_stats["safe_encounters"] += 1

        # 5. Check termination
        terminated = self.time_s >= self.horizon_s
        truncated = False

        obs = self._get_obs()
        info = {k: v for k, v in self.episode_stats.items()}
        info["fuel_remaining"] = self.fuel_remaining
        info["step_fuel"] = dv_norm

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            print(
                f"T={self.time_s/3600:5.1f}h | Fuel={self.fuel_remaining:5.3f} m/s | "
                f"Drift={np.linalg.norm(self.sat_state[:3]):6.1f}m | "
                f"Passed={len(self.passed_conjunctions)}/{len(self.conjunctions)}"
            )


# Gymnasium environment registration
gym.register(
    id="OrbitEnv-v0",
    entry_point="orbitx.env:OrbitEnv",
)
