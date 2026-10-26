"""OrbitX Terminal Mission Control HUD & Real-Time Conjunction Monitor.

Renders an interactive, real-time aerospace control room dashboard in the terminal
showing spacecraft orbital deviation (RTN), propellant expenditure, threat radar (B-plane),
and live reinforcement learning guidance decisions.

Authors: Mayank Gupta & Soumya Sekhar Parida
Usage:
    python scripts/mission_control.py [--agent ppo|dqn|cara|coast] [--speed 0.05] [--seed 42]
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

# Enable ANSI colors on Windows Console
if os.name == "nt":
    os.system("")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

# Fix for PyTorch 2.4+ on Windows ZipExtFile stream reader
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

from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from orbitx.baselines import CaraThresholdRule, CoastPolicy  # noqa: E402
from orbitx.constants import PC_ACCEPTABLE, PC_THRESHOLD  # noqa: E402
from orbitx.env import OrbitEnv  # noqa: E402
from orbitx.generator import EncounterGenerator  # noqa: E402

# ANSI Color Codes for Aerospace HUD Aesthetics
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_CYAN = "\033[96m"
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_MAGENTA = "\033[95m"
C_WHITE = "\033[97m"


def clear_console() -> None:
    """Clear terminal screen reliably on Windows and Unix."""
    os.system("cls" if os.name == "nt" else "clear")


def progress_bar(val: float, max_val: float, width: int = 18, fill_color=C_CYAN) -> str:
    frac = max(0.0, min(1.0, val / max_val))
    filled = int(round(width * frac))
    bar = "█" * filled + "░" * (width - filled)
    return f"{fill_color}[{bar}]{C_RESET} {val:5.3f}/{max_val:5.3f} m/s ({frac * 100:4.1f}%)"


def risk_gauge(pc: float) -> str:
    if pc < PC_ACCEPTABLE:
        badge = f"{C_GREEN}{C_BOLD}[ SAFE ]{C_RESET}"
        color = C_GREEN
    elif pc < PC_THRESHOLD:
        badge = f"{C_YELLOW}{C_BOLD}[ ELEVATED RISK ]{C_RESET}"
        color = C_YELLOW
    else:
        badge = f"{C_RED}{C_BOLD}[ CRITICAL HAZARD ]{C_RESET}"
        color = C_RED
    return f"{badge} {color}Pc={pc:.1e}{C_RESET}"


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def render_dashboard(
    step_num: int,
    total_steps: int,
    met_s: float,
    sat_state: np.ndarray,
    fuel_rem: float,
    fuel_max: float,
    thruster_status: str,
    target_info: dict,
    est_pc: float,
    action_name: str,
    agent_name: str,
    log_feed: list,
) -> None:
    r_mag = np.linalg.norm(sat_state[:3])
    v_mag = np.linalg.norm(sat_state[3:])

    # Clear terminal in place so it animates cleanly without scrolling
    clear_console()

    w = 84
    h_bar = "═" * (w - 2)
    s_bar = "─" * (w - 2)

    # 1. Operations Header
    print(f"{C_BLUE}{C_BOLD}╔{h_bar}╗{C_RESET}")
    print(f"{C_BLUE}{C_BOLD}║  🛰️  ORBITX SPACE OPERATIONS CENTER (SOC) │ LEO TELEMETRY & CONJUNCTION MONITOR   ║{C_RESET}")
    print(f"{C_BLUE}{C_BOLD}║  STATION: ISRO/ESA CONJUNCTION NODE       │ CONTROLLER: {C_CYAN}{agent_name:<20}{C_BLUE}{C_BOLD} ║{C_RESET}")
    print(f"{C_BLUE}{C_BOLD}╚{h_bar}╝{C_RESET}")

    # 2. Mission Clock
    print(f"{C_WHITE}{C_BOLD} MISSION TIME (MET): {C_CYAN}T+ {format_duration(met_s)}{C_WHITE} │ STEP: {step_num:03d}/{total_steps:03d} │ ORBIT: 550 km LEO{C_RESET}")
    print(f"{C_DIM} {s_bar} {C_RESET}")

    # 3. Spacecraft Telemetry Panel
    print(f"{C_BOLD}{C_WHITE}┌── 📡 SPACECRAFT ONBOARD TELEMETRY (RTN FRAME) {s_bar[44:]}┐{C_RESET}")
    print(f"│  Position Deviation : {C_CYAN}R:{sat_state[0]:+7.1f}m{C_RESET} │ {C_CYAN}T:{sat_state[1]:+7.1f}m{C_RESET} │ {C_CYAN}N:{sat_state[2]:+7.1f}m{C_RESET} │ Slot Drift: {C_YELLOW}{r_mag:6.1f}m{C_RESET}")
    print(f"│  Velocity Deviation : {C_CYAN}Vx:{sat_state[3]:+6.3f}{C_RESET} │ {C_CYAN}Vy:{sat_state[4]:+6.3f}{C_RESET} │ {C_CYAN}Vz:{sat_state[5]:+6.3f}{C_RESET} │ Rel Speed : {C_WHITE}{v_mag:6.3f} m/s{C_RESET}")
    print(f"│  Propellant Delta-V : {progress_bar(fuel_rem, fuel_max, width=20)}")
    print(f"│  Thruster Actuator  : {thruster_status}")
    print(f"{C_BOLD}{C_WHITE}└── {s_bar[2:]}┘{C_RESET}")

    # 4. Threat Radar Panel
    print(f"{C_BOLD}{C_WHITE}┌── 🎯 CONJUNCTION ASSESSMENT & DEBRIS RADAR (B-PLANE) {s_bar[49:]}┐{C_RESET}")
    if target_info:
        deb_name = target_info.get("name", "COSMOS-DEBRIS-FRAG")
        tca_rem = max(0.0, target_info.get("tca_s", 0.0) - met_s)
        rel_v = target_info.get("rel_v_km_s", 11.5)
        miss_m = target_info.get("miss_m", 1500.0)
        hbr = target_info.get("hbr", 10.0)

        print(f"│  Tracked Threat   : {C_RED}{C_BOLD}{deb_name:<26}{C_RESET} │ V_rel: {C_WHITE}{rel_v:5.2f} km/s{C_RESET}")
        print(f"│  Time-to-TCA      : {C_YELLOW}{C_BOLD}{format_duration(tca_rem):<26}{C_RESET} │ Miss : {C_CYAN}{miss_m:8.1f} m (HBR: {hbr:.0f}m){C_RESET}")
        print(f"│  Collision Status : {risk_gauge(est_pc)}")
    else:
        print(f"│  Threat Radar     : {C_GREEN}● NO IMMEDIATE CONJUNCTIONS IN TRACKING WINDOW (< 20 km){C_RESET}")
    print(f"{C_BOLD}{C_WHITE}└── {s_bar[2:]}┘{C_RESET}")

    # 5. Autonomous Decision Log Feed
    print(f"{C_BOLD}{C_WHITE}┌── 📋 AUTONOMOUS GUIDANCE LOG FEED {s_bar[33:]}┐{C_RESET}")
    for item in log_feed[-4:]:
        print(f"│  {item}")
    print(f"{C_BOLD}{C_WHITE}└── {s_bar[2:]}┘{C_RESET}")
    sys.stdout.flush()


def run_mission_control(agent_type: str = "ppo", speed_multiplier: float = 0.05, seed: int = 42) -> None:
    processed_dir = ROOT / "data" / "processed"
    if not (processed_dir / "events.parquet").exists():
        sys.exit("Processed data not found. Run data/kelvins_stats.py first.")

    gen = EncounterGenerator.from_processed(processed_dir, split="test", seed=seed)
    env = OrbitEnv(generator=gen, horizon_s=72.0 * 3600.0, dt_s=720.0)

    # Load agent
    models_dir = ROOT / "models"
    agent_name = "PPO-NEURAL-AGENT"
    vec_norm = None

    if agent_type.lower() == "ppo":
        ppo_path = models_dir / "ppo" / "best_model.zip"
        if not ppo_path.exists():
            ppo_path = models_dir / "ppo" / "final_model.zip"
        if ppo_path.exists():
            model = PPO.load(str(ppo_path))
            norm_path = models_dir / "ppo" / "vec_normalize.pkl"
            if norm_path.exists():
                dummy = DummyVecEnv([lambda: env])
                vec_norm = VecNormalize.load(str(norm_path), dummy)
                vec_norm.training = False
                vec_norm.norm_reward = False
            agent = model
            agent_name = "PPO (Reinforcement Learning)"
        else:
            print("PPO model not found, falling back to CARA heuristic.")
            agent = CaraThresholdRule()
            agent_name = "CARA Threshold Rule"

    elif agent_type.lower() == "dqn":
        dqn_path = models_dir / "dqn" / "best_model.zip"
        if not dqn_path.exists():
            dqn_path = models_dir / "dqn" / "final_model.zip"
        if dqn_path.exists():
            agent = DQN.load(str(dqn_path))
            agent_name = "DQN (Value-Based Q-Net)"
        else:
            agent = CaraThresholdRule()
            agent_name = "CARA Threshold Rule"

    elif agent_type.lower() == "cara":
        agent = CaraThresholdRule()
        agent_name = "CARA Industry Threshold Rule"
    else:
        agent = CoastPolicy()
        agent_name = "Coast / Do Nothing"

    obs, info = env.reset(seed=seed)
    total_steps = int(env.horizon_s // env.dt_s)

    action_labels = {
        0: f"{C_GREEN}IDLE / COASTING (Thrusters Off){C_RESET}",
        1: f"{C_YELLOW}FIRING +R RADIAL THRUSTER (10 mm/s){C_RESET}",
        2: f"{C_YELLOW}FIRING -R RADIAL THRUSTER (10 mm/s){C_RESET}",
        3: f"{C_YELLOW}{C_BOLD}FIRING +T ALONG-TRACK PROGRADE BURN (10 mm/s){C_RESET}",
        4: f"{C_YELLOW}{C_BOLD}FIRING -T ALONG-TRACK RETROGRADE BURN (10 mm/s){C_RESET}",
        5: f"{C_YELLOW}FIRING +N CROSS-TRACK THRUSTER (10 mm/s){C_RESET}",
        6: f"{C_YELLOW}FIRING -N CROSS-TRACK THRUSTER (10 mm/s){C_RESET}",
        7: f"{C_RED}{C_BOLD}HEAVY BURN: +R RADIAL (50 mm/s){C_RESET}",
        8: f"{C_RED}{C_BOLD}HEAVY BURN: -R RADIAL (50 mm/s){C_RESET}",
        9: f"{C_RED}{C_BOLD}HEAVY EMERGENCY BURN: +T ALONG-TRACK (50 mm/s){C_RESET}",
        10: f"{C_RED}{C_BOLD}HEAVY EMERGENCY BURN: -T ALONG-TRACK (50 mm/s){C_RESET}",
        11: f"{C_RED}{C_BOLD}HEAVY BURN: +N CROSS-TRACK (50 mm/s){C_RESET}",
        12: f"{C_RED}{C_BOLD}HEAVY BURN: -N CROSS-TRACK (50 mm/s){C_RESET}",
    }

    log_feed = [
        f"{C_CYAN}[00:00:00]{C_RESET} MISSION START: Spacecraft telemetry nominal at 550 km LEO.",
        f"{C_CYAN}[00:00:00]{C_RESET} GUIDANCE: Autonomous controller initialized ({agent_name}).",
        f"{C_CYAN}[00:00:00]{C_RESET} RADAR-SWEEP: 72-hour conjunction assessment pipeline active.",
    ]

    done = False
    step = 0

    while not done:
        step += 1
        met = env.time_s

        # Predict action
        if vec_norm is not None:
            norm_obs = vec_norm.normalize_obs(obs)
            action, _ = agent.predict(norm_obs, deterministic=True)
        else:
            action, _ = agent.predict(obs, deterministic=True)

        act_idx = int(np.asarray(action).ravel()[0])
        thruster_str = action_labels.get(act_idx, f"THRUST: {act_idx}")

        # Active target info
        target_info = {}
        est_pc = 1e-15
        if env.conjunctions:
            cands = [c for i, c in enumerate(env.conjunctions) if i not in env.passed_conjunctions and c.tca_s >= met]
            if cands:
                c = cands[0]
                idx = env.conjunctions.index(c)
                m_hat, sig, _ = env.active_cdms[idx]
                target_info = {
                    "name": f"DEBRIS-{c.event_id or 'NORAD-44873'}",
                    "tca_s": c.tca_s,
                    "rel_v_km_s": float(np.linalg.norm(c.vrel_rtn) / 1000.0),
                    "miss_m": float(np.linalg.norm(m_hat)),
                    "hbr": c.hbr,
                    "sig_t": float(sig[1]),
                }
                est_pc = c.estimated_pc(m_hat, sig)

        # Logging events
        if act_idx != 0:
            log_feed.append(f"{C_YELLOW}[{format_duration(met)}]{C_RESET} {C_BOLD}MANOEUVRE ENGAGED:{C_RESET} {thruster_str}")
        if target_info and (met < target_info["tca_s"] <= met + env.dt_s):
            log_feed.append(f"{C_GREEN}[{format_duration(met)}]{C_RESET} {C_BOLD}TCA PASSED:{C_RESET} Debris successfully cleared!")

        render_dashboard(
            step_num=step,
            total_steps=total_steps,
            met_s=met,
            sat_state=env.sat_state,
            fuel_rem=env.fuel_remaining,
            fuel_max=env.dv_budget,
            thruster_status=thruster_str,
            target_info=target_info,
            est_pc=est_pc,
            action_name=thruster_str,
            agent_name=agent_name,
            log_feed=log_feed,
        )

        obs, reward, terminated, truncated, info = env.step(act_idx)
        done = terminated or truncated
        time.sleep(speed_multiplier)

    # Final Mission Complete Summary
    print(f"\n{C_GREEN}{C_BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_GREEN}{C_BOLD}║  ✅  MISSION COMPLETE: 72-HOUR SIMULATION WINDOW CONCLUDED                       ║{C_RESET}")
    print(f"{C_GREEN}{C_BOLD}║  TOTAL FUEL EXPENDED : {info.get('fuel_used_m_s', 0.0):6.4f} m/s  │  MANOEUVRES EXECUTED: {info.get('manoeuvres_count', 0):02d}            ║{C_RESET}")
    print(f"{C_GREEN}{C_BOLD}║  COLLISION VIOLATIONS: {info.get('violations_count', 0):02d}          │  PEAK STATION DRIFT : {info.get('max_drift_m', 0.0):6.1f} m         ║{C_RESET}")
    print(f"{C_GREEN}{C_BOLD}╚══════════════════════════════════════════════════════════════════════════════════╝{C_RESET}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", type=str, default="ppo", choices=["ppo", "dqn", "cara", "coast"], help="Agent to run live")
    ap.add_argument("--speed", type=float, default=0.04, help="Playback delay per step in seconds")
    ap.add_argument("--seed", type=int, default=42, help="Encounter episode random seed")
    args = ap.parse_args()

    run_mission_control(agent_type=args.agent, speed_multiplier=args.speed, seed=args.seed)


if __name__ == "__main__":
    main()
