"""
OrbitX Master Launcher
=======================
Single command entry point for the entire OrbitX project:
  - 3D WebGL Mission Control Console (Browser)
  - Space Station Terminal HUD (ANSI Console)
  - Benchmark Evaluation (PPO, DQN, Baselines)
  - Full Test Suite (pytest)
  - PDF Research Report Compilation

Usage:
    python run.py             # Interactive menu (Default: 3D Mission Control)
    python run.py --dashboard # Directly launch 3D WebGL Dashboard
    python run.py --terminal  # Directly launch Terminal HUD
    python run.py --bench     # Directly run Benchmark Evaluation
    python run.py --test      # Directly run Pytest Suite
    python run.py --report    # Directly build Research PDF Report
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Locate repository root (supports running from parent or repo directory)
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "orbitX" / "orbitx").exists():
    REPO_ROOT = SCRIPT_DIR / "orbitX"
elif (SCRIPT_DIR / "orbitx").exists():
    REPO_ROOT = SCRIPT_DIR
else:
    REPO_ROOT = SCRIPT_DIR

sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║  {BOLD}{WHITE}🚀  ORBITX : AUTONOMOUS SATELLITE COLLISION AVOIDANCE ENGINE  🛰️{RESET}{CYAN}      ║
║  {YELLOW}Deep Reinforcement Learning & Astrodynamics Simulation Environment{RESET}{CYAN}          ║
║  {WHITE}Project Team: Mayank Gupta  &  Soumya Sekhar Parida{RESET}{CYAN}                         ║
║  {WHITE}GitHub: https://github.com/Mayank-Gupta-25/OrbitX{RESET}{CYAN}                           ║
╚══════════════════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def launch_dashboard(port=8000):
    print(f"\n{GREEN}[*] Launching 3D WebGL Mission Control Console on http://localhost:{port} ...{RESET}")
    print(f"{YELLOW}    (Press Ctrl+C to stop the dashboard server){RESET}\n")
    script = REPO_ROOT / "scripts" / "launch_dashboard.py"
    subprocess.run([sys.executable, str(script), "--port", str(port)])


def launch_terminal(agent="ppo"):
    print(f"\n{GREEN}[*] Initializing Space Station Operations Control Room (Terminal HUD)...{RESET}")
    print(f"{YELLOW}    Agent: {agent.upper()} | Press Ctrl+C to return{RESET}\n")
    script = REPO_ROOT / "scripts" / "mission_control.py"
    subprocess.run([sys.executable, str(script), "--agent", agent])


def run_benchmark(episodes=30):
    print(f"\n{GREEN}[*] Executing Unified Benchmark Evaluation ({episodes} episodes / agent / split)...{RESET}")
    script = REPO_ROOT / "scripts" / "evaluate_all.py"
    subprocess.run([sys.executable, str(script), "--episodes", str(episodes), "--split", "both"])


def run_tests():
    print(f"\n{GREEN}[*] Running OrbitX Test Suite via pytest...{RESET}\n")
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"])


def build_report():
    print(f"\n{GREEN}[*] Recompiling 2-Page Research Report & High-Resolution Figures...{RESET}\n")
    fig_script = REPO_ROOT / "reports" / "orbitx_descriptive_report" / "scripts" / "gen_figures.py"
    build_script = REPO_ROOT / "reports" / "orbitx_descriptive_report" / "scripts" / "build_report.py"

    if fig_script.exists():
        subprocess.run([sys.executable, str(fig_script)])
    if build_script.exists():
        subprocess.run([sys.executable, str(build_script)])
    print(f"\n{GREEN}[✓] Report PDF generated at:{RESET}")
    print(f"    {REPO_ROOT / 'reports' / 'orbitx_descriptive_report' / 'report.pdf'}\n")


def interactive_menu():
    print_banner()
    print(f"{BOLD}Select an operation to execute:{RESET}\n")
    print(f"  {CYAN}[1]{RESET} {BOLD}Launch 3D WebGL Mission Control Console{RESET} {GREEN}(Browser Operations Dashboard){RESET} {YELLOW}★ DEFAULT{RESET}")
    print(f"  {CYAN}[2]{RESET} Launch Space Station Terminal HUD (Live ANSI Telemetry & B-Plane Radar)")
    print(f"  {CYAN}[3]{RESET} Run Benchmark Evaluation (PPO, DQN, and Baselines B0–B5 on Kelvins & TLE splits)")
    print(f"  {CYAN}[4]{RESET} Run OrbitX Automated Test Suite (60 unit tests)")
    print(f"  {CYAN}[5]{RESET} Rebuild Publication 2-Page Research Report & Figures (report.pdf)")
    print(f"  {CYAN}[6]{RESET} Exit\n")

    try:
        choice = input(f"{YELLOW}Enter your choice [1-6, press Enter for 1]: {RESET}").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    if choice in ("", "1"):
        launch_dashboard()
    elif choice == "2":
        launch_terminal()
    elif choice == "3":
        run_benchmark()
    elif choice == "4":
        run_tests()
    elif choice == "5":
        build_report()
    elif choice == "6":
        print("Goodbye.")
    else:
        print(f"{YELLOW}Invalid choice, defaulting to 3D Mission Control Console...{RESET}")
        launch_dashboard()


def main():
    parser = argparse.ArgumentParser(description="OrbitX Single-Command Launcher")
    parser.add_argument("--dashboard", action="store_true", help="Launch 3D WebGL Mission Control Console")
    parser.add_argument("--terminal", action="store_true", help="Launch Terminal HUD Console")
    parser.add_argument("--bench", action="store_true", help="Run benchmark evaluation")
    parser.add_argument("--test", action="store_true", help="Run pytest suite")
    parser.add_argument("--report", action="store_true", help="Recompile 2-page research PDF report")
    parser.add_argument("--port", type=int, default=8000, help="Port for web dashboard")
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo", "dqn", "cara", "coast"])
    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard(args.port)
    elif args.terminal:
        launch_terminal(args.agent)
    elif args.bench:
        run_benchmark()
    elif args.test:
        run_tests()
    elif args.report:
        build_report()
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
