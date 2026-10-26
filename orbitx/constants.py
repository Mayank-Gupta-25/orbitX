"""Physical constants and project-wide defaults. SI units everywhere (m, s, m/s)."""

MU_EARTH = 3.986004418e14          # Earth gravitational parameter [m^3/s^2]
R_EARTH = 6_378_137.0              # Earth equatorial radius [m]
SECONDS_PER_DAY = 86_400.0

DEFAULT_ALTITUDE_M = 550e3         # reference circular orbit altitude [m]
DEFAULT_HBR_M = 10.0               # default combined hard-body radius [m]

PC_FLOOR = 1e-30                   # Pc is never reported below this (avoids log(0))
PC_ACCEPTABLE = 1e-6               # "safe" collision probability
PC_THRESHOLD = 1e-4                # industry manoeuvre threshold (NASA CARA "red")
