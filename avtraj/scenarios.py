"""Driving scenarios: a straight two-lane road with circular obstacles.

Road frame: x points along the road, y to the left.  The ego lane is centred
at y = 0 and the left lane at y = 4; the drivable band is y in [-2, 6].
Obstacles are placed slightly off the lane centre on purpose: an obstacle
centred exactly on the initial straight-line guess makes the gradient of the
squared-distance constraint vanish there (a symmetric saddle).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Scenario:
    name: str
    obstacles: np.ndarray                     # (M, 3) rows [ox, oy, radius]
    x_init: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 10.0]))
    y_ref: float = 0.0                        # lane centre to track
    v_ref: float = 10.0                       # desired speed [m/s]
    road: tuple = (-2.0, 6.0)                 # drivable band in y [m]
    T: float = 5.0                            # horizon length [s]
    N: int = 50                               # number of intervals

    @property
    def dt(self):
        return self.T / self.N

    @property
    def M(self):
        return len(self.obstacles)

    def with_horizon(self, N):
        return replace(self, N=N)


def single(N=50):
    """One static obstacle blocking the ego lane."""
    return Scenario("single", np.array([[20.0, -0.3, 1.5]]), N=N)


def slalom(N=50):
    """Alternating obstacles: move to the left lane, then back to the right."""
    obs = np.array([[15.0, -0.3, 1.5],
                    [32.0, 4.3, 1.5],
                    [46.0, -0.3, 1.2]])
    return Scenario("slalom", obs, N=N)


def narrow(N=50):
    """Two obstacles leave a narrow gap around y = 2 between the lanes."""
    obs = np.array([[22.0, -1.0, 1.5],
                    [22.0, 5.0, 1.5],
                    [38.0, -0.6, 1.0]])
    return Scenario("narrow", obs, N=N)


def trap(N=50):
    """Like `narrow`, but the third obstacle sits just *above* the lane centre,
    so its constraint gradient pushes a lane-centre initial guess towards the
    road edge, where passing is impossible: local solvers started there tend
    to converge to a point of locally minimal infeasibility."""
    obs = np.array([[22.0, -1.0, 1.5],
                    [22.0, 5.0, 1.5],
                    [36.0, 0.6, 1.0]])
    return Scenario("trap", obs, N=N)


def random_clutter(M, seed, N=50):
    """M random obstacles ahead of the vehicle (some instances may be infeasible)."""
    rng = np.random.default_rng(seed)
    obs = np.column_stack([
        rng.uniform(12.0, 46.0, M),
        rng.uniform(-1.5, 5.5, M),
        rng.uniform(0.6, 1.4, M),
    ])
    return Scenario(f"random_M{M}_s{seed}", obs, N=N)


SCENARIOS = {"single": single, "slalom": slalom, "narrow": narrow, "trap": trap}


def get(name, N=50):
    """Look up a named scenario, or parse 'random_M<M>_s<seed>'."""
    if name in SCENARIOS:
        return SCENARIOS[name](N=N)
    if name.startswith("random_M"):
        m, s = name[len("random_M"):].split("_s")
        return random_clutter(int(m), int(s), N=N)
    raise KeyError(name)
