"""Trajectory-optimisation NLP for obstacle avoidance (direct multiple shooting).

Decision vector  z = [x_0, ..., x_N, u_0, ..., u_{N-1}]  (n = 4(N+1) + 2N).

    minimise   dt * sum_k [ w_y (py_k - y_ref)^2 + w_v (v_k - v_ref)^2 + w_th theta_k^2 ]
             + dt * sum_k [ w_a a_k^2 + w_d delta_k^2 ]
             + dt * sum_k [ w_j ((a_k - a_{k-1})/dt)^2 + w_dd ((delta_k - delta_{k-1})/dt)^2 ]
             + w_yN (py_N - y_ref)^2 + w_thN theta_N^2

    s.t.  x_0 = x_init,   x_{k+1} = F(x_k, u_k)                  (equalities, nonlinear)
          a_min <= a_k <= a_max,  |delta_k| <= delta_max           (linear)
          |delta_k - delta_{k-1}| <= ddelta_max * dt               (linear)
          v_min <= v_k <= v_max,  road edges on py_k               (linear)
          (R_j^2 - ||p_k - o_j||^2) / (2 R_j) <= 0                 (nonconvex, concave in p)

The objective is a convex quadratic; all nonconvexity comes from the dynamics
and the collision-avoidance constraints.  The obstacle constraint is divided
by 2R_j so that near the boundary it behaves like  R_j - ||p_k - o_j||  [m].
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from .problem import NLP
from .scenarios import Scenario
from .vehicle import NU, NX, Vehicle, discrete_dynamics


@dataclass(frozen=True)
class Weights:
    w_y: float = 1.0
    w_v: float = 1.0
    w_th: float = 1.0
    w_a: float = 0.1
    w_d: float = 1.0
    w_j: float = 0.01
    w_dd: float = 0.1
    w_yN: float = 5.0
    w_thN: float = 5.0


def unpack(z, N):
    """z -> (X with rows x_k, U with rows u_k)."""
    z = np.asarray(z)
    X = z[: NX * (N + 1)].reshape(N + 1, NX)
    U = z[NX * (N + 1):].reshape(N, NU)
    return X, U


def pack(X, U):
    return np.concatenate([np.asarray(X).ravel(), np.asarray(U).ravel()])


def build_nlp(sc: Scenario, veh: Vehicle = Vehicle(), w: Weights = Weights()) -> NLP:
    N, dt = sc.N, sc.dt
    F = discrete_dynamics(veh, dt)
    X = ca.SX.sym("X", NX, N + 1)
    U = ca.SX.sym("U", NU, N)
    z = ca.vertcat(ca.vec(X), ca.vec(U))

    # ---------------- objective ----------------
    f = 0
    for k in range(1, N + 1):
        f += dt * (w.w_y * (X[1, k] - sc.y_ref) ** 2 + w.w_v * (X[3, k] - sc.v_ref) ** 2
                   + w.w_th * X[2, k] ** 2)
    for k in range(N):
        f += dt * (w.w_a * U[0, k] ** 2 + w.w_d * U[1, k] ** 2)
    for k in range(1, N):
        f += dt * (w.w_j * ((U[0, k] - U[0, k - 1]) / dt) ** 2
                   + w.w_dd * ((U[1, k] - U[1, k - 1]) / dt) ** 2)
    f += w.w_yN * (X[1, N] - sc.y_ref) ** 2 + w.w_thN * X[2, N] ** 2

    # ---------------- equalities: initial state + dynamics ----------------
    h = [X[:, 0] - sc.x_init]
    for k in range(N):
        h.append(X[:, k + 1] - F(X[:, k], U[:, k]))
    h = ca.vertcat(*h)

    # ---------------- inequalities g(z) <= 0 ----------------
    g = []
    for k in range(N):                                   # control bounds
        a, d = U[0, k], U[1, k]
        g += [a - veh.a_max, veh.a_min - a, d - veh.delta_max, -veh.delta_max - d]
    rate = veh.ddelta_max * dt
    for k in range(1, N):                                # steering rate
        dd = U[1, k] - U[1, k - 1]
        g += [dd - rate, -dd - rate]
    for k in range(1, N + 1):                            # speed limits
        g += [X[3, k] - veh.v_max, veh.v_min - X[3, k]]
    y_lo, y_hi = sc.road[0] + veh.radius, sc.road[1] - veh.radius
    for k in range(1, N + 1):                            # road edges
        g += [X[1, k] - y_hi, y_lo - X[1, k]]
    n_lin = len(g)

    obs_rows = []
    for k in range(1, N + 1):                            # collision avoidance
        for j, (ox, oy, r) in enumerate(sc.obstacles):
            R = r + veh.radius
            d2 = (X[0, k] - ox) ** 2 + (X[1, k] - oy) ** 2
            obs_rows.append((len(g), k, j))
            g.append((R ** 2 - d2) / (2 * R))
    g = ca.vertcat(*g) if g else ca.SX(0, 1)

    meta = dict(
        scenario=sc, vehicle=veh, weights=w, N=N, dt=dt, F=F,
        n_lin=n_lin, obs_rows=np.array(obs_rows, dtype=int).reshape(-1, 3),
    )
    return NLP(z, f, h, g, z0=straight_guess(sc, veh), name=sc.name, meta=meta)


def straight_guess(sc: Scenario, veh: Vehicle = Vehicle()):
    """Roll the model forward with zero controls: dynamically feasible, but it
    drives straight through every obstacle in the ego lane."""
    F = discrete_dynamics(veh, sc.dt)
    X = np.zeros((sc.N + 1, NX))
    X[0] = sc.x_init
    for k in range(sc.N):
        X[k + 1] = F(X[k], np.zeros(NU)).full().ravel()
    return pack(X, np.zeros((sc.N, NU)))


def min_clearance(nlp: NLP, z):
    """Smallest distance between the vehicle circle and any obstacle circle [m]
    (negative means collision)."""
    sc, veh, N = nlp.meta["scenario"], nlp.meta["vehicle"], nlp.meta["N"]
    if sc.M == 0:
        return np.inf
    X, _ = unpack(z, N)
    p = X[1:, None, :2]                                   # (N, 1, 2)
    o = sc.obstacles[None, :, :2]                         # (1, M, 2)
    dist = np.linalg.norm(p - o, axis=2) - sc.obstacles[None, :, 2] - veh.radius
    return float(dist.min())
