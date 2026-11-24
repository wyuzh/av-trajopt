"""Kinematic bicycle model of the ego vehicle.

State   x = [px, py, theta, v]   (position [m], heading [rad], speed [m/s])
Control u = [a, delta]           (acceleration [m/s^2], steering angle [rad])
"""
from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

NX, NU = 4, 2


@dataclass(frozen=True)
class Vehicle:
    wheelbase: float = 2.7
    radius: float = 1.0          # collision circle around the vehicle centre [m]
    v_min: float = 0.0
    v_max: float = 15.0
    a_min: float = -6.0
    a_max: float = 3.0
    delta_max: float = 0.5       # steering angle limit [rad]
    ddelta_max: float = 1.0      # steering rate limit [rad/s]


def bicycle_ode(x, u, wheelbase):
    """Continuous-time kinematic bicycle dynamics  dx/dt = f_c(x, u)."""
    return ca.vertcat(
        x[3] * ca.cos(x[2]),
        x[3] * ca.sin(x[2]),
        x[3] / wheelbase * ca.tan(u[1]),
        u[0],
    )


def rk4_step(x, u, dt, wheelbase):
    """One explicit Runge-Kutta-4 step (zero-order hold on u)."""
    k1 = bicycle_ode(x, u, wheelbase)
    k2 = bicycle_ode(x + dt / 2 * k1, u, wheelbase)
    k3 = bicycle_ode(x + dt / 2 * k2, u, wheelbase)
    k4 = bicycle_ode(x + dt * k3, u, wheelbase)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def discrete_dynamics(veh: Vehicle, dt: float) -> ca.Function:
    """CasADi function F(x, u) -> x_next."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    return ca.Function("F", [x, u], [rk4_step(x, u, dt, veh.wheelbase)])
