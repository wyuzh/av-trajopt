"""Unconstrained minimisers used as inner solvers by the penalty and augmented
Lagrangian methods:

    gd     - gradient descent with backtracking (Armijo) line search
    newton - Newton's method with a positive-definite Hessian shift + backtracking
    adam   - Adam (first-order, fixed learning rate, no line search)
    lbfgs  - limited-memory BFGS (SciPy implementation, for reference)

All share one signature and call `callback(x)` once per iteration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

from .linalg import newton_direction


@dataclass
class InnerResult:
    x: np.ndarray
    iters: int
    converged: bool
    grad_norm: float
    backtracks: int = 0


def _armijo(fun, x, fx, gx, d, t0=1.0, c1=1e-4, beta=0.5, max_halvings=60, grad=None):
    """Backtracking line search: largest t = t0*beta^i with sufficient decrease.

    With `grad` given, a step whose predicted decrease |t * slope| is below the
    floating-point resolution of f is accepted if it reduces ||grad||_inf
    instead: there, f-values are pure rounding noise.  This matters for the
    penalty method at large rho, where |f| ~ 1 but the decrease is ~1e-14."""
    slope = float(gx @ d)
    t = t0
    for i in range(max_halvings):
        x_new = x + t * d
        f_new = fun(x_new)
        if np.isfinite(f_new) and f_new <= fx + c1 * t * slope:
            return t, x_new, f_new, i
        if grad is not None and abs(t * slope) <= 1e-12 * max(1.0, abs(fx)):
            if np.abs(grad(x_new)).max() < np.abs(gx).max():
                return t, x_new, f_new, i
        t *= beta
    return 0.0, x, fx, max_halvings


def minimize(fun, grad, x0, method="newton", hess=None, tol=1e-6, max_iter=500,
             callback=None, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
    x = np.array(x0, dtype=float)
    if method == "lbfgs":
        res = _scipy_minimize(fun, x, jac=grad, method="L-BFGS-B", callback=callback,
                              options=dict(maxiter=max_iter, gtol=tol, ftol=0.0, maxcor=20))
        gn = float(np.abs(grad(res.x)).max())
        return InnerResult(res.x, int(res.nit), gn <= tol, gn)

    fx, gx = fun(x), grad(x)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t_prev, backtracks = 1.0, 0
    for k in range(1, max_iter + 1):
        gn = float(np.abs(gx).max())
        if gn <= tol:
            return InnerResult(x, k - 1, True, gn, backtracks)

        if method in ("gd", "newton"):
            if method == "gd":
                d, t0 = -gx, min(1.0, 2.0 * t_prev)
            else:
                d, _ = newton_direction(hess(x), gx)
                if gx @ d >= 0:      # safeguard: fall back to steepest descent
                    d = -gx
                t0 = 1.0
            t, x, fx, nb = _armijo(fun, x, fx, gx, d, t0=t0,
                                   grad=grad if method == "newton" else None)
            backtracks += nb
            if t == 0.0:             # line search failed: no further progress possible
                return InnerResult(x, k, False, gn, backtracks)
            t_prev = t
        elif method == "adam":
            m = betas[0] * m + (1 - betas[0]) * gx
            v = betas[1] * v + (1 - betas[1]) * gx ** 2
            m_hat = m / (1 - betas[0] ** k)
            v_hat = v / (1 - betas[1] ** k)
            x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
            fx = fun(x)
        else:
            raise ValueError(f"unknown inner method {method!r}")

        gx = grad(x)
        if callback is not None:
            callback(x)
    gn = float(np.abs(gx).max())
    return InnerResult(x, max_iter, gn <= tol, gn, backtracks)
