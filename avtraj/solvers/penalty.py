"""M1 - Quadratic penalty method.

    P(z; rho) = f(z) + rho/2 * ( ||h(z)||^2 + ||max(g(z), 0)||^2 )

A sequence of *unconstrained* problems min_z P(z; rho_k) is solved with an
inner method (gradient descent, Newton, Adam or L-BFGS), warm-starting each
from the previous solution, while rho_k -> infinity.

Theory: minimisers of P(.; rho) approach a KKT point as rho -> inf and
    lam ~ rho * h(z),   mu ~ rho * max(g(z), 0)
estimate the Lagrange multipliers; the constraint violation decays like O(1/rho).
The price is conditioning: the Hessian of P contains rho * J^T J, so its
condition number grows like O(rho) and first-order inner methods slow down.
"""
from __future__ import annotations

import numpy as np

from ..metrics import TOL_FEAS, TOL_KKT, History, kkt_residual
from ..unconstrained import minimize


def penalty_functions(nlp, rho):
    """Return P, grad P and Hessian of P for a fixed rho."""

    def P(z):
        h, gp = nlp.h(z), np.maximum(nlp.g(z), 0.0)
        return nlp.f(z) + 0.5 * rho * (h @ h + gp @ gp)

    def dP(z):
        h, gp = nlp.h(z), np.maximum(nlp.g(z), 0.0)
        return nlp.grad_f(z) + rho * (nlp.jac_h(z).T @ h + nlp.jac_g(z).T @ gp)

    def HP(z):
        h, g = nlp.h(z), nlp.g(z)
        act = g > 0
        Jh, Ja = nlp.jac_h(z), nlp.jac_g(z)[act]
        H = nlp.hess_lag(z, rho * h, rho * np.maximum(g, 0.0))
        return H + rho * (Jh.T @ Jh + Ja.T @ Ja)

    return P, dP, HP


def multiplier_estimate(nlp, rho):
    return lambda z: (rho * nlp.h(z), rho * np.maximum(nlp.g(z), 0.0))


def solve(nlp, z0=None, inner="newton", rho0=1.0, rho_factor=10.0, rho_max=1e9,
          tol=TOL_FEAS, tol_kkt=TOL_KKT, max_outer=12, max_inner=500,
          inner_tol0=1e-2, lr=1e-2, fixed_rho=None):
    """If `fixed_rho` is given, solve a single penalty subproblem (used in the
    conditioning experiment); otherwise run the full continuation in rho."""
    z = np.array(nlp.z0 if z0 is None else z0, dtype=float)
    hist = History(nlp)
    rho = fixed_rho if fixed_rho is not None else rho0
    omega = inner_tol0 if fixed_rho is None else tol_kkt
    total, status = 0, "max_outer"
    inner_iters = []

    for outer in range(1 if fixed_rho is not None else max_outer):
        P, dP, HP = penalty_functions(nlp, rho)
        mult = multiplier_estimate(nlp, rho)
        if outer == 0:
            hist.log(z, mult_fn=mult, outer=outer, rho=rho)
        res = minimize(P, dP, z, method=inner, hess=HP, tol=omega, max_iter=max_inner, lr=lr,
                       callback=lambda x: hist.log(x, mult_fn=mult, outer=outer, rho=rho))
        z = res.x
        total += res.iters
        inner_iters.append(res.iters)
        with nlp.no_count():
            r = kkt_residual(nlp, z, *mult(z))
        if r["feas"] <= tol and r["kkt"] <= tol_kkt:
            status = "converged"
            break
        if fixed_rho is not None:
            status = "fixed_rho"
            break
        if rho >= rho_max:
            status = "rho_max"
            break
        rho = min(rho * rho_factor, rho_max)
        omega = max(tol_kkt, 0.1 * omega)

    lam, mu = multiplier_estimate(nlp, rho)(z)
    return hist.result(f"penalty-{inner}", z, lam, mu, total, status, rho=rho,
                       inner_iters=inner_iters, inner_converged=bool(res.converged))
