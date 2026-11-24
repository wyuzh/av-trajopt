"""M3 - Augmented Lagrangian method (method of multipliers, PHR form).

    L_A(z; lam, mu, rho) = f + lam^T h + rho/2 ||h||^2
                         + 1/(2 rho) * sum_i ( max(0, mu_i + rho g_i)^2 - mu_i^2 )

Each outer iteration approximately minimises L_A over z (unconstrained inner
solver), then performs a dual ascent step on the multipliers

    lam <- lam + rho h(z),     mu <- max(0, mu + rho g(z)),

which is a gradient step on the (augmented) dual function.  Because the
multipliers absorb the constraint forces, feasibility is reached with a
*bounded* rho, avoiding the ill-conditioning of the pure penalty method.
rho is increased only when the violation fails to drop by a factor `eta`.
"""
from __future__ import annotations

import numpy as np

from ..metrics import TOL_FEAS, TOL_KKT, History, kkt_residual
from ..unconstrained import minimize


def augmented_lagrangian(nlp, lam, mu, rho):
    def LA(z):
        h, g = nlp.h(z), nlp.g(z)
        mp = np.maximum(mu + rho * g, 0.0)
        return nlp.f(z) + lam @ h + 0.5 * rho * h @ h + (mp @ mp - mu @ mu) / (2 * rho)

    def dLA(z):
        h, g = nlp.h(z), nlp.g(z)
        mp = np.maximum(mu + rho * g, 0.0)
        return nlp.grad_f(z) + nlp.jac_h(z).T @ (lam + rho * h) + nlp.jac_g(z).T @ mp

    def HLA(z):
        h, g = nlp.h(z), nlp.g(z)
        mp = mu + rho * g
        act = mp > 0
        Jh, Ja = nlp.jac_h(z), nlp.jac_g(z)[act]
        H = nlp.hess_lag(z, lam + rho * h, np.maximum(mp, 0.0))
        return H + rho * (Jh.T @ Jh + Ja.T @ Ja)

    def mult(z):   # first-order multiplier estimates at z
        return lam + rho * nlp.h(z), np.maximum(mu + rho * nlp.g(z), 0.0)

    return LA, dLA, HLA, mult


def solve(nlp, z0=None, inner="newton", rho0=10.0, rho_factor=10.0, rho_max=1e8,
          eta=0.25, tol=TOL_FEAS, tol_kkt=TOL_KKT, max_outer=50, max_inner=200,
          inner_tol0=1e-2, lr=1e-2):
    z = np.array(nlp.z0 if z0 is None else z0, dtype=float)
    lam, mu = np.zeros(nlp.m_eq), np.zeros(nlp.m_in)
    rho, omega = rho0, inner_tol0
    hist = History(nlp)
    viol_prev, total, status = np.inf, 0, "max_outer"
    rhos = []

    for outer in range(max_outer):
        LA, dLA, HLA, mult = augmented_lagrangian(nlp, lam, mu, rho)
        if outer == 0:
            hist.log(z, mult_fn=mult, outer=outer, rho=rho)
        res = minimize(LA, dLA, z, method=inner, hess=HLA, tol=omega, max_iter=max_inner, lr=lr,
                       callback=lambda x: hist.log(x, mult_fn=mult, outer=outer, rho=rho))
        z = res.x
        total += res.iters
        rhos.append(rho)

        h, g = nlp.h(z), nlp.g(z)
        lam = lam + rho * h                       # dual ascent on the multipliers
        mu = np.maximum(mu + rho * g, 0.0)
        with nlp.no_count():
            r = kkt_residual(nlp, z, lam, mu)
        if r["feas"] <= tol and r["kkt"] <= tol_kkt:
            status = "converged"
            break
        if r["feas"] > eta * viol_prev:           # insufficient progress -> stiffen
            rho = min(rho * rho_factor, rho_max)
        viol_prev = r["feas"]
        omega = max(0.1 * tol_kkt, 0.1 * omega)

    return hist.result(f"alm-{inner}", z, lam, mu, total, status, rho=rho, rhos=rhos)
