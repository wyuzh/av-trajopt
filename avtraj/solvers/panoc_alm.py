"""PANOC-ALM: augmented Lagrangian method with the PANOC inner solver
(Stella et al. 2017; OpEn, Sopasakis et al. 2020; alpaqa, Pas et al. 2021).

Simple bounds  l <= z <= u  are detected among the inequality rows and kept as
a set C onto which we can project.  The other constraints (dynamics, steering
rate, obstacles) are handled by an augmented Lagrangian outer loop, so each
inner problem is

    min_{z in C}  psi(z) = f + lam^T h + rho/2 ||h||^2
                           + 1/(2 rho) sum_i ( max(0, mu_i + rho g_i)^2 - mu_i^2 ).

PANOC solves it without Hessians or linear systems.  With the projected-
gradient map  T(z) = Pi_C(z - gamma grad psi(z))  and residual r = z - T(z),
the forward-backward envelope

    phi_gamma(z) = psi(z) - grad psi(z)^T r + ||r||^2 / (2 gamma)

is a continuous, real-valued merit function whose minimisers are the
stationary points.  Each iteration blends the safe projected-gradient step
with a fast L-BFGS step d (built from residual differences):

    z+ = z - (1 - tau) r + tau d,   tau in {1, 1/2, ...},

accepting the first tau that decreases phi_gamma sufficiently (tau = 0 always
does).  gamma = 0.95 / L with L estimated and increased on the fly.  The
multiplier update is the usual dual ascent  lam += rho h,  mu = max(0, mu + rho g).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..metrics import TOL_FEAS, TOL_KKT, History, kkt_residual


# ---------------------------------------------------------------------------
# problem splitting: simple bounds vs general constraints
# ---------------------------------------------------------------------------
def split_bounds(nlp, z0, seed=0):
    """Find rows of g that are simple bounds  c * z_j + b <= 0.
    Returns (lb, ub, box_rows, other_rows, row_var, row_coef)."""
    rng = np.random.default_rng(seed)
    J0 = nlp.jac_g_sp(z0).tocsr()
    J1 = nlp.jac_g_sp(z0 + 0.1 * rng.standard_normal(z0.size)).tocsr()
    g0 = nlp.g(z0)
    lb, ub = np.full(nlp.n, -np.inf), np.full(nlp.n, np.inf)
    box, other, var, coef = [], [], [], []
    for i in range(nlp.m_in):
        r0, r1 = J0.getrow(i), J1.getrow(i)
        if r0.nnz == 1 and r1.nnz == 1 and r0.indices[0] == r1.indices[0] \
                and np.isclose(r0.data[0], r1.data[0]) and r0.data[0] != 0:
            j, c = int(r0.indices[0]), float(r0.data[0])
            b = g0[i] - c * z0[j]
            bound = -b / c
            if c > 0:
                ub[j] = min(ub[j], bound)
            else:
                lb[j] = max(lb[j], bound)
            box.append(i)
            var.append(j)
            coef.append(c)
        else:
            other.append(i)
    return lb, ub, np.array(box, int), np.array(other, int), np.array(var, int), np.array(coef)


# ---------------------------------------------------------------------------
# PANOC for  min psi(z)  s.t.  z in [lb, ub]
# ---------------------------------------------------------------------------
def panoc(psi, lb, ub, z0, tol, max_iter, callback=None, memory=10, beta=0.05, L0=None):
    """psi(z) -> (value, gradient).  Returns (z_bar, iters, converged, gamma, L)."""
    proj = lambda x: np.clip(x, lb, ub)  # noqa: E731
    z = proj(z0)
    f, g = psi(z)
    if L0 is None:                                   # Lipschitz estimate by finite difference
        dz = 1e-6 * np.maximum(np.abs(z), 1e-3)
        L0 = max(np.linalg.norm(psi(z + dz)[1] - g) / np.linalg.norm(dz), 1e-6)
    L = L0
    gamma = 0.95 / L
    S, Y = deque(maxlen=memory), deque(maxlen=memory)

    def fbe_point(zz, ff, gg, gam):
        zb = proj(zz - gam * gg)
        rr = zz - zb
        return zb, rr, ff - gg @ rr + (rr @ rr) / (2 * gam)

    zbar, r, phi = fbe_point(z, f, g, gamma)
    it = 0
    for it in range(1, max_iter + 1):
        # Lipschitz backtracking: the quadratic upper bound must hold at z_bar
        fbar, gbar = psi(zbar)
        while fbar > f - g @ r + 0.5 * L * (r @ r) + 1e-12 * abs(f):
            L *= 2.0
            gamma *= 0.5
            S.clear()
            Y.clear()
            zbar, r, phi = fbe_point(z, f, g, gamma)
            fbar, gbar = psi(zbar)
        if np.abs(r).max() / gamma <= tol:
            return zbar, it - 1, True, gamma, L

        # L-BFGS direction on the fixed-point residual
        q = -r.copy()
        alphas = []
        for s_, y_ in reversed(list(zip(S, Y))):
            a = (s_ @ q) / (y_ @ s_)
            alphas.append(a)
            q -= a * y_
        if S:
            q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        for (s_, y_), a in zip(zip(S, Y), reversed(alphas)):
            b = (y_ @ q) / (y_ @ s_)
            q += (a - b) * s_
        d = q

        sigma = beta * (1 - gamma * L) / (2 * gamma)
        tau = 1.0
        while True:
            z_new = z - (1 - tau) * r + tau * d if tau > 0 else zbar
            f_new, g_new = psi(z_new)
            zbar_new, r_new, phi_new = fbe_point(z_new, f_new, g_new, gamma)
            if phi_new <= phi - sigma * (r @ r) or tau == 0:
                break
            tau = tau / 2 if tau > 1e-6 else 0.0
        s_, y_ = z_new - z, r_new - r
        if s_ @ y_ > 1e-12 * (s_ @ s_):
            S.append(s_)
            Y.append(y_)
        z, f, g, zbar, r, phi = z_new, f_new, g_new, zbar_new, r_new, phi_new
        if callback is not None:
            callback(zbar)
    return zbar, it, False, gamma, L


# ---------------------------------------------------------------------------
# augmented Lagrangian outer loop
# ---------------------------------------------------------------------------
def solve(nlp, z0=None, rho0=10.0, rho_factor=5.0, eta=0.25, tol=TOL_KKT, tol_feas=TOL_FEAS,
          inner_tol0=1e-2, inner_shrink=0.1, max_outer=60, max_inner=3000, max_total=60000,
          memory=10):
    z = np.array(nlp.z0 if z0 is None else z0, dtype=float)
    hist = History(nlp)
    with nlp.no_count():
        lb, ub, box, other, var, coef = split_bounds(nlp, z)
    up_row = box[coef > 0]
    lo_row = box[coef < 0]
    up_var, lo_var = var[coef > 0], var[coef < 0]
    up_c, lo_c = coef[coef > 0], coef[coef < 0]

    lam = np.zeros(nlp.m_eq)
    mu = np.zeros(len(other))
    rho, eps = rho0, inner_tol0
    total, status, viol_prev, L_est = 0, "max_outer", np.inf, None
    outers = []

    def make_psi(lam_, mu_, rho_):
        def psi(zz):
            h = nlp.h(zz)
            go = nlp.g(zz)[other]
            mp = np.maximum(mu_ + rho_ * go, 0.0)
            val = nlp.f(zz) + lam_ @ h + 0.5 * rho_ * h @ h + (mp @ mp - mu_ @ mu_) / (2 * rho_)
            grad = (nlp.grad_f(zz) + nlp.jac_h_sp(zz).T @ (lam_ + rho_ * h)
                    + nlp.jac_g_sp(zz)[other].T @ mp)
            return val, grad
        return psi

    def multipliers(zz, lam_, mu_, rho_):
        """Full multiplier vector for the KKT test: ALM rows + bound rows (the
        bound multipliers come from the normal cone of the projection)."""
        h = nlp.h(zz)
        g = nlp.g(zz)
        lam_full = lam_ + rho_ * h
        mu_full = np.zeros(nlp.m_in)
        mu_full[other] = np.maximum(mu_ + rho_ * g[other], 0.0)
        _, grad = make_psi(lam_, mu_, rho_)(zz)
        w = -grad                                    # psi stationarity: grad + w = 0, w in N_C
        # bound row  c z_j + b <= 0  contributes  mu c e_j  to grad L, so mu = w_j / c;
        # assign each variable's multiplier to its tightest (most active) bound row
        for rows, vars_, cs in ((up_row, up_var, up_c), (lo_row, lo_var, lo_c)):
            seen = set()
            for t in np.argsort(-(g[rows])):
                j = vars_[t]
                if j in seen:
                    continue
                seen.add(j)
                val = w[j] / cs[t]
                if val > 0:
                    mu_full[rows[t]] = val
        return lam_full, mu_full

    hist.log(z, mult_fn=lambda zz: multipliers(zz, lam, mu, rho), outer=0, rho=rho)
    for outer in range(max_outer):
        psi = make_psi(lam, mu, rho)
        lam_o, mu_o, rho_o = lam.copy(), mu.copy(), rho
        z, iters, conv, gamma, L_est = panoc(
            psi, lb, ub, z, eps, min(max_inner, max_total - total), memory=memory, L0=L_est,
            callback=lambda x: hist.log(x, mult_fn=lambda zz: multipliers(zz, lam_o, mu_o, rho_o),
                                        outer=outer, rho=rho_o))
        total += iters
        outers.append(dict(rho=rho, inner=iters, converged=bool(conv)))
        h, g = nlp.h(z), nlp.g(z)
        lam_full, mu_full = multipliers(z, lam, mu, rho)
        lam = lam + rho * h
        mu = np.maximum(mu + rho * g[other], 0.0)
        with nlp.no_count():
            r = kkt_residual(nlp, z, lam_full, mu_full)
        if r["feas"] <= tol_feas and r["kkt"] <= tol:
            status = "converged"
            break
        if total >= max_total:
            status = "max_total_iter"
            break
        viol = max(np.abs(h).max(), max(g[other].max(), 0.0))
        if viol > eta * viol_prev:
            rho *= rho_factor
        viol_prev = viol
        eps = max(0.1 * tol, inner_shrink * eps)
        L_est = None                                 # psi changed: re-estimate L

    with nlp.no_count():                             # multipliers of the last subproblem
        lam_full, mu_full = multipliers(z, lam_o, mu_o, rho_o)
    return hist.result("panoc-alm", z, lam_full, mu_full, total, status, rho=rho,
                       outer=outers, n_bounds=int(len(box)), n_alm_rows=int(len(other)))
