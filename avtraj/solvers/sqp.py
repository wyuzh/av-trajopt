"""M4 - Sequential Quadratic Programming (line-search SQP with a filter).

At z_k we apply Newton's method to the KKT conditions, which is equivalent to
solving the quadratic subproblem

    min_d   grad f^T d + 1/2 d^T B d
    s.t.    h + J_h d = 0,     g + J_g d <= 0,

where B is (a positive-definite modification of) the Hessian of the
Lagrangian ("exact") or a damped BFGS approximation of it ("bfgs").  The QP
multipliers are the new Lagrange multiplier estimates.  The QP is solved with
the interior-point QP solver Clarabel.  If the linearised constraints are
inconsistent (possible far from feasibility) an elastic l1 QP is solved
instead.  The step length is chosen by a backtracking filter line search on
(theta, f) with theta = ||h||_1 + ||max(g, 0)||_1 (see linesearch.py);
`globalization="merit"` uses the exact l1 penalty

    phi(z) = f(z) + nu * theta(z),   nu >= ||(lam, mu)||_inf

instead, which suffers from the Maratos effect (kept for the ablation study).
"""
from __future__ import annotations

import clarabel
import numpy as np
import scipy.sparse as sp

from ..linalg import make_positive_definite
from ..linesearch import Filter
from ..metrics import TOL_FEAS, TOL_KKT, History, kkt_residual

_OK = ("Solved", "AlmostSolved")


def _clarabel(P, q, A, b, me, mi):
    """min 1/2 x'Px + q'x  s.t.  A[:me] x = b[:me],  A[me:] x <= b[me:]."""
    cones = []
    if me:
        cones.append(clarabel.ZeroConeT(me))
    if mi:
        cones.append(clarabel.NonnegativeConeT(mi))
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-10
    sol = clarabel.DefaultSolver(sp.triu(sp.csc_matrix(P)).tocsc(), q, sp.csc_matrix(A), b,
                                 cones, st).solve()
    return str(sol.status), np.array(sol.x), np.array(sol.z)


def convexify(W, J_act, rhos=(0.0, 1.0, 10.0, 1e2, 1e3, 1e4)):
    """Make the QP Hessian positive definite without destroying Newton's step.

    Only the curvature on the null space of the active constraint Jacobian
    matters for the KKT step, so we first add rho * J_act^T J_act (augmented-
    Lagrangian convexification: for active constraints J_act d = -c is fixed,
    so the QP solution does not change).  By Finsler's lemma this is positive
    definite for large rho whenever the reduced Hessian is.  Only if that fails
    do we fall back to a diagonal shift."""
    JtJ = J_act.T @ J_act
    for rho in rhos:
        try:
            np.linalg.cholesky(W + rho * JtJ)
            return W + rho * JtJ
        except np.linalg.LinAlgError:
            pass
    return make_positive_definite(W + rhos[-1] * JtJ)


def qp_subproblem(B, gf, h, Jh, g, Jg, nu_elastic=1e4):
    """Return (d, lam, mu, elastic_flag) or None if both QPs fail."""
    n, me, mi = len(gf), len(h), len(g)
    A = np.vstack([Jh, Jg])
    status, x, dual = _clarabel(B, gf, A, np.concatenate([-h, -g]), me, mi)
    if status in _OK:
        return x, dual[:me], dual[me:], False

    # elastic mode: h + J_h d = p - q,  g + J_g d <= t,  p, q, t >= 0
    nv = n + 2 * me + mi
    P = sp.block_diag([sp.csc_matrix(B), sp.csc_matrix((nv - n, nv - n))])
    q = np.concatenate([gf, nu_elastic * np.ones(nv - n)])
    I_me, I_mi = sp.eye(me), sp.eye(mi)
    A_eq = sp.hstack([sp.csc_matrix(Jh), -I_me, I_me, sp.csc_matrix((me, mi))])
    A_in = sp.vstack([
        sp.hstack([sp.csc_matrix(Jg), sp.csc_matrix((mi, 2 * me)), -I_mi]),
        sp.hstack([sp.csc_matrix((nv - n, n)), -sp.eye(nv - n)]),
    ])
    b = np.concatenate([-h, -g, np.zeros(nv - n)])
    status, x, dual = _clarabel(P, q, sp.vstack([A_eq, A_in]), b, me, mi + nv - n)
    if status in _OK:
        return x[:n], dual[:me], dual[me:me + mi], True
    return None


def solve(nlp, z0=None, hessian="exact", tol=TOL_KKT, tol_feas=TOL_FEAS, max_iter=200,
          eta=1e-4, globalization="filter"):
    z = np.array(nlp.z0 if z0 is None else z0, dtype=float)
    me, mi, n = nlp.m_eq, nlp.m_in, nlp.n
    lam, mu = np.zeros(me), np.zeros(mi)
    B = np.eye(n)
    nu, alpha, status, n_elastic = 1.0, 0.0, "max_iter", 0
    hist = History(nlp)

    def theta(zz):
        return np.abs(nlp.h(zz)).sum() + np.maximum(nlp.g(zz), 0).sum()

    filt = Filter(theta(z))
    n_ls_fail = 0

    def grad_lag(zz, lam_, mu_):
        return nlp.grad_f(zz) + nlp.jac_h(zz).T @ lam_ + nlp.jac_g(zz).T @ mu_

    for it in range(max_iter):
        gf, h, g = nlp.grad_f(z), nlp.h(z), nlp.g(z)
        Jh, Jg = nlp.jac_h(z), nlp.jac_g(z)
        hist.log(z, lam, mu, alpha=alpha)
        with nlp.no_count():
            r = kkt_residual(nlp, z, lam, mu)
        if r["kkt"] <= tol and r["feas"] <= tol_feas:
            status = "converged"
            break

        if hessian == "exact":
            B = convexify(nlp.hess_lag(z, lam, mu), np.vstack([Jh, Jg[(mu > 1e-8) | (g > -1e-6)]]))
        elif hessian == "exact-shift":             # naive: shift the whole Hessian (ablation)
            B = make_positive_definite(nlp.hess_lag(z, lam, mu))
        out = qp_subproblem(B, gf, h, Jh, g, Jg)
        if out is None:
            status = "qp_failed"
            break
        d, lam_qp, mu_qp, elastic = out
        n_elastic += elastic

        nu = max(nu, 1.1 * max(np.abs(lam_qp).max(initial=0.0), np.abs(mu_qp).max(initial=0.0)))
        th, f0, m = np.abs(h).sum() + np.maximum(g, 0).sum(), nlp.f(z), gf @ d
        alpha, accepted = 1.0, False
        for _ in range(40):
            zt = z + alpha * d
            if globalization == "merit":
                ok = nlp.f(zt) + nu * theta(zt) <= f0 + nu * th + eta * alpha * min(m - nu * th, 0.0)
            else:
                ok = filt.accept(theta(zt), nlp.f(zt), th, f0, m, alpha)
            if ok:
                accepted = True
                break
            alpha *= 0.5
        n_ls_fail += not accepted
        z_new = z + alpha * d
        lam_new = lam + alpha * (lam_qp - lam)
        mu_new = mu + alpha * (mu_qp - mu)

        if hessian == "bfgs":                      # damped BFGS (Powell) on the Lagrangian
            s_ = z_new - z
            y_ = grad_lag(z_new, lam_new, mu_new) - grad_lag(z, lam_new, mu_new)
            Bs = B @ s_
            sBs = s_ @ Bs
            if sBs > 1e-16:
                damp = 1.0 if s_ @ y_ >= 0.2 * sBs else 0.8 * sBs / (sBs - s_ @ y_)
                r_ = damp * y_ + (1 - damp) * Bs
                B = B - np.outer(Bs, Bs) / sBs + np.outer(r_, r_) / (s_ @ r_)

        if np.abs(z_new - z).max() < 1e-14 and alpha < 1e-10:
            status = "line_search_failed"
            z, lam, mu = z_new, lam_new, mu_new
            break
        z, lam, mu = z_new, lam_new, mu_new

    return hist.result(f"sqp-{hessian}", z, lam, mu, it, status, n_elastic=n_elastic,
                       ls_failures=n_ls_fail, globalization=globalization)
