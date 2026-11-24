"""M2 - Primal-dual interior-point (log-barrier) method.

Inequalities get slacks s > 0:  g(z) + s = 0.  For a barrier parameter mu_b > 0
we approximately solve the barrier problem

    min  f(z) - mu_b * sum_i log s_i    s.t.  h(z) = 0,  g(z) + s = 0,

whose KKT conditions are the *perturbed* KKT conditions of the original NLP

    grad f + J_h^T lam + J_g^T y = 0,   h = 0,   g + s = 0,   S y = mu_b e.

Each iteration takes one Newton step on this system.  Eliminating ds and dy
gives the symmetric reduced KKT system

    [ W + J_g^T Sigma J_g    J_h^T ] [dz  ]     [ -(grad f + J_h^T lam + J_g^T (mu_b/s + Sigma (g+s))) ]
    [ J_h                    0     ] [dlam]  =  [ -h                                                  ]

with W = Hessian of the Lagrangian and Sigma = S^{-1} Y.  For a nonconvex
problem the matrix must have inertia (n, m_eq, 0); otherwise W is shifted by
delta_w I (inertia correction).  Steps keep s, y > 0 by the fraction-to-the-
boundary rule.  The step length is chosen by a backtracking *filter* line
search (Waechter & Biegler, 2006): a trial point is accepted if it improves
either the constraint violation theta or the barrier objective phi enough,
which avoids the Maratos effect that makes l1-merit line searches reject good
Newton steps on curved constraints.  `globalization="merit"` switches back to
an l1 merit function for comparison.  If no acceptable step exists (typically
far from feasibility), a feasibility restoration phase takes Levenberg-
Marquardt steps on  1/2 ||[h; max(g, 0)]||^2  until the filter accepts the
point; if the violation cannot be reduced, the problem is declared locally
infeasible (a stationary point of the infeasibility measure).  mu_b is decreased (monotone Fiacco-McCormick strategy) once the
barrier problem is solved to accuracy kappa_eps * mu_b, so the iterates follow
the central path towards a KKT point.  Starting points need not be feasible:
the slacks absorb any initial violation of g(z) <= 0.

Linear algebra.  `kkt="dense"` factorises the reduced KKT matrix with a dense
LDL^T (inertia from Sylvester's law): O(n^3) per iteration.  `kkt="riccati"`
is the structure-exploiting variant (FATROP-style): sparse derivatives and a
Riccati recursion over the time stages (avtraj/riccati.py), O(N) per
iteration, with the same iterates up to round-off.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from ..linalg import inertia
from ..riccati import InertiaError, StageIndex, riccati_solve
from ..linesearch import Filter
from ..metrics import TOL_FEAS, TOL_KKT, History


def _frac_to_boundary(v, dv, tau):
    neg = dv < 0
    if not neg.any():
        return 1.0
    return float(min(1.0, (-tau * v[neg] / dv[neg]).min()))


def _solve_reduced_kkt(H, J, r1, r2, delta_last):
    """Solve the reduced KKT system with inertia correction (IPOPT-style)."""
    n, m = H.shape[0], J.shape[0]
    delta, delta_c, tries = 0.0, 0.0, 0
    while True:
        K = np.block([[H + delta * np.eye(n), J.T], [J, -delta_c * np.eye(m)]])
        pos, neg, zero = inertia(K)
        if pos == n and neg == m and zero == 0:
            break
        tries += 1
        if zero > 0 and delta_c == 0.0:
            delta_c = 1e-8                          # rank-deficient constraint Jacobian
            continue
        if delta == 0.0:
            delta = 1e-4 if delta_last == 0.0 else max(1e-20, delta_last / 3.0)
        else:
            delta *= 8.0 if delta_last == 0.0 else 3.0
        if delta > 1e40:
            raise np.linalg.LinAlgError("inertia correction failed")
    sol = np.linalg.solve(K, np.concatenate([r1, r2]))
    return sol[:n], sol[n:], delta, tries


def _next_delta(delta, delta_last):
    if delta == 0.0:
        return 1e-4 if delta_last == 0.0 else max(1e-20, delta_last / 3.0)
    return delta * (8.0 if delta_last == 0.0 else 3.0)


def _solve_reduced_kkt_riccati(H, Jh, r1, h, delta_last, idx):
    """Same system and inertia-correction rule as `_solve_reduced_kkt`, solved
    by the Riccati recursion.  Q_uu positive definite <=> inertia (n, m, 0)."""
    eye = sp.identity(H.shape[0], format="csr")
    delta, tries = 0.0, 0
    while True:
        try:
            dz, dlam = riccati_solve(H + delta * eye if delta else H, -r1, Jh, h, idx)
            return dz, dlam, delta, tries
        except InertiaError:
            tries += 1
            delta = _next_delta(delta, delta_last)
            if delta > 1e40:
                raise


def _restoration(nlp, z, s_floor, acceptable, max_steps=100, sparse=False):
    """Levenberg-Marquardt on the infeasibility  1/2 ||[h(z); max(g(z), 0)]||^2.
    Returns (z, s, ok); ok=False means a stationary point of the infeasibility."""
    lm = 1e-4

    def infeas(zz):
        hh, gg = nlp.h(zz), nlp.g(zz)
        gp = np.maximum(gg, 0.0)
        return hh @ hh + gp @ gp, hh, gg

    val, h, g = infeas(z)
    for _ in range(max_steps):
        if val <= 1e-14:                           # feasible: always a valid restart point
            return z, np.maximum(-g, s_floor), True
        act = g > 0
        if sparse:
            J = sp.vstack([nlp.jac_h_sp(z), nlp.jac_g_sp(z)[np.flatnonzero(act)]]).tocsc()
            eye = sp.identity(len(z), format="csc")
        else:
            J = np.vstack([nlp.jac_h(z), nlp.jac_g(z)[act]])
            eye = np.eye(len(z))
        r = np.concatenate([h, g[act]])
        grad = J.T @ r
        if np.abs(grad).max() <= 1e-9 * max(1.0, val):
            return z, np.maximum(-g, s_floor), False
        JtJ = J.T @ J
        while True:
            d = -(spsolve((JtJ + lm * eye).tocsc(), grad) if sparse
                  else np.linalg.solve(JtJ + lm * eye, grad))
            val_t, h_t, g_t = infeas(z + d)
            if val_t < val:
                lm = max(lm / 10, 1e-12)
                break
            lm *= 10
            if lm > 1e12:
                return z, np.maximum(-g, s_floor), False
        z, val, h, g = z + d, val_t, h_t, g_t
        s = np.maximum(-g, s_floor)
        if acceptable(z, s):
            return z, s, True
    return z, np.maximum(-g, s_floor), False


def solve(nlp, z0=None, tol=TOL_KKT, tol_feas=TOL_FEAS, max_iter=300, mu0=0.1,
          kappa_mu=0.2, theta_mu=1.5, kappa_eps=10.0, tau_min=0.99, eta=1e-4,
          s_min=1e-2, globalization="filter", mu_strategy="monotone", kkt="dense"):
    """mu_strategy: "monotone" (Fiacco-McCormick, superlinear decrease) or
    "fixed" (mu <- kappa_mu * mu after every iteration, no inner accuracy test)."""
    z = np.array(nlp.z0 if z0 is None else z0, dtype=float)
    me, mi = nlp.m_eq, nlp.m_in
    hist = History(nlp)
    sparse = kkt == "riccati"
    idx = StageIndex(nlp.meta["N"]) if sparse else None

    g = nlp.g(z)
    s = np.maximum(-g, s_min)                       # slacks absorb initial infeasibility
    mu_b = mu0
    y = mu_b / s
    lam = np.zeros(me)
    nu, delta_last, alpha, status, n_ls_fail, n_resto = 1.0, 0.0, 0.0, "max_iter", 0, 0
    info = {}

    def theta(zz, ss):                              # constraint violation
        return np.abs(nlp.h(zz)).sum() + np.abs(nlp.g(zz) + ss).sum()

    def phi(zz, ss):                                # barrier objective
        return nlp.f(zz) - mu_b * np.log(ss).sum()

    filt = Filter(theta(z, s))

    it = 0
    for it in range(max_iter):
        gf, h, g = nlp.grad_f(z), nlp.h(z), nlp.g(z)
        Jh, Jg = (nlp.jac_h_sp(z), nlp.jac_g_sp(z)) if sparse else (nlp.jac_h(z), nlp.jac_g(z))
        rd = gf + Jh.T @ lam + Jg.T @ y
        s_d = max(100.0, (np.abs(lam).sum() + np.abs(y).sum()) / max(me + mi, 1)) / 100.0

        def barrier_error(m):
            return max(np.abs(rd).max() / s_d,
                       np.abs(h).max() if me else 0.0,
                       np.abs(g + s).max() if mi else 0.0,
                       np.abs(s * y - m).max() / s_d if mi else 0.0)

        hist.log(z, lam, y, mu_b=mu_b, alpha=alpha, delta_w=delta_last, **info)
        if barrier_error(0.0) <= tol and max(np.abs(h).max() if me else 0.0,
                                             max(g.max(), 0.0) if mi else 0.0) <= tol_feas:
            status = "converged"
            break
        mu_old = mu_b
        if mu_strategy == "fixed":
            if it > 0:
                mu_b = max(tol / 10, kappa_mu * mu_b)
        else:
            while mu_b > tol / 10 and barrier_error(mu_b) <= kappa_eps * mu_b:
                mu_b = max(tol / 10, min(kappa_mu * mu_b, mu_b ** theta_mu))
        if mu_b != mu_old:
            filt.reset()                            # new barrier problem -> reset filter

        # ---- Newton step on the perturbed KKT system + line search ----
        Sig = y / s
        if sparse:
            Hred = (nlp.hess_lag_sp(z, lam, y) + Jg.T @ sp.diags(Sig) @ Jg).tocsr()
        else:
            Hred = nlp.hess_lag(z, lam, y) + Jg.T @ (Sig[:, None] * Jg)
        r1 = -(gf + Jh.T @ lam + Jg.T @ (mu_b / s + Sig * (g + s)))
        th, ph = theta(z, s), phi(z, s)
        try:
            if sparse:
                dz, dlam, delta_last, _ = _solve_reduced_kkt_riccati(Hred, Jh, r1, h, delta_last,
                                                                     idx)
            else:
                dz, dlam, delta_last, _ = _solve_reduced_kkt(Hred, Jh, r1, -h, delta_last)
        except np.linalg.LinAlgError:
            dz = None                                 # singular KKT system -> restoration
        accepted, a_p, a_d = False, 0.0, 0.0
        if dz is not None:
            ds = -(g + s) - Jg @ dz
            dy = mu_b / s - y - Sig * ds
            tau = max(tau_min, 1.0 - mu_b)
            a_p = _frac_to_boundary(s, ds, tau)
            a_d = _frac_to_boundary(y, dy, tau)
            m = gf @ dz - mu_b * np.sum(ds / s)      # directional derivative of phi
            alpha = a_p
            if globalization == "merit":
                if th > 0:
                    curv = max(dz @ (Hred @ dz) + delta_last * dz @ dz, 0.0)
                    nu = max(nu, (m + 0.5 * curv) / (0.9 * th) + 1e-6)
                D, phi0 = m - nu * th, ph + nu * th
                for _ in range(40):
                    z_t, s_t = z + alpha * dz, s + alpha * ds
                    if phi(z_t, s_t) + nu * theta(z_t, s_t) <= phi0 + eta * alpha * min(D, 0.0):
                        accepted = True
                        break
                    alpha *= 0.5
            else:
                for _ in range(40):
                    z_t, s_t = z + alpha * dz, s + alpha * ds
                    if filt.accept(theta(z_t, s_t), phi(z_t, s_t), th, ph, m, alpha):
                        accepted = True
                        break
                    alpha *= 0.5
        info = dict(a_p=a_p, a_d=a_d)
        if not accepted and (globalization == "filter" or dz is None):
            # ---- feasibility restoration phase ----
            n_ls_fail += 1
            n_resto += 1
            filt.add(th, ph)

            def acceptable(zz, ss, th_r=th):
                th_t = theta(zz, ss)
                return th_t <= 0.9 * th_r and not filt.dominated(th_t, phi(zz, ss))

            z, s, ok = _restoration(nlp, z, mu_b, acceptable, sparse=sparse)
            if not ok:
                status = "locally_infeasible"
                break
            y = mu_b / s                              # re-initialise inequality multipliers
            filt.reset()
            alpha, info["restoration"] = 0.0, True
            continue
        n_ls_fail += not accepted
        z, s = z_t, s_t
        lam = lam + alpha * dlam
        y = y + a_d * dy
        y = np.clip(y, mu_b / (1e10 * s), 1e10 * mu_b / s)   # keep y close to mu_b / s

    return hist.result("ipm-riccati" if sparse else "ipm", z, lam, y, it, status, mu_final=mu_b, ls_failures=n_ls_fail,
                       restorations=n_resto,
                       globalization=globalization, mu_strategy=mu_strategy)
