"""Convex corridor QP and Sequential Convex Programming (SCP).

Corridor QP.  Around a reference trajectory z_ref the NLP is replaced by a
convex QP:
  * objective: the exact (convex quadratic) objective;
  * dynamics: linearised,  h(z_ref) + J_h(z_ref)(z - z_ref) = 0;
  * linear inequalities (bounds, rate limits, road edges): kept exactly;
  * each obstacle disc ||p_k - o_j|| >= R_j is replaced by a *separating
    hyperplane*: the supporting half-space of the disc at the point closest to
    the reference position p_ref_k,
          n^T (p_k - o_j) >= R_j,    n = (p_ref_k - o_j) / ||p_ref_k - o_j||.
    By Cauchy-Schwarz every point of this half-space is collision-free, so the
    convex QP is an inner (conservative) approximation of the free space.
Being convex, the QP is solved to global optimality with a duality-gap
certificate, and its multipliers show which obstacles are active.

SCP.  Repeatedly solve the corridor QP around the current iterate inside a
trust region ||z - z_k||_inf <= Delta.  As in TrajOpt (Schulman et al., 2014)
the linearised dynamics and obstacle half-spaces are made *soft* with an exact
l1 penalty nu * (slack), so every subproblem is feasible even for a small
trust region; a step is accepted if the true l1 merit decreases by a fraction
of the decrease predicted by the convex model (trust-region ratio test), and
nu is increased while the iterates remain infeasible.  At convergence the
half-spaces are tangent to the discs and the iterate is a KKT point.
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from ..metrics import TOL_FEAS, TOL_KKT, History, kkt_residual
from ..vehicle import NX


def _obstacle_halfspaces(nlp, z_ref):
    """Rows a_r^T z <= b_r replacing the obstacle constraints of `nlp`."""
    sc, veh, N = nlp.meta["scenario"], nlp.meta["vehicle"], nlp.meta["N"]
    rows = nlp.meta["obs_rows"]
    data, ii, jj = [], [], []
    b = np.zeros(len(rows))
    for r, (_, k, j) in enumerate(rows):
        o = sc.obstacles[j, :2]
        R = sc.obstacles[j, 2] + veh.radius
        p = z_ref[NX * k: NX * k + 2]
        d = p - o
        nd = np.linalg.norm(d)
        nvec = d / nd if nd > 1e-9 else np.array([0.0, 1.0])
        # R - n^T (p - o) <= 0   <=>   -n^T p <= -R - n^T o
        ii += [r, r]
        jj += [NX * k, NX * k + 1]
        data += [-nvec[0], -nvec[1]]
        b[r] = -R - nvec @ o
    A = sp.csr_matrix((data, (ii, jj)), shape=(len(rows), nlp.n))
    return A, b


def corridor_qp(nlp, z_ref, trust=None, penalty=None, solver=cp.CLARABEL):
    """Solve the convexified problem around z_ref.  With `penalty` = nu the
    dynamics and obstacle constraints are softened by l1-penalised slacks.
    Returns a dict (z=None if the QP is infeasible)."""
    n, n_lin = nlp.n, nlp.meta["n_lin"]
    f0, gf = nlp.f(z_ref), nlp.grad_f(z_ref)
    Q = nlp.hess_lag(z_ref, np.zeros(nlp.m_eq), np.zeros(nlp.m_in))   # = Hessian of f
    h0, Jh = nlp.h(z_ref), sp.csr_matrix(nlp.jac_h(z_ref))
    g0, Jg = nlp.g(z_ref), nlp.jac_g(z_ref)
    G_lin, g_lin = sp.csr_matrix(Jg[:n_lin]), g0[:n_lin]
    A_obs, b_obs = _obstacle_halfspaces(nlp, z_ref)

    z = cp.Variable(n)
    dz = z - z_ref
    obj = f0 + gf @ dz + 0.5 * cp.quad_form(dz, cp.psd_wrap(Q))
    if penalty is None:
        c_dyn = [Jh @ dz + h0 == 0]
        c_obs = A_obs @ z <= b_obs
    else:
        e = cp.Variable(nlp.m_eq)                   # |h_lin| <= e
        t = cp.Variable(A_obs.shape[0], nonneg=True)
        obj = obj + penalty * (cp.sum(e) + cp.sum(t))
        c_dyn = [Jh @ dz + h0 <= e, -(Jh @ dz + h0) <= e]
        c_obs = A_obs @ z - t <= b_obs
    c_lin = G_lin @ dz + g_lin <= 0
    cons = c_dyn + [c_lin] + ([c_obs] if A_obs.shape[0] else [])
    if trust is not None:
        cons.append(cp.abs(dz) <= trust)
    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=solver)
    except cp.SolverError:
        return dict(status="solver_error", z=None)
    wall = time.perf_counter() - t0
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return dict(status=prob.status, z=None, wall=wall)

    lam = np.asarray(c_dyn[0].dual_value, dtype=float)
    if penalty is not None:                     # h <= e and -h <= e  ->  lam = y+ - y-
        lam = lam - np.asarray(c_dyn[1].dual_value, dtype=float)
    mu = np.zeros(nlp.m_in)
    mu[:n_lin] = c_lin.dual_value
    if A_obs.shape[0]:
        mu[nlp.meta["obs_rows"][:, 0]] = c_obs.dual_value
    stats = prob.solver_stats
    return dict(status=prob.status, z=np.asarray(z.value), value=float(prob.value),
                lam=lam, mu=np.maximum(mu, 0.0), wall=wall,
                solve_time=stats.solve_time, qp_iters=stats.num_iters)


def solve_corridor(nlp, z0=None):
    """One convex QP around z0 (default: straight-line guess)."""
    z0 = np.array(nlp.z0 if z0 is None else z0, dtype=float)
    hist = History(nlp)
    hist.log(z0)
    out = corridor_qp(nlp, z0)
    if out["z"] is None:
        return hist.result("corridor-qp", z0, np.zeros(nlp.m_eq), np.zeros(nlp.m_in), 1,
                           out["status"])
    hist.log(out["z"], out["lam"], out["mu"])
    return hist.result("corridor-qp", out["z"], out["lam"], out["mu"], 1, out["status"],
                       qp_value=out["value"], qp_iters=out["qp_iters"])


def solve(nlp, z0=None, trust0=1.0, trust_max=20.0, nu0=10.0, nu_max=1e6, tol=TOL_KKT, tol_feas=TOL_FEAS,
          max_iter=100, accept=0.1):
    """Trust-region sequential convex programming with l1-penalised slacks."""
    z = np.array(nlp.z0 if z0 is None else z0, dtype=float)
    lam, mu = np.zeros(nlp.m_eq), np.zeros(nlp.m_in)
    hist = History(nlp)
    delta, nu, status, qp_time, n_qp = trust0, nu0, "max_iter", 0.0, 0

    def merit(zz):
        return nlp.f(zz) + nu * (np.abs(nlp.h(zz)).sum() + np.maximum(nlp.g(zz), 0).sum())

    it = 0
    for it in range(max_iter):
        hist.log(z, lam, mu, trust=delta, nu=nu)
        with nlp.no_count():
            r = kkt_residual(nlp, z, lam, mu)
        if r["kkt"] <= tol and r["feas"] <= tol_feas:
            status = "converged"
            break
        out = corridor_qp(nlp, z, trust=delta, penalty=nu)
        qp_time += out.get("wall", 0.0)
        n_qp += 1
        if out["z"] is None:
            status = f"qp_{out['status']}"
            break
        z_c = out["z"]
        phi = merit(z)
        pred = phi - out["value"]                # decrease predicted by the convex model
        ared = phi - merit(z_c)
        step = np.abs(z_c - z).max()
        if pred <= 1e-12 * max(1.0, abs(phi)):   # model sees no further improvement
            z, lam, mu = z_c, out["lam"], out["mu"]
            if r["feas"] > tol_feas:
                nu *= 10.0                       # stationary but infeasible: stiffen penalty
                if nu > nu_max:
                    status = "infeasible_stationary"
                    break
            else:
                delta = max(delta, 1e-3)
        elif ared >= accept * pred:
            z, lam, mu = z_c, out["lam"], out["mu"]
            if ared >= 0.75 * pred and step >= 0.9 * delta:
                delta = min(2 * delta, trust_max)
        else:
            delta *= 0.5
        if delta < 1e-9:
            status = "trust_region_collapsed"
            break

    return hist.result("scp", z, lam, mu, it, status, qp_wall=qp_time, n_qp=n_qp,
                       trust=delta, nu=nu)
