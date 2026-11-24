"""KKT residuals, iteration history and the common Result record.

All solvers log through `History`, so convergence curves are measured in
exactly the same way for every method.  Time spent on logging (evaluating
KKT residuals for the plots) is excluded from the reported solve time, and
the evaluations it needs are not counted.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

TOL_FEAS = 1e-6     # max constraint violation for "converged"
TOL_KKT = 1e-6      # scaled KKT residual for "converged"


def _inf(v):
    return float(np.abs(v).max()) if np.size(v) else 0.0


def kkt_residual(nlp, z, lam, mu):
    """Scaled first-order optimality residuals for  L = f + lam^T h + mu^T g.

    stat : ||grad_z L||_inf / s_d
    feas : max(||h||_inf, max(g, 0))
    dual : max(-mu, 0)
    comp : max |mu_i * g_i| / s_d
    s_d is the IPOPT-style scaling  max(100, mean|multiplier|) / 100.
    """
    gf = nlp.grad_f(z)
    h, g = nlp.h(z), nlp.g(z)
    r = gf.copy()
    if nlp.m_eq:
        r += nlp.jac_h_sp(z).T @ lam
    if nlp.m_in:
        r += nlp.jac_g_sp(z).T @ mu
    m = nlp.m_eq + nlp.m_in
    s_d = max(100.0, (np.abs(lam).sum() + np.abs(mu).sum()) / max(m, 1)) / 100.0
    stat = _inf(r) / s_d
    feas = max(_inf(h), float(max(g.max(), 0.0)) if g.size else 0.0)
    dual = float(max((-mu).max(), 0.0)) if mu.size else 0.0
    comp = _inf(mu * g) / s_d if mu.size else 0.0
    return dict(stat=stat, feas=feas, dual=dual, comp=comp, kkt=max(stat, dual, comp))


@dataclass
class Result:
    solver: str
    status: str
    success: bool
    f: float
    viol: float
    kkt: float
    iters: int
    time: float
    z: np.ndarray
    lam: np.ndarray
    mu: np.ndarray
    counts: dict
    history: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def summary(self):
        d = asdict(self)
        for k in ("z", "lam", "mu", "history"):
            d.pop(k)
        return d


class History:
    """Per-iteration log; `elapsed()` excludes the logging overhead."""

    def __init__(self, nlp):
        self.nlp = nlp
        self.records = []
        self._excluded = 0.0
        nlp.reset_counts()
        self.t0 = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.t0 - self._excluded

    def log(self, z, lam=None, mu=None, mult_fn=None, **extra):
        """Record f, violation and KKT residual at z.  Multipliers are given
        directly or through `mult_fn(z) -> (lam, mu)` (evaluated off the clock)."""
        t_in = time.perf_counter()
        t = t_in - self.t0 - self._excluded
        with self.nlp.no_count():
            if mult_fn is not None:
                lam, mu = mult_fn(z)
            lam = np.zeros(self.nlp.m_eq) if lam is None else lam
            mu = np.zeros(self.nlp.m_in) if mu is None else mu
            r = kkt_residual(self.nlp, z, lam, mu)
            rec = dict(k=len(self.records), t=t, f=self.nlp.f(z), viol=r["feas"],
                       kkt=r["kkt"], stat=r["stat"], comp=r["comp"], **extra)
        self.records.append(rec)
        self._excluded += time.perf_counter() - t_in
        return rec

    def result(self, solver, z, lam, mu, iters, status, **info):
        t = self.elapsed()
        with self.nlp.no_count():
            r = kkt_residual(self.nlp, z, lam, mu)
            f = self.nlp.f(z)
        success = r["feas"] <= TOL_FEAS and r["kkt"] <= TOL_KKT
        return Result(solver=solver, status=status, success=bool(success), f=f,
                      viol=r["feas"], kkt=r["kkt"], iters=int(iters), time=t,
                      z=np.asarray(z), lam=np.asarray(lam), mu=np.asarray(mu),
                      counts=dict(self.nlp.counts), history=self.records, info=info)
