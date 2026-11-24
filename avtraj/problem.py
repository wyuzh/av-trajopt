"""Generic smooth nonlinear program

        minimize    f(z)
        subject to  h(z) = 0          (m_eq equality constraints)
                    g(z) <= 0         (m_in inequality constraints)

Every solver in this repository talks to a problem *only* through this class,
so all methods see exactly the same functions, the same exact derivatives
(CasADi automatic differentiation) and the same evaluation counters.

Sign convention for multipliers (used everywhere):
        L(z, lam, mu) = f(z) + lam^T h(z) + mu^T g(z),   mu >= 0.
"""
from __future__ import annotations

from contextlib import contextmanager

import casadi as ca
import numpy as np
from scipy.sparse import csc_matrix as sp_csc


class _DenseFn:
    """CasADi function with a dense numpy output, evaluated through CasADi's
    zero-copy buffer interface (much faster than converting DM -> numpy).
    The CasADi function is built lazily on first use."""

    def __init__(self, name, args, expr):
        self._spec = (name, args, expr)
        self._fn = None

    def _build(self):
        name, args, expr = self._spec
        expr = ca.densify(expr)
        self.shape = expr.shape
        self._fn = ca.Function(name, args, [expr])
        try:
            self._buf, self._eval = self._fn.buffer()
            self._out = np.zeros(self.shape[0] * self.shape[1])
            self._buf.set_res(0, memoryview(self._out))
        except (AttributeError, RuntimeError):     # older CasADi: plain call
            self._buf = None

    def __call__(self, *xs):
        if self._fn is None:
            self._build()
        if self._buf is None:
            return self._fn(*xs).full()
        arrs = [np.ascontiguousarray(np.atleast_1d(x), dtype=float) for x in xs]
        for i, a in enumerate(arrs):
            self._buf.set_arg(i, memoryview(a))
        self._eval()
        return self._out.reshape(self.shape, order="F").copy()


class _SparseFn:
    """CasADi function with a scipy.sparse CSC output (structural nonzeros only).
    Used by the structure-exploiting solvers and for large horizons."""

    def __init__(self, name, args, expr):
        self._spec = (name, args, expr)
        self._fn = None

    def _build(self):
        name, args, expr = self._spec
        self._fn = ca.Function(name, args, [expr])
        sp = self._fn.sparsity_out(0)
        self.shape = (sp.size1(), sp.size2())
        self._indptr = np.array(sp.colind(), dtype=np.int64)
        self._indices = np.array(sp.row(), dtype=np.int64)
        self._out = np.zeros(sp.nnz())
        self._buf, self._eval = self._fn.buffer()
        self._buf.set_res(0, memoryview(self._out))

    def __call__(self, *xs):
        if self._fn is None:
            self._build()
        arrs = [np.ascontiguousarray(np.atleast_1d(x), dtype=float) for x in xs]
        for i, a in enumerate(arrs):
            self._buf.set_arg(i, memoryview(a))
        self._eval()
        return sp_csc((self._out.copy(), self._indices, self._indptr), shape=self.shape)


class NLP:
    def __init__(self, z, f, h, g, z0, name="nlp", meta=None):
        self.name = name
        self.meta = meta or {}
        self.n = z.numel()
        self.m_eq = h.numel()
        self.m_in = g.numel()
        self.z0 = np.asarray(z0, dtype=float).ravel()
        self.sym = dict(z=z, f=f, h=h, g=g)

        lam = ca.SX.sym("lam", self.m_eq)
        mu = ca.SX.sym("mu", self.m_in)
        sigma = ca.SX.sym("sigma")
        lag = sigma * f + ca.mtimes(lam.T, h) + ca.mtimes(mu.T, g)

        self._f = _DenseFn("f", [z], f)
        self._grad_f = _DenseFn("grad_f", [z], ca.gradient(f, z))
        self._h = _DenseFn("h", [z], h)
        self._jac_h = _DenseFn("jac_h", [z], ca.jacobian(h, z))
        self._g = _DenseFn("g", [z], g)
        self._jac_g = _DenseFn("jac_g", [z], ca.jacobian(g, z))
        hess = ca.hessian(lag, z)[0]
        self._hess = _DenseFn("hess_lag", [z, lam, mu, sigma], hess)
        self._jac_h_sp = _SparseFn("jac_h_sp", [z], ca.jacobian(h, z))
        self._jac_g_sp = _SparseFn("jac_g_sp", [z], ca.jacobian(g, z))
        self._hess_sp = _SparseFn("hess_lag_sp", [z, lam, mu, sigma], hess)

        self.counts = dict(f=0, grad=0, h=0, g=0, jac=0, hess=0)
        self._counting = True

    # ------------------------------------------------------------------ #
    # evaluation counters (logging/diagnostics are excluded via no_count)
    def _tick(self, key):
        if self._counting:
            self.counts[key] += 1

    def reset_counts(self):
        for k in self.counts:
            self.counts[k] = 0

    @contextmanager
    def no_count(self):
        old, self._counting = self._counting, False
        try:
            yield
        finally:
            self._counting = old

    # ------------------------------------------------------------------ #
    # problem functions (numpy in, numpy out)
    def f(self, z):
        self._tick("f")
        return float(self._f(z)[0, 0])

    def grad_f(self, z):
        self._tick("grad")
        return self._grad_f(z).ravel()

    def h(self, z):
        self._tick("h")
        return self._h(z).ravel()

    def g(self, z):
        self._tick("g")
        return self._g(z).ravel()

    def jac_h(self, z):
        self._tick("jac")
        return self._jac_h(z).reshape(self.m_eq, self.n)

    def jac_g(self, z):
        self._tick("jac")
        return self._jac_g(z).reshape(self.m_in, self.n)

    def hess_lag(self, z, lam, mu, sigma=1.0):
        """Hessian of sigma*f + lam^T h + mu^T g with respect to z."""
        self._tick("hess")
        return self._hess(z, lam, mu, sigma)

    # sparse (scipy CSC) variants: same values, counted the same way
    def jac_h_sp(self, z):
        self._tick("jac")
        return self._jac_h_sp(z)

    def jac_g_sp(self, z):
        self._tick("jac")
        return self._jac_g_sp(z)

    def hess_lag_sp(self, z, lam, mu, sigma=1.0):
        self._tick("hess")
        return self._hess_sp(z, lam, mu, sigma)

    # ------------------------------------------------------------------ #
    def violation(self, z):
        """Maximum constraint violation  max(||h||_inf, max(g, 0))."""
        v = 0.0
        if self.m_eq:
            v = max(v, np.abs(self.h(z)).max())
        if self.m_in:
            v = max(v, max(self.g(z).max(), 0.0))
        return v

    def __repr__(self):
        return f"NLP({self.name}: n={self.n}, m_eq={self.m_eq}, m_in={self.m_in})"
