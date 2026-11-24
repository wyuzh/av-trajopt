"""Small dense linear-algebra helpers shared by the Newton-type solvers."""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def shifted_cholesky(H, beta=1e-8):
    """Factor H + tau*I with the smallest tau in {0, beta, 2beta, ...} that makes
    it positive definite (Nocedal & Wright, Alg. 3.3).  Returns (factor, tau)."""
    n = H.shape[0]
    scale = max(1.0, float(np.abs(np.diag(H)).max()))
    beta = beta * scale
    dmin = float(np.diag(H).min())
    tau = 0.0 if dmin > 0 else -dmin + beta
    eye = np.eye(n)
    for _ in range(100):
        try:
            return sla.cho_factor(H + tau * eye, lower=True, check_finite=False), tau
        except np.linalg.LinAlgError:
            tau = max(2.0 * tau, beta)
    raise np.linalg.LinAlgError("shifted_cholesky: could not make H positive definite")


def newton_direction(H, g, beta=1e-8):
    """Solve (H + tau I) d = -g with the positive-definite shift above."""
    c, tau = shifted_cholesky(H, beta)
    return -sla.cho_solve(c, g, check_finite=False), tau


def make_positive_definite(H, beta=1e-8):
    """H + tau*I with tau chosen as in `shifted_cholesky`."""
    _, tau = shifted_cholesky(H, beta)
    return H + tau * np.eye(H.shape[0])


def inertia(K, tol=1e-10):
    """(#positive, #negative, #zero) eigenvalues of symmetric K via an LDL^T
    factorisation and Sylvester's law of inertia."""
    _, D, _ = sla.ldl(K, lower=True, check_finite=False)
    eig = []
    i, n = 0, D.shape[0]
    while i < n:
        if i + 1 < n and D[i + 1, i] != 0.0:
            eig.extend(np.linalg.eigvalsh(D[i:i + 2, i:i + 2]))
            i += 2
        else:
            eig.append(D[i, i])
            i += 1
    eig = np.asarray(eig)
    t = tol * max(1.0, float(np.abs(eig).max()))
    return int((eig > t).sum()), int((eig < -t).sum()), int((np.abs(eig) <= t).sum())
