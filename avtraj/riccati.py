"""Structure-exploiting solution of the interior-point KKT system by a Riccati
recursion (the approach of FATROP, Vanroye et al. 2023, and HPIPM).

The Newton step of the interior-point method solves the equality-constrained QP

    min_dz  1/2 dz^T H dz + q^T dz    s.t.   J_h dz + h = 0,

where H = W + J_g^T Sigma J_g + delta I.  For our multiple-shooting NLP,
z = (x_0..x_N, u_0..u_{N-1}) and J_h encodes  dx_0 = -h_0  and the linearised
dynamics  dx_{k+1} = A_k dx_k + B_k du_k - h_{k+1}.  H couples only (x_k, x_k),
(x_k, u_k), (u_k, u_k) and (u_k, u_{k-1}); the last block comes from the
control-smoothness cost and the steering-rate constraints.  We make the problem
stage-wise by *lifting* the previous control into the state,

    xi_k = (dx_k, du_{k-1}),   xi_{k+1} = Ahat_k xi_k + Bhat_k du_k + chat_k,

and solve it by a backward Riccati recursion and a forward rollout: O(N) work
instead of the O(n^3) of a dense factorisation.  The KKT matrix has the
inertia (n, m_eq, 0) required by the interior-point method if and only if every
Q_uu block in the recursion is positive definite, so inertia correction is a
by-product (a failed 2x2 Cholesky => increase delta).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class InertiaError(np.linalg.LinAlgError):
    """Some Q_uu block is not positive definite: H needs a larger shift."""


class StageIndex:
    """Index bookkeeping for z = (x_0..x_N, u_0..u_{N-1})."""

    def __init__(self, N, nx=4, nu=2):
        self.N, self.nx, self.nu = N, nx, nu
        self.x = [np.arange(nx * k, nx * (k + 1)) for k in range(N + 1)]
        off = nx * (N + 1)
        self.u = [np.arange(off + nu * k, off + nu * (k + 1)) for k in range(N)]
        # equality rows: block 0 = initial condition, block k+1 = dynamics k -> k+1
        self.h = [np.arange(nx * k, nx * (k + 1)) for k in range(N + 1)]


def _block(M, rows, cols):
    return M[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].toarray()


def riccati_solve(H, q, Jh, h, idx: StageIndex):
    """Solve  [H Jh^T; Jh 0] [dz; dlam] = [-q; -h]  by a lifted Riccati recursion.

    H, Jh: scipy sparse; q, h: arrays.  Returns (dz, dlam).  Raises InertiaError
    if the KKT matrix does not have the interior-point inertia (n, m_eq, 0)."""
    N, nx, nu = idx.N, idx.nx, idx.nu
    ns = nx + nu                                   # lifted state: (dx_k, du_{k-1})
    H = sp.csr_matrix(H)
    Jh = sp.csr_matrix(Jh)

    # ---- stage data ----
    A, B = [], []
    for k in range(N):
        A.append(-_block(Jh, idx.h[k + 1], idx.x[k]))
        B.append(-_block(Jh, idx.h[k + 1], idx.u[k]))
    Qxx = [_block(H, idx.x[k], idx.x[k]) for k in range(N + 1)]
    Qux = [_block(H, idx.u[k], idx.x[k]) for k in range(N)]
    Quu = [_block(H, idx.u[k], idx.u[k]) for k in range(N)]
    Qup = [np.zeros((nu, nu))] + [_block(H, idx.u[k], idx.u[k - 1]) for k in range(1, N)]
    qx = [q[idx.x[k]] for k in range(N + 1)]
    qu = [q[idx.u[k]] for k in range(N)]

    def lifted(k):
        """Ahat, Bhat, chat for xi_{k+1} = Ahat xi_k + Bhat du_k + chat."""
        Ah = np.zeros((ns, ns))
        Ah[:nx, :nx] = A[k]
        Bh = np.vstack([B[k], np.eye(nu)])
        ch = np.concatenate([-h[idx.h[k + 1]], np.zeros(nu)])
        return Ah, Bh, ch

    # ---- backward Riccati recursion ----
    P = np.zeros((ns, ns))
    P[:nx, :nx] = Qxx[N]
    p = np.concatenate([qx[N], np.zeros(nu)])
    Ps, ps, Ks, ks = [None] * (N + 1), [None] * (N + 1), [None] * N, [None] * N
    Ps[N], ps[N] = P, p
    for k in range(N - 1, -1, -1):
        Ah, Bh, ch = lifted(k)
        Sxx = np.zeros((ns, ns))
        Sxx[:nx, :nx] = Qxx[k]
        Sux = np.hstack([Qux[k], Qup[k]])
        sx = np.concatenate([qx[k], np.zeros(nu)])
        Pc_p = P @ ch + p
        Gxx = Sxx + Ah.T @ P @ Ah
        Guu = Quu[k] + Bh.T @ P @ Bh
        Gux = Sux + Bh.T @ P @ Ah
        gx = sx + Ah.T @ Pc_p
        gu = qu[k] + Bh.T @ Pc_p
        Guu = 0.5 * (Guu + Guu.T)
        try:
            Lc = np.linalg.cholesky(Guu)
        except np.linalg.LinAlgError:
            raise InertiaError(f"Q_uu not positive definite at stage {k}")
        K = -np.linalg.solve(Lc.T, np.linalg.solve(Lc, Gux))
        kff = -np.linalg.solve(Lc.T, np.linalg.solve(Lc, gu))
        P = Gxx + Gux.T @ K
        P = 0.5 * (P + P.T)
        p = gx + Gux.T @ kff
        Ps[k], ps[k], Ks[k], ks[k] = P, p, K, kff

    # ---- forward rollout ----
    dz = np.zeros(H.shape[0])
    dlam = np.zeros(Jh.shape[0])
    xi = np.concatenate([-h[idx.h[0]], np.zeros(nu)])
    for k in range(N):
        v = Ks[k] @ xi + ks[k]
        dz[idx.x[k]] = xi[:nx]
        dz[idx.u[k]] = v
        dlam[idx.h[k]] = -(Ps[k] @ xi + ps[k])[:nx]
        Ah, Bh, ch = lifted(k)
        xi = Ah @ xi + Bh @ v + ch
    dz[idx.x[N]] = xi[:nx]
    dlam[idx.h[N]] = -(Ps[N] @ xi + ps[N])[:nx]
    return dz, dlam


def check_structure(H, idx: StageIndex):
    """True if H has no couplings outside the blocks the recursion uses."""
    allowed = sp.lil_matrix(H.shape, dtype=bool)
    for k in range(idx.N + 1):
        allowed[np.ix_(idx.x[k], idx.x[k])] = True
    for k in range(idx.N):
        for a, b in [(idx.u[k], idx.x[k]), (idx.x[k], idx.u[k]), (idx.u[k], idx.u[k])]:
            allowed[np.ix_(a, b)] = True
        if k:
            allowed[np.ix_(idx.u[k], idx.u[k - 1])] = True
            allowed[np.ix_(idx.u[k - 1], idx.u[k])] = True
    Hc = sp.coo_matrix(H)
    A = allowed.tocsr()
    nz = np.abs(Hc.data) > 0
    return bool(np.all(np.asarray(A[Hc.row[nz], Hc.col[nz]]).ravel()))
