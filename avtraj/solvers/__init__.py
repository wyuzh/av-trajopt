"""Solver registry.  Every entry has the signature  solve(nlp, z0=None, **opts) -> Result.

    M1  penalty-*   quadratic penalty + unconstrained inner solver
    M2  ipm         primal-dual interior-point (log-barrier) method
    M3  alm-*       augmented Lagrangian / method of multipliers
    M4  sqp-*       sequential quadratic programming
    new ipm-riccati structure-exploiting interior point (Riccati KKT solve, FATROP-style)
    new panoc-alm   augmented Lagrangian with the PANOC inner solver (OpEn / alpaqa-style)
    +   scp         sequential convex programming (convex corridor QPs)
    ref ipopt       IPOPT reference solution
"""
from functools import partial

from . import alm, convex, interior_point, ipopt_ref, panoc_alm, penalty, sqp

SOLVERS = {
    "penalty-newton": partial(penalty.solve, inner="newton"),
    "penalty-gd": partial(penalty.solve, inner="gd"),
    "penalty-adam": partial(penalty.solve, inner="adam"),
    "penalty-lbfgs": partial(penalty.solve, inner="lbfgs"),
    "ipm": interior_point.solve,
    "ipm-riccati": partial(interior_point.solve, kkt="riccati"),
    "panoc-alm": panoc_alm.solve,
    "alm-newton": partial(alm.solve, inner="newton"),
    "alm-lbfgs": partial(alm.solve, inner="lbfgs"),
    "sqp-exact": partial(sqp.solve, hessian="exact"),
    "sqp-bfgs": partial(sqp.solve, hessian="bfgs"),
    "sqp-shift": partial(sqp.solve, hessian="exact-shift"),
    "scp": convex.solve,
    "corridor-qp": convex.solve_corridor,
    "ipopt": ipopt_ref.solve,
}


def get(name):
    return SOLVERS[name]
