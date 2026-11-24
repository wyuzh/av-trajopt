"""Reference solver: IPOPT (via CasADi) on the identical NLP with exact Hessians.

Used as ground truth for objective values / local optima, not as a team member's
algorithm.  Its per-iteration objective and infeasibility are taken from the
IPOPT log; per-iteration times are interpolated from the total wall time.
"""
from __future__ import annotations

import casadi as ca
import numpy as np

from ..metrics import TOL_KKT, History


def solve(nlp, z0=None, tol=TOL_KKT, max_iter=3000, linear_solver="mumps"):
    z0 = np.array(nlp.z0 if z0 is None else z0, dtype=float)
    s = nlp.sym
    opts = {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
            "ipopt.tol": tol, "ipopt.constr_viol_tol": tol, "ipopt.max_iter": max_iter,
            "ipopt.linear_solver": linear_solver}
    S = ca.nlpsol("ipopt", "ipopt", {"x": s["z"], "f": s["f"], "g": ca.vertcat(s["h"], s["g"])},
                  opts)
    hist = History(nlp)
    lbg = np.concatenate([np.zeros(nlp.m_eq), -np.inf * np.ones(nlp.m_in)])
    ubg = np.zeros(nlp.m_eq + nlp.m_in)
    sol = S(x0=z0, lbg=lbg, ubg=ubg)
    t_total = hist.elapsed()
    st = S.stats()
    z = sol["x"].full().ravel()
    lg = sol["lam_g"].full().ravel()
    lam, mu = lg[:nlp.m_eq], np.maximum(lg[nlp.m_eq:], 0.0)

    its = st.get("iterations", {})
    objs, infs = its.get("obj", []), its.get("inf_pr", [])
    for k, (fo, ip) in enumerate(zip(objs, infs)):
        hist.records.append(dict(k=k, t=t_total * k / max(len(objs) - 1, 1), f=float(fo),
                                 viol=float(ip), kkt=float(its["inf_du"][k]), stat=np.nan,
                                 comp=np.nan))
    res = hist.result("ipopt", z, lam, mu, st.get("iter_count", len(objs)),
                      st.get("return_status", "unknown"))
    res.time = t_total
    return res
