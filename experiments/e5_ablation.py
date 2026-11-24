"""E5 - Ablation and sensitivity studies of algorithmic design choices.

  M1 penalty : rho update factor {2, 10, 100}
  M2 IPM     : globalisation {filter, l1-merit} x barrier update {monotone, fixed}
  M3 ALM     : initial rho {1, 10, 100, 1000}, inner solver {Newton, L-BFGS}
  M4 SQP     : Hessian {exact+convexified, exact+diagonal shift, damped BFGS}
               x globalisation {filter, l1-merit}
  SCP        : initial trust radius {0.25, 1, 4}

Each variant runs on the three named scenarios plus 20 random fields with
M = 6 obstacles (straight-line initial guess).

Produces results/e5_ablation.jsonl and results/e5_table.md
"""
import argparse
from collections import defaultdict

import numpy as np
from common import RESULTS, load, run_all

VARIANTS = {
    "M1 penalty":     [("penalty-newton", {"rho_factor": f}, f"rho x{f:g}") for f in (2, 10, 100)],
    "M2 interior point": [("ipm", {"globalization": g, "mu_strategy": m}, f"{g} / {m} mu")
                          for g in ("filter", "merit") for m in ("monotone", "fixed")],
    "M3 ALM": [("alm-newton", {"rho0": r}, f"Newton, rho0={r:g}") for r in (1, 10, 100, 1000)]
              + [("alm-lbfgs", {"rho0": 10}, "L-BFGS, rho0=10")],
    "M4 SQP": [("sqp-exact", {"globalization": g}, f"exact+convexify / {g}") for g in ("filter", "merit")]
              + [("sqp-shift", {"globalization": "filter"}, "exact+shift / filter")]
              + [("sqp-bfgs", {"globalization": g}, f"BFGS / {g}") for g in ("filter", "merit")],
    "SCP": [("scp", {"trust0": t}, f"trust0={t:g}") for t in (0.25, 1.0, 4.0)],
    "PANOC-ALM": [("panoc-alm", {"memory": m}, f"L-BFGS memory {m}") for m in (3, 10, 30)]
                 + [("panoc-alm", {"rho0": r}, f"rho0={r:g}") for r in (1, 100)],
}
INSTANCES = ["single", "slalom", "narrow"] + [f"random_M6_s{s}" for s in range(20)]
OUT = "e5_ablation.jsonl"


def run():
    cases = []
    for group, vs in VARIANTS.items():
        for solver, opts, name in vs:
            for sc in INSTANCES:
                cases.append(dict(scenario=sc, solver=solver, init="straight", opts=opts,
                                  group=group, variant=name))
    run_all(cases, OUT)


def plot():
    recs = load(OUT)
    inst_best = defaultdict(lambda: np.inf)
    for r in recs:
        if r.get("success"):
            inst_best[r["scenario"]] = min(inst_best[r["scenario"]], r["f"])
    solvable = set(inst_best)
    lines = ["| method | variant | success | median iters | median time [s] | "
             "median Hessian/Jacobian evals |", "|---|---|---|---|---|---|"]
    for group, vs in VARIANTS.items():
        for _, _, name in vs:
            rs = [r for r in recs if r["variant"] == name and r["group"] == group
                  and r["scenario"] in solvable]
            ok = [r for r in rs if r.get("success")]
            med = (lambda xs: np.median(xs) if xs else np.nan)
            ev = [r["counts"].get("hess", 0) + r["counts"].get("jac", 0) for r in ok]
            lines.append(f"| {group} | {name} | {len(ok)}/{len(rs)} | "
                         f"{med([r['iters'] for r in ok]):.0f} | {med([r['time'] for r in ok]):.2f} | "
                         f"{med(ev):.0f} |")
    (RESULTS / "e5_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run()
    plot()
