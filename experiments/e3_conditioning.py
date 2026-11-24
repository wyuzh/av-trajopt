"""E3 - Why the penalty parameter cannot simply be made huge.

(a) 2-D illustration: contours of the quadratic penalty function of
        min (x-2)^2 + (y-1)^2   s.t.  x^2 + y^2 <= 1
    for rho = 1, 10, 100 with gradient-descent and Newton paths - the
    feasible boundary turns into a long, narrow valley.
(b) Trajectory problem ('single'): one penalty subproblem per fixed rho in
    {1e0, ..., 1e6}, solved by GD, Adam, L-BFGS and Newton.  We record inner
    iterations to ||grad P||_inf <= 1e-6, the final violation (theory: O(1/rho))
    and the condition number of the Hessian of P at the solution (theory: O(rho)).
(c) Contrast: the augmented Lagrangian method with the *same fixed* rho
    reaches feasibility through its multiplier updates.

Produces results/e3_conditioning.jsonl and figures/e3_toy.*, figures/e3_conditioning.*
"""
import argparse

import numpy as np
from scipy.optimize import minimize_scalar
from common import RESULTS, initial_guess, load, run_all

from avtraj import build_nlp, scenarios
from avtraj.solvers import penalty

RHOS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
INNER = ["gd", "adam", "lbfgs", "newton"]
OUT = "e3_conditioning.jsonl"
MAX_INNER = 20000


def run():
    cases = [dict(scenario="single", solver=f"penalty-{m}", init="straight", rho=r,
                  opts=dict(fixed_rho=r, max_inner=MAX_INNER, lr=1e-2))
             for r in RHOS for m in INNER]
    cases += [dict(scenario="single", solver="alm-newton", init="straight", rho=r,
                   opts=dict(rho0=r, rho_factor=1.0, max_outer=200))
              for r in RHOS]
    run_all(cases, OUT)
    # Hessian condition numbers at the Newton solutions (cheap, done serially)
    nlp = build_nlp(scenarios.get("single"))
    conds = {}
    for r in RHOS:
        res = penalty.solve(nlp, initial_guess(nlp, "straight"), inner="newton", fixed_rho=r)
        _, _, HP = penalty.penalty_functions(nlp, r)
        ev = np.abs(np.linalg.eigvalsh(HP(res.z)))
        conds[r] = float(ev.max() / max(ev.min(), 1e-300))
    (RESULTS / "e3_cond.txt").write_text("\n".join(f"{r} {c}" for r, c in conds.items()))


def toy_figure():
    import matplotlib.pyplot as plt
    from plots import save

    def P(x, y, rho):
        return (x - 2) ** 2 + (y - 1) ** 2 + 0.5 * rho * np.maximum(x ** 2 + y ** 2 - 1, 0) ** 2

    def grad(p, rho):
        x, y = p
        c = max(x ** 2 + y ** 2 - 1, 0)
        return np.array([2 * (x - 2) + 2 * rho * c * x, 2 * (y - 1) + 2 * rho * c * y])

    def hess(p, rho):
        x, y = p
        c = x ** 2 + y ** 2 - 1
        H = 2 * np.eye(2)
        if c > 0:
            H += rho * (4 * np.outer([x, y], [x, y]) + 2 * c * np.eye(2))
        return H

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    xs, ys = np.meshgrid(np.linspace(-0.3, 2.3, 400), np.linspace(-0.6, 1.6, 400))
    for ax, rho in zip(axes, [1, 10, 100]):
        Z = P(xs, ys, rho)
        ax.contour(xs, ys, np.log(Z - Z.min() + 1e-3), 25, cmap="viridis", linewidths=0.6)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="k", lw=1.2))
        for method, col in (("gd", "#d62728"), ("newton", "#1f77b4")):
            p, path = np.array([0.2, 1.3]), []
            for _ in range(40):
                path.append(p.copy())
                g = grad(p, rho)
                if np.linalg.norm(g) < 1e-10:
                    break
                if method == "gd":          # steepest descent with exact line search
                    d = -g
                    t = minimize_scalar(lambda a: P(*(p + a * d), rho), bounds=(0, 10),
                                        method="bounded", options={"xatol": 1e-12}).x
                else:                       # Newton with backtracking
                    d = -np.linalg.solve(hess(p, rho), g)
                    t = 1.0
                    while P(*(p + t * d), rho) > P(*p, rho) + 1e-4 * t * g @ d:
                        t *= 0.5
                p = p + t * d
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], "-o", ms=2, lw=0.9, color=col,
                    label={"gd": "gradient descent (exact line search)",
                           "newton": "Newton"}[method])
        xstar = np.array([2, 1]) / np.sqrt(5)
        ax.plot(*xstar, "k*", ms=9, label="constrained optimum")
        ax.set(title=rf"$\rho = {rho}$  (40 iterations each)", aspect="equal",
               xlim=(-0.3, 2.3), ylim=(-0.6, 1.6))
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(r"Quadratic penalty $P(x)=f(x)+\frac{\rho}{2}\max(0,g(x))^2$: "
                 "larger $\\rho$ turns the boundary into a narrow valley", y=1.02)
    save(fig, "e3_toy")


def plot():
    import matplotlib.pyplot as plt
    from plots import color, save

    recs = load(OUT)
    conds = dict(tuple(map(float, l.split())) for l in (RESULTS / "e3_cond.txt").read_text().splitlines())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for m in INNER:
        v = f"penalty-{m}"
        rs = [r for r in recs if r["solver"] == v]
        rs.sort(key=lambda r: r["rho"])
        it = [r["iters"] for r in rs]
        axes[0].loglog([r["rho"] for r in rs], it, "o-", color=color(v), ms=3, label=m.upper()
                       if m != "newton" else "Newton")
        axes[1].loglog([r["rho"] for r in rs], [max(r["viol"], 1e-16) for r in rs], "o-",
                       color=color(v), ms=3, label=f"penalty ({m})")
    axes[0].axhline(MAX_INNER, color="k", ls=":", lw=0.8)
    axes[0].text(1.2, MAX_INNER * 1.15, "iteration cap", fontsize=7)
    alm = sorted([r for r in recs if r["solver"] == "alm-newton"], key=lambda r: r["rho"])
    axes[1].loglog([r["rho"] for r in alm], [max(r["viol"], 1e-16) for r in alm], "s-",
                   color=color("alm-newton"), ms=4, label="ALM, same fixed rho")
    rr = np.array(RHOS)
    axes[1].loglog(rr, 2.0 / rr, "k--", lw=0.8, label=r"$O(1/\rho)$")
    axes[2].loglog(list(conds), list(conds.values()), "o-", color="k", ms=3,
                   label=r"cond$(\nabla^2 P)$ at solution")
    axes[2].loglog(rr, conds[RHOS[0]] * rr / rr[0], "k--", lw=0.8, label=r"$O(\rho)$")
    axes[0].set(xlabel=r"penalty parameter $\rho$", ylabel=r"inner iterations to $\|\nabla P\|\leq 10^{-6}$")
    axes[1].set(xlabel=r"penalty parameter $\rho$", ylabel="max constraint violation")
    axes[2].set(xlabel=r"penalty parameter $\rho$", ylabel="condition number")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Penalty method on the 'single' scenario: accuracy vs conditioning", y=1.02)
    fig.tight_layout()
    save(fig, "e3_conditioning")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run()
    toy_figure()
    plot()
