"""E6 - What the Lagrange multipliers tell us, and why initialisation matters.

(a) Complementary slackness: along the optimal slalom trajectory, the obstacle
    multipliers mu_kj are non-zero only where the clearance is exactly zero.
(b) Sensitivity: the multipliers predict how the optimal cost changes if an
    obstacle grows,  df*/dr_j = sum_k mu_kj * dg_kj/dR_j  with
    g = (R^2 - d^2)/(2R), dg/dR = (R^2 + d^2)/(2R^2)  (= 1 on the boundary).
    We compare with central finite differences obtained by re-solving the NLP.
(c) Nonconvexity: in the 'trap' scenario a lane-centre initial guess leads
    to a point of local infeasibility, while shifting the initial guess into
    the other homotopy class (passing above the third obstacle) is solvable.

Produces results/e6_sensitivity.md and figures/e6_multipliers.*, figures/e6_trap.*
"""
import json

import numpy as np
from common import RESULTS, initial_guess

from avtraj import build_nlp, min_clearance, scenarios, unpack
from avtraj.solvers import SOLVERS

EPS = 1e-3


def solve(sc, solver, z0, **opts):
    nlp = build_nlp(sc)
    return nlp, SOLVERS[solver](nlp, z0, **opts)


def sensitivity(name="slalom", solver="ipm"):
    sc = scenarios.get(name)
    nlp, base = solve(sc, solver, None)
    assert base.success, base.status
    rows, veh = [], nlp.meta["vehicle"]
    X, _ = unpack(base.z, sc.N)
    obs_rows = nlp.meta["obs_rows"]
    for j in range(sc.M):
        R = sc.obstacles[j, 2] + veh.radius
        sel = obs_rows[obs_rows[:, 2] == j]
        d2 = ((X[sel[:, 1], :2] - sc.obstacles[j, :2]) ** 2).sum(1)
        pred = float((base.mu[sel[:, 0]] * (R ** 2 + d2) / (2 * R ** 2)).sum())
        fs = []
        for sgn in (+1, -1):
            sc2 = scenarios.get(name)
            sc2.obstacles[j, 2] += sgn * EPS
            _, r = solve(sc2, "ipopt", base.z, tol=1e-10)
            fs.append(r.f)
        fd = (fs[0] - fs[1]) / (2 * EPS)
        rows.append(dict(obstacle=j, predicted=pred, finite_diff=fd,
                         active_steps=int((base.mu[sel[:, 0]] > 1e-6).sum())))
    return nlp, base, rows


def main():
    import matplotlib.pyplot as plt
    from plots import color, draw_scene, label, save

    nlp, base, rows = sensitivity()
    sc = nlp.meta["scenario"]
    N, dt = sc.N, sc.dt
    X, _ = unpack(base.z, N)
    t = np.arange(N + 1) * dt
    obs_rows = nlp.meta["obs_rows"]

    fig = plt.figure(figsize=(12, 5.5))
    ax0 = fig.add_subplot(2, 2, (1, 2))
    draw_scene(ax0, sc)
    sca = ax0.scatter(X[:, 0], X[:, 1], c=t, s=10, cmap="viridis", zorder=3)
    fig.colorbar(sca, ax=ax0, label="time [s]", fraction=0.02)
    for j, (ox, oy, _) in enumerate(sc.obstacles):
        ax0.annotate(f"obstacle {j}", (ox, oy), ha="center", va="center", fontsize=7, color="w")
    ax0.set_title(f"slalom - {label('ipm')} solution, f* = {base.f:.4f}")

    ax1 = fig.add_subplot(2, 2, 3)
    ax2 = ax1.twinx()
    for j in range(sc.M):
        sel = obs_rows[obs_rows[:, 2] == j]
        R = sc.obstacles[j, 2] + nlp.meta["vehicle"].radius
        clear = np.linalg.norm(X[sel[:, 1], :2] - sc.obstacles[j, :2], axis=1) - R
        c = f"C{j}"
        ax1.plot(t[sel[:, 1]], base.mu[sel[:, 0]], "-", color=c, lw=1.5, label=rf"$\mu_{{k,{j}}}$")
        ax2.plot(t[sel[:, 1]], clear, ":", color=c, lw=1.0)
    ax1.set(xlabel="time [s]", ylabel=r"obstacle multiplier $\mu_{kj}$ (solid)")
    ax2.set_ylabel("clearance [m] (dotted)")
    ax2.set_ylim(-0.5, 8)
    ax1.legend(loc="upper right")
    ax1.set_title(r"complementary slackness: $\mu_{kj} > 0$ only at zero clearance")

    ax3 = fig.add_subplot(2, 2, 4)
    idx = np.arange(len(rows))
    ax3.bar(idx - 0.18, [r["predicted"] for r in rows], 0.36, label=r"predicted $\sum_k \mu_{kj}\,\partial g/\partial R$")
    ax3.bar(idx + 0.18, [r["finite_diff"] for r in rows], 0.36, label="finite difference (re-solve)")
    ax3.set(xticks=idx, xticklabels=[f"obstacle {r['obstacle']}" for r in rows],
            ylabel=r"$\partial f^\star / \partial r_j$")
    ax3.legend(fontsize=7)
    ax3.set_title("multipliers = sensitivities of the optimal cost")
    fig.tight_layout()
    save(fig, "e6_multipliers")

    lines = ["| obstacle | active time steps | predicted df*/dr | finite difference | abs. error |",
             "|---|---|---|---|---|"]
    for r in rows:
        err = abs(r["predicted"] - r["finite_diff"])
        lines.append(f"| {r['obstacle']} | {r['active_steps']} | {r['predicted']:.6f} | "
                     f"{r['finite_diff']:.6f} | {err:.1e} |")
    (RESULTS / "e6_sensitivity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- trap: initialisation decides the outcome ----------------
    sc = scenarios.get("trap")
    nlp = build_nlp(sc)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    draw_scene(ax, sc)
    out = {}
    for init, ls in (("straight", ":"), ("offset:3.0", "-")):
        for v in ("ipm", "sqp-exact", "panoc-alm", "ipopt"):
            res = SOLVERS[v](nlp, initial_guess(nlp, init))
            out[f"{v}/{init}"] = dict(status=res.status, f=res.f, viol=res.viol,
                                      clearance=min_clearance(nlp, res.z))
            Xr, _ = unpack(res.z, sc.N)
            ax.plot(Xr[:, 0], Xr[:, 1], ls, color=color(v), lw=1.3,
                    label=f"{label(v)}, init={init}: {res.status}")
    ax.set_xlim(-1, 52)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=6.5)
    ax.set_title("'trap': from the lane-centre guess (dotted) only PANOC-ALM succeeds; "
                 "shifted guess (solid): all succeed", fontsize=9)
    save(fig, "e6_trap")
    (RESULTS / "e6_trap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
