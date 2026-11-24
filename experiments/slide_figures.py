"""Presentation versions of the key figures: fewer panels, larger fonts.

Reads the saved results of E1-E4 and writes figures/slide_*.png.
Run after the experiments:  python experiments/slide_figures.py
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, load
from plots import color, draw_scene, floor, label, save

from avtraj import scenarios, unpack

plt.rcParams.update({"font.size": 13, "legend.fontsize": 10.5, "axes.titlesize": 13,
                     "axes.labelsize": 13, "lines.linewidth": 2.0})
MAIN = ["ipopt", "penalty-newton", "ipm", "alm-newton", "sqp-exact", "scp"]
SHORT = {"ipopt": "IPOPT (ref.)", "penalty-newton": "M1 Penalty", "ipm": "M2 Interior point",
         "alm-newton": "M3 ALM", "sqp-exact": "M4 SQP", "sqp-bfgs": "M4 SQP-BFGS",
         "scp": "SCP"}


def traj():
    by = {(r["scenario"], r["solver"], r["init"]): r for r in load("e1_benchmark.jsonl")}
    for s in ["slalom", "narrow"]:
        sc = scenarios.get(s)
        fig, ax = plt.subplots(figsize=(9, 3.3))
        draw_scene(ax, sc)
        for v in MAIN:
            r = by[(s, v, "straight")]
            X, _ = unpack(np.array(r["z"]), sc.N)
            ax.plot(X[:, 0], X[:, 1], color=color(v), label=f"{SHORT[v]}  f*={r['f']:.3f}")
        ax.set_xlim(-1, 52)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
        save(fig, f"slide_traj_{s}")


def convergence():
    by = {(r["scenario"], r["solver"], r["init"]): r for r in load("e1_benchmark.jsonl")}
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for v in MAIN + ["sqp-bfgs"]:
        H = by[("slalom", v, "straight")]["history"]
        err = floor(np.maximum([h["viol"] for h in H], [h["kkt"] for h in H]))
        ax.semilogy(np.arange(len(H)), err, color=color(v), label=SHORT[v])
    ax.axhline(1e-6, color="k", ls=":", lw=1)
    ax.text(88, 2e-6, "tolerance", ha="right", fontsize=11)
    ax.set(xlabel="iteration", ylabel="max(violation, KKT)", xlim=(0, 90), ylim=(1e-12, 1e4),
           title="slalom, straight initial guess")
    ax.legend(loc="upper right", fontsize=9.5, ncol=2)
    fig.tight_layout()
    save(fig, "slide_conv_slalom")


def conditioning():
    recs = load("e3_conditioning.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for m, name in [("gd", "GD"), ("adam", "Adam"), ("lbfgs", "L-BFGS"), ("newton", "Newton")]:
        rs = sorted([r for r in recs if r["solver"] == f"penalty-{m}"], key=lambda r: r["rho"])
        axes[0].loglog([r["rho"] for r in rs], [r["iters"] for r in rs], "o-",
                       color=color(f"penalty-{m}"), label=name)
        if m == "newton":
            axes[1].loglog([r["rho"] for r in rs], [max(r["viol"], 1e-16) for r in rs], "o-",
                           color=color("penalty-newton"), label="Penalty")
    alm = sorted([r for r in recs if r["solver"] == "alm-newton"], key=lambda r: r["rho"])
    axes[1].loglog([r["rho"] for r in alm], [max(r["viol"], 1e-16) for r in alm], "s-",
                   color=color("alm-newton"), label="ALM (same fixed ρ)")
    axes[0].axhline(20000, color="k", ls=":", lw=1)
    axes[0].set(xlabel="penalty parameter ρ", ylabel="inner iterations", title="cost of solving P(·;ρ)")
    axes[1].set(xlabel="penalty parameter ρ", ylabel="constraint violation", title="accuracy reached")
    axes[0].legend(loc="center right")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "slide_conditioning")


def montecarlo():
    recs = load("e2_random.jsonl")
    inst = defaultdict(dict)
    for r in recs:
        inst[r["scenario"]][r["solver"]] = r
    solvable = [k for k, d in inst.items() if any(r.get("success") for r in d.values())]
    MS = sorted({r["M"] for r in recs})
    solvers = ["ipopt", "penalty-newton", "ipm", "alm-newton", "sqp-exact", "scp"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for v in solvers:
        sr = []
        for m in MS:
            keys = [k for k in solvable if inst[k][v]["M"] == m]
            sr.append(100 * np.mean([inst[k][v].get("success", False) for k in keys]))
        axes[0].plot(MS, sr, "o-", color=color(v), label=SHORT[v])
    T = np.array([[inst[k][v]["time"] if inst[k][v].get("success") else np.inf for v in solvers]
                  for k in solvable])
    ratio = T / T.min(axis=1, keepdims=True)
    taus = np.logspace(0, 3, 200)
    for i, v in enumerate(solvers):
        axes[1].semilogx(taus, [(ratio[:, i] <= t).mean() for t in taus], color=color(v))
    axes[0].set(xlabel="number of obstacles", ylabel="success rate [%]", ylim=(0, 105),
                title="robustness")
    axes[1].set(xlabel="time ratio τ to fastest", ylabel="fraction solved", ylim=(0, 1.02),
                title="performance profile")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.1), fontsize=9.5)
    fig.tight_layout()
    save(fig, "slide_montecarlo")


def scaling():
    recs = load("e4_scaling.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for v in ["ipm", "sqp-exact", "scp", "alm-newton", "penalty-newton"]:
        rs = sorted([r for r in recs if r["solver"] == v and r["scenario"] == "slalom"],
                    key=lambda r: r["N"])
        n = [r["n"] for r in rs]
        it = [r["iters"] if r.get("success") else np.nan for r in rs]
        tpi = [r["time"] / max(r["iters"], 1) if r.get("success") else np.nan for r in rs]
        axes[0].semilogx(n, it, "o-", color=color(v), label=SHORT[v])
        axes[1].loglog(n, tpi, "o-", color=color(v))
    n = np.array([154, 1204])
    axes[1].loglog(n, 1.5e-2 * (n / 154) ** 3, "k:", lw=1.2, label="O(n³)")
    for ax in axes:
        ax.set_xticks([154, 304, 604, 1204])
        ax.set_xticklabels(["154", "304", "604", "1204"])
        ax.minorticks_off()
        ax.set_xlabel("number of variables n")
    axes[0].set(ylabel="iterations", yscale="log", title="iterations (slalom)")
    axes[1].set(ylabel="seconds / iteration", title="cost per iteration")
    axes[0].legend(fontsize=9.5)
    axes[1].legend()
    fig.tight_layout()
    save(fig, "slide_scaling")


if __name__ == "__main__":
    traj()
    convergence()
    conditioning()
    montecarlo()
    scaling()
    print("results dir:", RESULTS)
