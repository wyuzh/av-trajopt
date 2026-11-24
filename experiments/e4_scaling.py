"""E4 - Scaling with the horizon length N (fixed horizon T = 5 s, so dt = T/N).

n = 6N + 4 decision variables.  Newton-type methods (SQP, interior point)
should need a roughly N-independent number of iterations, while the cost per
iteration grows with the (dense) linear algebra.

Produces results/e4_scaling.jsonl and figures/e4_scaling.*
"""
import argparse

import numpy as np
from common import load, run_all

NS = [25, 50, 100, 200, 400, 800]
SCEN = ["single", "slalom"]
SOLVERS = ["ipopt", "penalty-newton", "ipm", "ipm-riccati", "alm-newton", "panoc-alm",
           "sqp-exact", "scp"]
# dense O(n^3) linear algebra: too slow beyond N = 200 (n = 1204)
MAX_N = {"penalty-newton": 200, "ipm": 200, "alm-newton": 200, "sqp-exact": 200, "scp": 400}
OUT = "e4_scaling.jsonl"


def run():
    cases = [dict(scenario=s, solver=v, init="straight", N=N)
             for s in SCEN for v in SOLVERS for N in NS if N <= MAX_N.get(v, 10 ** 9)]
    run_all(cases, OUT)


def plot():
    import matplotlib.pyplot as plt
    from plots import color, label, save

    recs = load(OUT)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for s, ls in zip(SCEN, ["-", "--"]):
        for v in SOLVERS:
            rs = sorted([r for r in recs if r["solver"] == v and r["scenario"] == s],
                        key=lambda r: r["N"])
            n = np.array([r["n"] for r in rs if "n" in r])
            it = np.array([r["iters"] if r.get("success") else np.nan for r in rs if "n" in r])
            t = np.array([r["time"] if r.get("success") else np.nan for r in rs if "n" in r])
            kw = dict(color=color(v), ls=ls, marker="o", ms=3, lw=1.2,
                      label=label(v) if s == SCEN[0] else None)
            axes[0].semilogx(n, it, **kw)
            axes[1].loglog(n, t, **kw)
            axes[2].loglog(n, t / np.maximum(it, 1), **kw)
    n = np.array([154, 4804])
    axes[2].loglog(n, 2e-2 * (n / 154) ** 3, "k:", lw=0.9, label=r"$O(n^3)$")
    axes[2].loglog(n, 1.5e-2 * (n / 154), "k--", lw=0.9, label=r"$O(n)$")
    ticks = [6 * N + 4 for N in NS]
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t}\n(N={N})" for t, N in zip(ticks, NS)], fontsize=7)
        ax.minorticks_off()
    axes[0].set_yscale("log")
    axes[0].set(xlabel="number of variables n", ylabel="iterations")
    axes[1].set(xlabel="number of variables n", ylabel="wall time [s]")
    axes[2].set(xlabel="number of variables n", ylabel="time per iteration [s]")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.16),
               fontsize=7.5)
    rh, rl = axes[2].get_legend_handles_labels()
    axes[2].legend(rh[-2:], rl[-2:], fontsize=7, loc="lower right")
    fig.suptitle("Scaling with horizon length (solid: single, dashed: slalom)", y=1.02)
    fig.tight_layout()
    save(fig, "e4_scaling")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run()
    plot()
