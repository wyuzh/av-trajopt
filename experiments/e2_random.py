"""E2 - Monte Carlo on random obstacle fields.

M in {2, 4, ..., 12} obstacles x 30 random seeds x 6 solvers, straight-line
initial guess.  An instance counts as *solvable* if any solver found a
feasible KKT point; success rates are reported over solvable instances, and
solution quality as the relative gap to the best objective found on that
instance (different solvers may converge to different local minima).

Produces results/e2_random.jsonl, results/e2_table.md and
figures/e2_success.*, figures/e2_profile.*, figures/e2_gap.*
"""
import argparse
from collections import defaultdict

import numpy as np
from common import RESULTS, load, run_all

MS = [2, 4, 6, 8, 10, 12]
SEEDS = range(30)
SOLVERS = ["ipopt", "penalty-newton", "ipm", "ipm-riccati", "alm-newton", "panoc-alm",
           "sqp-exact", "scp"]
OUT = "e2_random.jsonl"


def run():
    cases = [dict(scenario=f"random_M{m}_s{s}", solver=v, init="straight", M=m, seed=s)
             for m in MS for s in SEEDS for v in SOLVERS]
    run_all(cases, OUT)


def analyse(recs):
    inst = defaultdict(dict)
    for r in recs:
        inst[r["scenario"]][r["solver"]] = r
    best = {k: min((r["f"] for r in d.values() if r.get("success")), default=None)
            for k, d in inst.items()}
    solvable = {k for k, b in best.items() if b is not None}
    return inst, best, solvable


def plot():
    import matplotlib.pyplot as plt
    from plots import color, label, save

    recs = load(OUT)
    inst, best, solvable = analyse(recs)

    # ---- success rate and median time vs number of obstacles ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    rows = []
    for v in SOLVERS:
        sr, tmed, itmed = [], [], []
        for m in MS:
            keys = [k for k in solvable if inst[k][v]["M"] == m]
            ok = [inst[k][v] for k in keys if inst[k][v].get("success")]
            sr.append(len(ok) / max(len(keys), 1))
            tmed.append(np.median([r["time"] for r in ok]) if ok else np.nan)
            itmed.append(np.median([r["iters"] for r in ok]) if ok else np.nan)
        kw = dict(color=color(v), marker="o", ms=3, lw=1.3, label=label(v))
        axes[0].plot(MS, np.array(sr) * 100, **kw)
        axes[1].semilogy(MS, tmed, **kw)
        axes[2].plot(MS, itmed, **kw)
        allk = list(solvable)
        succ = [inst[k][v] for k in allk if inst[k][v].get("success")]
        gaps = [(inst[k][v]["f"] - best[k]) / max(abs(best[k]), 1e-9) for k in allk
                if inst[k][v].get("success")]
        rows.append((v, len(succ) / max(len(allk), 1), np.median([r["iters"] for r in succ]),
                     np.median([r["time"] for r in succ]),
                     np.mean(np.array(gaps) <= 1e-4), np.median(gaps)))
    axes[0].set(xlabel="number of obstacles M", ylabel="success rate [%]", ylim=(0, 105))
    axes[1].set(xlabel="number of obstacles M", ylabel="median time (successful) [s]")
    axes[2].set(xlabel="number of obstacles M", ylabel="median iterations (successful)")
    axes[0].legend(loc="lower left")
    n_inst = len({r["scenario"] for r in recs})
    fig.suptitle(f"Random obstacle fields: {len(solvable)}/{n_inst} instances solvable by at "
                 f"least one method", y=1.02)
    fig.tight_layout()
    save(fig, "e2_success")

    # ---- Dolan-More performance profile on wall time ----
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    keys = sorted(solvable)
    T = np.array([[inst[k][v]["time"] if inst[k][v].get("success") else np.inf for v in SOLVERS]
                  for k in keys])
    ratio = T / T.min(axis=1, keepdims=True)
    taus = np.logspace(0, 3, 200)
    for i, v in enumerate(SOLVERS):
        ax.semilogx(taus, [(ratio[:, i] <= t).mean() for t in taus], color=color(v), lw=1.4,
                    label=label(v))
    ax.set(xlabel=r"performance ratio $\tau$ (time / best time)",
           ylabel=r"fraction of instances solved within $\tau$", ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "e2_profile")

    # ---- objective gap to best local optimum found ----
    fig, ax = plt.subplots(figsize=(5, 3))
    data = []
    for v in SOLVERS:
        g = [(inst[k][v]["f"] - best[k]) / max(abs(best[k]), 1e-9) for k in keys
             if inst[k][v].get("success")]
        data.append(np.maximum(g, 1e-8))
    ax.boxplot(data, labels=[label(v).split(" (")[0] for v in SOLVERS], showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("relative gap to best objective found")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "e2_gap")

    lines = ["| solver | success rate | median iters | median time [s] | "
             "reaches best local opt. | median rel. gap |", "|---|---|---|---|---|---|"]
    for v, sr, it, t, best_frac, gap in rows:
        lines.append(f"| {v} | {sr:.1%} | {it:.0f} | {t:.2f} | {best_frac:.1%} | {gap:.1e} |")
    (RESULTS / "e2_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run()
    plot()
