"""E1 - Main benchmark.

Three scenarios x seven solvers x two initialisations.  Produces
  results/e1_benchmark.jsonl
  figures/e1_traj_<scenario>.{png,pdf}         optimal trajectories
  figures/e1_conv_<scenario>_<init>.{png,pdf}  violation / KKT / objective vs iteration and time
  results/e1_table.md                          summary table for the report

Usage:  python experiments/e1_benchmark.py [--plot-only]
"""
import argparse

import numpy as np
from common import RESULTS, load, run_all

from avtraj import scenarios, unpack

SCENARIOS = ["single", "slalom", "narrow"]
SOLVERS = ["ipopt", "penalty-newton", "ipm", "ipm-riccati", "alm-newton", "panoc-alm",
           "sqp-exact", "sqp-bfgs", "scp"]
INITS = ["straight", "corridor"]
OUT = "e1_benchmark.jsonl"


def run():
    cases = [dict(scenario=s, solver=v, init=i, keep_z=True)
             for s in SCENARIOS for v in SOLVERS for i in INITS]
    run_all(cases, OUT)


def plot():
    import matplotlib.pyplot as plt
    from plots import color, draw_scene, floor, label, save

    recs = load(OUT)
    by = {(r["scenario"], r["solver"], r["init"]): r for r in recs}

    # ---------------- trajectories ----------------
    for s in SCENARIOS:
        sc = scenarios.get(s)
        fig, ax = plt.subplots(figsize=(7.0, 2.4))
        draw_scene(ax, sc)
        for v in SOLVERS:
            r = by.get((s, v, "straight"))
            if not r or "z" not in r:
                continue
            X, _ = unpack(np.array(r["z"]), sc.N)
            ls = "-" if r["success"] else ":"
            ax.plot(X[:, 0], X[:, 1], ls, lw=1.4, color=color(v),
                    label=f"{label(v)}  f={r['f']:.3f}")
        ax.set_xlim(-1, 52)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
        ax.set_title(f"Scenario '{s}': optimal trajectories (initial guess: straight line)")
        save(fig, f"e1_traj_{s}")

    # ---------------- convergence curves ----------------
    for s in SCENARIOS:
        for init in INITS:
            fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
            for v in SOLVERS:
                r = by.get((s, v, init))
                if not r or not r.get("history"):
                    continue
                H = r["history"]
                k = np.arange(1, len(H) + 1)
                t = np.array([h["t"] for h in H])
                viol = floor([h["viol"] for h in H])
                kkt = floor([h["kkt"] for h in H])
                f = np.array([h["f"] for h in H])
                kw = dict(color=color(v), lw=1.2, label=label(v))
                axes[0].loglog(k, viol, **kw)
                axes[1].loglog(k, kkt, **kw)
                axes[2].loglog(k, floor(np.abs(f - f[-1]) / max(abs(f[-1]), 1e-12)), **kw)
                pos = t > 0
                axes[3].loglog(t[pos], np.maximum(viol, kkt)[pos], **kw)
            axes[0].set(xlabel="iteration (log scale)", ylabel="max constraint violation")
            axes[1].set(xlabel="iteration (log scale)", ylabel="scaled KKT residual")
            axes[2].set(xlabel="iteration (log scale)", ylabel=r"$|f_k - f_{final}| / |f_{final}|$")
            axes[3].set(xlabel="wall time [s]", ylabel="max(violation, KKT)")
            for ax in axes[:3]:
                ax.set_xlim(1, 2e4)
                ax.set_ylim(bottom=1e-12)
            axes[3].set_ylim(bottom=1e-12)
            h, l = axes[0].get_legend_handles_labels()
            fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.14))
            fig.suptitle(f"Convergence on '{s}' (initial guess: {init})", y=1.02)
            fig.tight_layout()
            save(fig, f"e1_conv_{s}_{init}")

    # ---------------- summary table ----------------
    lines = ["| scenario | init | solver | status | f | max viol | KKT | iters | time [s] | "
             "f-evals | Hess-evals | clearance [m] |", "|" + "---|" * 12]
    for s in SCENARIOS:
        for init in INITS:
            for v in SOLVERS:
                r = by.get((s, v, init))
                if not r:
                    continue
                c = r.get("counts", {})
                lines.append(
                    f"| {s} | {init} | {v} | {r['status']} | {r.get('f', np.nan):.4f} | "
                    f"{r.get('viol', np.nan):.1e} | {r.get('kkt', np.nan):.1e} | {r.get('iters', '')} | "
                    f"{r.get('time', np.nan):.2f} | {c.get('f', '')} | {c.get('hess', '')} | "
                    f"{r.get('clearance', np.nan):.3f} |")
    (RESULTS / "e1_table.md").write_text("\n".join(lines) + "\n")
    print("wrote", RESULTS / "e1_table.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run()
    plot()
