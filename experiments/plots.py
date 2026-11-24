"""Plotting helpers with a consistent style and colour per solver."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import FIGURES  # noqa: E402

LABELS = {
    "ipopt": "IPOPT (reference)",
    "ipm": "M1 Interior point",
    "sqp-exact": "M2 SQP (exact Hessian)",
    "sqp-bfgs": "M2 SQP (BFGS)",
    "sqp-shift": "M2 SQP (shifted Hessian)",
    "panoc-alm": "M3 PANOC-ALM",
    "ipm-riccati": "M4 Riccati interior point",
    "penalty-newton": "Penalty (Newton)",
    "penalty-lbfgs": "Penalty (L-BFGS)",
    "penalty-gd": "Penalty (GD)",
    "penalty-adam": "Penalty (Adam)",
    "alm-newton": "ALM (Newton)",
    "alm-lbfgs": "ALM (L-BFGS)",
    "scp": "SCP (convex corridor)",
    "corridor-qp": "Convex corridor QP",
}
COLORS = {
    "ipopt": "#7f7f7f", "penalty-newton": "#d62728", "penalty-lbfgs": "#ff9896",
    "penalty-gd": "#e377c2", "penalty-adam": "#bcbd22", "ipm": "#1f77b4",
    "alm-newton": "#2ca02c", "alm-lbfgs": "#98df8a", "sqp-exact": "#9467bd",
    "sqp-bfgs": "#c5b0d5", "sqp-shift": "#8c564b", "scp": "#ff7f0e", "corridor-qp": "#7f6000",
    "ipm-riccati": "#0b2a5b", "panoc-alm": "#17becf",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.3, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False,
})


def label(s):
    return LABELS.get(s, s)


def color(s):
    return COLORS.get(s, None)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print("saved", FIGURES / f"{name}.png")


def draw_scene(ax, sc, veh_radius=1.0):
    ax.axhspan(sc.road[0], sc.road[1], color="#f2f2f2", zorder=0)
    for yb in sc.road:
        ax.axhline(yb, color="k", lw=1.2)
    ax.axhline(2.0, color="k", lw=0.8, ls=(0, (6, 6)))        # lane divider
    for ox, oy, r in sc.obstacles:
        ax.add_patch(plt.Circle((ox, oy), r, color="#555555", alpha=0.85, zorder=2))
        ax.add_patch(plt.Circle((ox, oy), r + veh_radius, fill=False, ls="--", lw=0.7,
                                color="#555555", zorder=2))
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def floor(v, lo=1e-16):
    v = np.asarray(v, dtype=float)
    return np.maximum(v, lo)
