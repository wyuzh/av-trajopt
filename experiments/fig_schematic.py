"""Schematic figures for the report and slides (no experiment data needed).

  figures/schematic_corridor.*  - why the collision constraint is nonconvex, and how a
                                  separating (supporting) hyperplane gives a convex
                                  inner approximation of the free space
  figures/schematic_scenarios.* - the three benchmark scenarios side by side
"""
import matplotlib.pyplot as plt
import numpy as np
from plots import draw_scene, save

from avtraj import scenarios


def corridor():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    o, R = np.array([0.0, 0.0]), 1.0

    ax = axes[0]
    ax.add_patch(plt.Circle(o, R, color="#bbbbbb", zorder=1))
    a, b = np.array([-1.6, 0.35]), np.array([1.6, 0.35])
    ax.plot(*np.c_[a, b], "k--", lw=1)
    ax.plot(*a, "o", color="#2ca02c", ms=7)
    ax.plot(*b, "o", color="#2ca02c", ms=7)
    ax.plot(*(0.5 * (a + b)), "x", color="#d62728", ms=9, mew=2)
    ax.text(0, 0.55, r"$\frac{1}{2}(p_a+p_b)$ collides", ha="center", color="#d62728", fontsize=9)
    ax.text(-1.6, 0.6, r"$p_a$ free", ha="center", fontsize=9)
    ax.text(1.6, 0.6, r"$p_b$ free", ha="center", fontsize=9)
    ax.text(0, -0.25, r"obstacle $\|p-o\|<R$", ha="center", fontsize=9)
    ax.set_title("free space $\\{p: \\|p-o\\|\\geq R\\}$ is not convex", fontsize=10)

    ax = axes[1]
    pref = np.array([0.45, 0.25])                 # reference point (inside the disc)
    n = (pref - o) / np.linalg.norm(pref - o)
    q = o + R * n                                 # supporting point on the boundary
    t = np.array([-n[1], n[0]])
    xs, ys = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1.6, 1.6, 300))
    inside = (n[0] * (xs - o[0]) + n[1] * (ys - o[1])) >= R
    ax.contourf(xs, ys, inside, levels=[0.5, 1.5], colors=["#c6dbef"], zorder=0)
    ax.add_patch(plt.Circle(o, R, color="#bbbbbb", zorder=1))
    ax.plot(*np.c_[q - 3 * t, q + 3 * t], color="#1f77b4", lw=1.5)
    ax.annotate("", q + 0.6 * n, q, arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
    ax.plot(*pref, "s", color="#ff7f0e", ms=6, zorder=3)
    ax.text(pref[0] - 0.05, pref[1] - 0.3, r"$\bar p_k$", fontsize=10, color="#ff7f0e")
    ax.text(*(q + 0.65 * n + np.array([0.05, 0.0])), r"$n$", color="#1f77b4", fontsize=10)
    ax.annotate(r"half-space $n^\top(p-o)\geq R$" + "\n(convex and collision-free)",
                xy=(1.6, -0.3), xytext=(-1.9, 1.2), fontsize=9, color="#08306b",
                arrowprops=dict(arrowstyle="->", color="#08306b", lw=0.8))
    ax.set_title("supporting hyperplane at the point nearest $\\bar p_k$", fontsize=10)
    for ax in axes:
        ax.set(xlim=(-2, 2), ylim=(-1.6, 1.6), aspect="equal", xticks=[], yticks=[])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    save(fig, "schematic_corridor")


def scenes():
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.6))
    for ax, name in zip(axes, ["single", "slalom", "narrow"]):
        sc = scenarios.get(name)
        draw_scene(ax, sc)
        ax.plot(0, 0, ">", color="#d62728", ms=9)
        ax.set_xlim(-1, 52)
        ax.set_title(f"'{name}': {sc.M} obstacle{'s' if sc.M > 1 else ''}", fontsize=9)
        if name != "narrow":
            ax.set_xlabel("")
    fig.tight_layout()
    save(fig, "schematic_scenarios")


if __name__ == "__main__":
    corridor()
    scenes()
