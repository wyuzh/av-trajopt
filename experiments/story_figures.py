"""Story figures and animations for the presentation (plain-language, big and bold).

    python experiments/story_figures.py prep    # solve the few cases whose iterates we need
    python experiments/story_figures.py draw    # write figures/story_*.png and *.gif

`prep` caches to results/story_cache.pkl; `draw` also reads the saved E1-E4 results.
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.font_manager

matplotlib.use("Agg")
import casadi as ca  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

from common import FIGURES, RESULTS, initial_guess, load  # noqa: E402

from avtraj import build_nlp, metrics, scenarios, unpack  # noqa: E402
from avtraj.problem import NLP  # noqa: E402
from avtraj.solvers import SOLVERS  # noqa: E402

CACHE = RESULTS / "story_cache.pkl"

# one colour per idea, used on every slide
C_FINE, C_FIELD, C_PRICE, C_SIMPLE = "#E4572E", "#2E86DE", "#17A673", "#7B3FA0"
C_REF, C_SCP = "#8A8F98", "#F5A623"
C_PANOC, C_RIC = "#0E9AA7", "#1B3A6B"            # M3 PANOC-ALM, M4 Riccati interior point
ASPHALT, PAINT, CONE, CAR = "#3B4048", "#F4F4F4", "#FF7A3D", "#4FC3F7"
INK = "#2B2B2B"
METHODS = [("penalty-newton", "Fines", C_FINE), ("ipm", "Force field", C_FIELD),
           ("alm-newton", "Smart prices", C_PRICE), ("sqp-exact", "Simplify & repeat", C_SIMPLE)]

for _f in Path.home().glob(".local/share/fonts/msttcorefonts/*.[Tt][Tt][Ff]"):
    matplotlib.font_manager.fontManager.addfont(str(_f))      # Arial, to match the slides
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "figure.dpi": 150,
                     "axes.titlesize": 17, "axes.labelsize": 15, "axes.titleweight": "bold"})


def save(fig, name, dpi=200):
    fig.savefig(FIGURES / f"{name}.png", dpi=dpi)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------------------
# road-scene drawing
# ---------------------------------------------------------------------------
def road(ax, sc, xlim=(-2, 53), margin=True):
    ax.add_patch(Rectangle((xlim[0], sc.road[0]), xlim[1] - xlim[0], sc.road[1] - sc.road[0],
                           color=ASPHALT, zorder=0))
    for yb in sc.road:
        ax.plot(xlim, [yb, yb], color=PAINT, lw=3, zorder=1)
    ax.plot(xlim, [2, 2], color=PAINT, lw=2, ls=(0, (8, 8)), zorder=1)
    for ox, oy, r in sc.obstacles:
        if margin:
            ax.add_patch(Circle((ox, oy), r + 1.0, fill=False, ls=(0, (3, 3)), lw=1.2,
                                color="#FFD9C7", zorder=2))
        ax.add_patch(Circle((ox, oy), r, color=CONE, ec="white", lw=1.5, zorder=3))
    ax.set_xlim(*xlim)
    ax.set_ylim(sc.road[0] - 0.7, sc.road[1] + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")


def car(ax, x, y, th, color=CAR, alpha=1.0, zorder=6, length=3.4, width=1.6):
    tr = Affine2D().rotate(th).translate(x, y) + ax.transData
    body = FancyBboxPatch((-length / 2, -width / 2), length, width,
                          boxstyle="round,pad=0,rounding_size=0.45", fc=color, ec="white", lw=1.2,
                          alpha=alpha, zorder=zorder, transform=tr)
    glass = Rectangle((0.25, -width / 2 + 0.22), 0.75, width - 0.44, fc="#12324A", alpha=alpha,
                      zorder=zorder + 1, transform=tr)
    ax.add_patch(body)
    ax.add_patch(glass)
    return body, glass


# ---------------------------------------------------------------------------
# prep: solve the handful of cases whose iterates / trajectories we need
# ---------------------------------------------------------------------------
def toy_nlp(z0):
    """min (x-2)^2 + (y-1)^2  s.t.  x^2 + y^2 <= 1  (a dummy equality keeps shapes regular)."""
    x = ca.SX.sym("x", 3)
    f = (x[0] - 2) ** 2 + (x[1] - 1) ** 2 + x[2] ** 2
    return NLP(x, f, x[2], x[0] ** 2 + x[1] ** 2 - 1, z0=list(z0) + [0.0], name="toy")


def _capture():
    """Make History.log remember the iterate, so we can replay how a solver 'thinks'."""
    orig = metrics.History.log

    def log(self, z, *a, **k):
        rec = orig(self, z, *a, **k)
        rec["z"] = np.array(z, dtype=float)
        return rec

    metrics.History.log = log


def prep():
    _capture()
    out = {}
    # how each solver's trajectory evolves on the slalom
    nlp = build_nlp(scenarios.get("slalom"))
    for name in ["scp", "ipm", "sqp-exact", "alm-newton", "penalty-newton"]:
        res = SOLVERS[name](nlp)
        out[f"think/{name}"] = dict(Z=np.array([h["z"] for h in res.history]), f=res.f,
                                    ok=res.success)
        print("think", name, res.status, len(res.history))
    # the four ideas on the 2-D toy problem
    for name in ["penalty-newton", "ipm", "alm-newton", "sqp-exact"]:
        res = SOLVERS[name](toy_nlp([-0.3, -0.6]))
        out[f"toy/{name}"] = dict(Z=np.array([h["z"][:2] for h in res.history]),
                                  outer=[h.get("outer", 0) for h in res.history], ok=res.success)
        print("toy", name, res.status, len(res.history), res.z[:2])
    # trap: same solver, two starting guesses
    nlp = build_nlp(scenarios.get("trap"))
    for init in ["straight", "offset:3.0"]:
        res = SOLVERS["ipm"](nlp, initial_guess(nlp, init))
        out[f"trap/{init}"] = dict(z=res.z, status=res.status, f=res.f)
        print("trap", init, res.status)
    # a gallery of random obstacle fields solved by the augmented Lagrangian method
    for m, s in [(4, 3), (4, 4), (6, 9), (6, 21), (8, 16), (8, 21), (10, 6), (12, 18)]:
        sc = scenarios.get(f"random_M{m}_s{s}")
        res = SOLVERS["alm-newton"](build_nlp(sc))
        out[f"gallery/{m}_{s}"] = dict(z=res.z, ok=res.success)
        print("gallery", m, s, res.status)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_drive(e1):
    """The hero: the car driving the optimised slalom (GIF + still)."""
    sc = scenarios.get("slalom")
    X, _ = unpack(np.array(e1[("slalom", "ipm", "straight")]["z"]), sc.N)
    t = np.linspace(0, sc.N, 4 * sc.N + 1)
    xs = np.interp(t, np.arange(sc.N + 1), X[:, 0])
    ys = np.interp(t, np.arange(sc.N + 1), X[:, 1])
    th = np.interp(t, np.arange(sc.N + 1), X[:, 2])

    def frame(i, ax):
        ax.clear()
        road(ax, sc)
        ax.plot(xs[: i + 1], ys[: i + 1], color=CAR, lw=3.5, zorder=4, solid_capstyle="round")
        car(ax, xs[i], ys[i], th[i])

    fig, ax = plt.subplots(figsize=(10, 2.4))
    fig.subplots_adjust(0, 0, 1, 1)
    frames = list(range(0, len(t), 2)) + [len(t) - 1] * 12
    anim = animation.FuncAnimation(fig, lambda i: frame(i, ax), frames=frames)
    anim.save(FIGURES / "story_drive.gif", writer=animation.PillowWriter(fps=20), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 2.4))
    road(ax, sc)
    ax.plot(xs, ys, color=CAR, lw=3.5, zorder=4)
    for i in (0, 60, 100, 140, 180):
        car(ax, xs[i], ys[i], th[i], alpha=0.55 if i < 180 else 1.0)
    save(fig, "story_drive")


def fig_problem(e1):
    """Lazy plan (straight, crashes) vs the plan we want."""
    sc = scenarios.get("single")
    X, _ = unpack(np.array(e1[("single", "ipm", "straight")]["z"]), sc.N)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    road(ax, sc)
    ax.plot([0, 50], [0, 0], color="#FF5A5A", lw=3, ls=(0, (4, 3)), zorder=4)
    ax.plot(X[:, 0], X[:, 1], color=CAR, lw=3.5, zorder=5)
    car(ax, 0, 0, 0)
    ax.text(20, -0.3, "✕", color="white", fontsize=22, ha="center", va="center", zorder=9,
            fontweight="bold", fontfamily="DejaVu Sans")
    ax.text(27, -1.25, "initial guess: straight line (infeasible)", color="#FF8A8A", fontsize=13, zorder=9)
    ax.text(24.5, 3.1, "optimal trajectory", color=CAR, fontsize=13, zorder=9, fontweight="bold")
    save(fig, "story_problem")


def fig_hole():
    """Why it is hard: the average of two safe plans is a crash."""
    sc = scenarios.Scenario("hole", np.array([[26.0, 2.0, 1.6]]))
    fig, ax = plt.subplots(figsize=(10, 2.9))
    road(ax, sc, margin=False)
    xx = np.linspace(8, 44, 200)
    bump = 3.1 * np.exp(-((xx - 26) / 6.0) ** 2)
    ax.plot(xx, 2 + bump, color="#7CE38B", lw=3.5, zorder=4)
    ax.plot(xx, 2 - bump, color="#7CE38B", lw=3.5, zorder=4)
    ax.plot(xx, 2 + 0 * xx, color="#FF5A5A", lw=3, ls=(0, (4, 3)), zorder=4)
    car(ax, 26, 5.1, 0, color="#7CE38B")
    car(ax, 26, -1.1, 0, color="#7CE38B")
    ax.text(31, 5.0, "feasible (pass left)", color="#7CE38B", fontsize=14, fontweight="bold", va="center",
            fontfamily="DejaVu Sans")
    ax.text(31, -1.1, "feasible (pass right)", color="#7CE38B", fontsize=14, fontweight="bold",
            va="center", fontfamily="DejaVu Sans")
    ax.text(-1.2, 1.6, "average of the two:\ninfeasible", color="#FF8A8A", fontsize=12, fontweight="bold", va="top",
            fontfamily="DejaVu Sans")
    save(fig, "story_hole")


def fig_think(cache, name="alm-newton"):
    """Watch the optimiser think: the plan bending step by step (GIF + still)."""
    sc = scenarios.get("slalom")
    P = [unpack(z, sc.N)[0] for z in cache[f"think/{name}"]["Z"]]
    last = len(P) - 1

    def still(ax, upto):
        road(ax, sc)
        for i in range(0, upto):
            ax.plot(P[i][:, 0], P[i][:, 1], color="#BFE9FF", lw=1.3, alpha=0.5, zorder=4)
        ax.plot(P[upto][:, 0], P[upto][:, 1], color=CAR if upto else "#FF5A5A", lw=3.5, zorder=5)

    fig, ax = plt.subplots(figsize=(10, 2.4))
    fig.subplots_adjust(0, 0, 1, 1)

    def frame(i):
        ax.clear()
        still(ax, i)
        ax.text(0.012, 0.955, "initial guess" if i == 0 else f"iteration {i}", transform=ax.transAxes,
                color=INK, fontsize=15, fontweight="bold", va="top", zorder=10,     # label on a white tag,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#9AA0A6", lw=1.0))  # readable on the road

    seq = [0] * 8 + list(range(last + 1)) + [last] * 16
    anim = animation.FuncAnimation(fig, frame, frames=[k for k in seq for _ in range(2)])
    anim.save(FIGURES / "story_think.gif", writer=animation.PillowWriter(fps=10), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 2.4))
    still(ax, last)
    ax.plot(P[0][:, 0], P[0][:, 1], color="#FF5A5A", lw=2.5, ls=(0, (4, 3)), zorder=5)
    save(fig, "story_think")


def fig_fines_road(cache):
    """The penalty method on the road: small fine = drive over the cone and pay."""
    sc = scenarios.get("slalom")
    P = [unpack(z, sc.N)[0] for z in cache["think/penalty-newton"]["Z"]]
    rows = [(2, r"small $\rho$:  constraint violated", 0.5),
            (9, r"larger $\rho$:  violation shrinks", 0.75),
            (len(P) - 1, r"$\rho \to \infty$:  feasible", 1.0)]
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.4))
    for ax, (i, lab, a) in zip(axes, rows):
        road(ax, sc, xlim=(-2, 36))
        ax.plot(P[i][:, 0], P[i][:, 1], color="#FFB199" if a < 1 else "#7CE38B", lw=3.5, zorder=5)
        k = int(np.argmin(np.abs(P[i][:, 0] - 15)))
        car(ax, P[i][k, 0], P[i][k, 1], P[i][k, 2], color="#FFB199" if a < 1 else "#7CE38B")
        ax.set_title(lab, color=C_FINE if a < 1 else "#2E7D32", loc="left", fontsize=15)
    fig.subplots_adjust(hspace=0.35)
    save(fig, "story_fines_road")


RACE = [("sqp-exact", "M2 SQP", C_SIMPLE), ("ipm", "M1 Interior point", C_FIELD),
        ("alm-newton", "ALM, Newton (baseline)", C_PRICE),
        ("penalty-newton", "Quadratic penalty (baseline)", C_FINE)]


def fig_race_gif(cache):
    """The race: four methods improve the same plan side by side, one step per frame."""
    sc = scenarios.get("slalom")
    paths = {k: [unpack(z, sc.N)[0] for z in cache[f"think/{k}"]["Z"]] for k, _, _ in RACE}
    longest = max(len(v) for v in paths.values()) - 1

    def frame(i, axes):
        for ax, (k, lab, col) in zip(axes, RACE):
            ax.clear()
            P = paths[k]
            j = min(i, len(P) - 1)
            done = i >= len(P) - 1
            road(ax, sc, margin=False)
            ax.plot(P[j][:, 0], P[j][:, 1], color="white" if not done else "#7CE38B", lw=3, zorder=5)
            ax.text(0.0, 1.04, lab, transform=ax.transAxes, color=col, fontsize=15,
                    fontweight="bold", va="bottom")
            ax.text(1.0, 1.04, f"converged: {len(P) - 1} iterations" if done else f"iteration {j}",
                    transform=ax.transAxes, color="#2E7D32" if done else INK, fontsize=14,
                    fontweight="bold", va="bottom", ha="right")

    fig, axes = plt.subplots(4, 1, figsize=(9, 6.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.01, hspace=0.42)
    seq = [0] * 6 + list(range(longest + 1)) + [longest] * 20
    anim = animation.FuncAnimation(fig, lambda i: frame(i, axes), frames=seq)
    anim.save(FIGURES / "story_race.gif", writer=animation.PillowWriter(fps=7), dpi=100)
    frame(longest, axes)
    save(fig, "story_race_still")


def _landscape(ax, title, color):
    x = np.linspace(-0.6, 2.7, 600)
    ax.axvspan(1, 2.7, color="#FDE3DC", zorder=0)
    ax.axvline(1, color=INK, lw=3)
    ax.plot(x, (x - 2) ** 2, color="#9AA0A6", lw=2.5, ls="--")
    ax.plot(2, 0, marker="*", color="#F5B700", ms=22, mec=INK, zorder=6)
    ax.text(2.0, -0.52, "unconstrained minimiser", fontsize=12, ha="center", color=INK)
    ax.text(1.85, 3.55, "infeasible", fontsize=13, ha="center", color="#C0392B",
            fontweight="bold")
    ax.text(0.2, -0.52, "feasible", fontsize=13, ha="center", color="#2E7D32", fontweight="bold")
    ax.set(xlim=(-0.6, 2.7), ylim=(-0.75, 4), xticks=[], yticks=[], ylabel="cost")
    ax.set_title(title, color=color, loc="left")
    return x


def fig_landscapes():
    """Three 1-D pictures: fines, force field, smart prices.  Problem: min (x-2)^2 s.t. x <= 1."""
    # fines
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = _landscape(ax, r"Quadratic penalty:  $f + \frac{\rho}{2}\max(0, g)^2$", C_FINE)
    for rho, lab, a in [(2, r"small $\rho$", 0.45), (10, "bigger fine", 0.7), (80, "huge fine", 1.0)]:
        y = (x - 2) ** 2 + 0.5 * rho * np.maximum(x - 1, 0) ** 2
        xs = (4 + rho) / (2 + rho)
        ax.plot(x, y, color=C_FINE, lw=3, alpha=a)
        ax.plot(xs, (xs - 2) ** 2 + 0.5 * rho * (xs - 1) ** 2, "o", color=C_FINE, ms=11, mec="white",
                zorder=7, alpha=a)
    ax.annotate("minimiser is always\nslightly infeasible", xy=(1.14, 0.8), xytext=(-0.5, 0.35), fontsize=13,
                color=C_FINE, fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_FINE, lw=2))
    ax.text(2.02, 1.02, r"small $\rho$", fontsize=12, color=C_FINE, alpha=0.75)
    ax.text(1.3, 2.85, r"large $\rho$", fontsize=12, color=C_FINE, fontweight="bold")
    save(fig, "story_fines")

    # force field
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = _landscape(ax, r"Log barrier:  $f - \mu \log(-g)$", C_FIELD)
    xi = np.linspace(-0.6, 0.9995, 2000)
    for mu, a in [(1.2, 0.45), (0.3, 0.7), (0.03, 1.0)]:
        y = (xi - 2) ** 2 - mu * np.log(1 - xi)
        ax.plot(xi, y, color=C_FIELD, lw=3, alpha=a)
        k = np.argmin(y)
        ax.plot(xi[k], y[k], "o", color=C_FIELD, ms=11, mec="white", zorder=7, alpha=a)
    ax.annotate("strictly feasible,\n" + r"$\to$ boundary as $\mu \to 0$", xy=(0.95, 1.18), xytext=(-0.5, 0.35), fontsize=13,
                color=C_FIELD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_FIELD, lw=2))
    save(fig, "story_field")

    # smart prices (augmented Lagrangian, PHR form, rho fixed)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = _landscape(ax, r"Augmented Lagrangian, fixed $\rho$", C_PRICE)
    rho, lam = 4.0, 0.0
    for k, a in enumerate([0.35, 0.55, 0.8, 1.0]):
        y = (x - 2) ** 2 + (np.maximum(lam + rho * (x - 1), 0) ** 2 - lam ** 2) / (2 * rho)
        i = np.argmin(y)
        ax.plot(x, y, color=C_PRICE, lw=3, alpha=a)
        ax.plot(x[i], y[i], "o", color=C_PRICE, ms=11, mec="white", zorder=7, alpha=a)
        lam = max(0.0, lam + rho * (x[i] - 1))
    ax.annotate("multiplier updates reach\nthe boundary, " + r"$\rho$ fixed", xy=(0.98, 0.98), xytext=(-0.5, 0.2),
                fontsize=13, color=C_PRICE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PRICE, lw=2))
    save(fig, "story_prices")


def _toy_axes(ax):
    xs, ys = np.meshgrid(np.linspace(-1.5, 2.5, 300), np.linspace(-1.5, 1.9, 300))
    ax.contour(xs, ys, (xs - 2) ** 2 + (ys - 1) ** 2, levels=np.arange(0.25, 12, 0.9) ** 1.0,
               colors="#C9CED6", linewidths=1.0, zorder=1)
    ax.add_patch(Circle((0, 0), 1, color="#DDF3E4", ec=INK, lw=2.5, zorder=2))
    ax.plot(2, 1, marker="*", color="#F5B700", ms=24, mec=INK, zorder=6)
    opt = np.array([2, 1]) / np.sqrt(5)
    ax.plot(*opt, marker="o", color="white", ms=13, mec=INK, mew=2.5, zorder=7)
    ax.set(xlim=(-1.5, 2.5), ylim=(-1.5, 1.9), aspect="equal", xticks=[], yticks=[])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_simplify(cache):
    """SQP on the toy problem: replace the curved fence by a straight one, step, repeat."""
    Z = cache["toy/sqp-exact"]["Z"]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    _toy_axes(ax)
    for k, z in enumerate(Z[:3]):
        if k == 0:
            continue
        g, gr = z @ z - 1, 2 * z
        if np.linalg.norm(gr) < 1e-9:
            continue
        nrm = gr / np.linalg.norm(gr)
        p0 = z - g / np.linalg.norm(gr) * nrm          # point on the linearised boundary
        tdir = np.array([-nrm[1], nrm[0]])
        ax.plot(*np.c_[p0 - 3 * tdir, p0 + 3 * tdir], color=C_SIMPLE, lw=1.8, ls="--",
                alpha=0.35 + 0.25 * k, zorder=3)
    ax.plot(Z[:, 0], Z[:, 1], "-o", color=C_SIMPLE, lw=3, ms=9, mec="white", zorder=8)
    for k, z in enumerate(Z[:3]):
        ax.text(z[0] - 0.12, z[1] + 0.15, str(k), color=C_SIMPLE, fontsize=13, fontweight="bold",
                ha="right", zorder=9)
    ax.text(-1.4, 1.62, "linearised constraint\nat each iterate", color=C_SIMPLE, fontsize=13,
            fontweight="bold", va="top")
    ax.set_title("SQP: quadratic model, linearised constraint", color=C_SIMPLE, loc="left")
    save(fig, "story_simplify")


def fig_canyon():
    """Small fine = wide bowl; huge fine = razor-thin canyon where simple downhill walking zig-zags."""
    from scipy.optimize import minimize_scalar

    def P(x, y, rho):
        return (x - 2) ** 2 + (y - 1) ** 2 + 0.5 * rho * np.maximum(x ** 2 + y ** 2 - 1, 0) ** 2

    def grad(p, rho):
        c = max(p @ p - 1, 0)
        return np.array([2 * (p[0] - 2), 2 * (p[1] - 1)]) + 2 * rho * c * p

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    xs, ys = np.meshgrid(np.linspace(-0.4, 2.3, 500), np.linspace(-0.6, 1.6, 500))
    for ax, rho, title in zip(axes, [1, 200], [r"$\rho = 1$:  well conditioned",
                                                r"$\rho = 200$:  ill-conditioned valley"]):
        Z = P(xs, ys, rho)
        ax.contourf(xs, ys, np.log(Z - Z.min() + 3e-3), 28, cmap="magma_r")
        ax.add_patch(Circle((0, 0), 1, fill=False, ec="white", lw=2, ls="--"))
        p, path = np.array([0.15, 1.35]), []
        for _ in range(60):
            path.append(p.copy())
            d = -grad(p, rho)
            if np.linalg.norm(d) < 1e-9:
                break
            t = minimize_scalar(lambda a: P(*(p + a * d), rho), bounds=(0, 5), method="bounded",
                                options={"xatol": 1e-12}).x
            p = p + t * d
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "-o", color="#4FC3F7", lw=2, ms=4, mec="white", mew=0.5)
        ax.plot(*(np.array([2, 1]) / np.sqrt(5)), marker="*", color="#F5B700", ms=20, mec=INK)
        ax.set(xticks=[], yticks=[], aspect="equal", xlim=(-0.4, 2.3), ylim=(-0.6, 1.6))
        ax.set_title(title, color=C_FINE)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    save(fig, "story_canyon")


def _hbar(ax, labels, values, colors, fmt="{:.0f}", xlabel=None, xlim=None, text=None, log=False):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.66, zorder=3)
    for yi, v, tx in zip(y, values, text or [fmt.format(v) for v in values]):
        ax.text(v * (1.08 if log else 1) + (0 if log else 0.012 * (xlim or max(values))), yi, tx,
                va="center", fontsize=16, fontweight="bold", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    if log:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(ax.get_xlim()[0] if log else 0, xlim)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25, zorder=0)


def fig_steps_canyon(e3):
    """How many steps to solve ONE sub-problem when the fine is huge (rho = 1e6)."""
    it = {r["solver"]: r["iters"] for r in e3 if r.get("rho") == 1e6 and r["solver"].startswith("penalty")}
    names = [("penalty-gd", "Gradient descent"), ("penalty-adam", "Adam"),
             ("penalty-lbfgs", "L-BFGS"), ("penalty-newton", "Newton")]
    vals = [it[k] for k, _ in names]
    txt = [("> 20,000 (cap)" if v >= 20000 else f"{v:,}") for v in vals]
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    _hbar(ax, [n for _, n in names], vals, ["#F2A08B", "#F2A08B", "#EB7C5E", C_FINE], text=txt,
          xlabel="inner iterations (log scale)", log=True, xlim=6e5)
    save(fig, "story_steps_canyon")


def fig_race(e1):
    """The race: average number of steps over the three scenarios (fewer is better)."""
    rows = [("sqp-exact", "SQP", C_SIMPLE), ("scp", "SCP (convex corridor)", C_SCP),
            ("ipopt", "IPOPT (reference)", C_REF), ("ipm", "Interior point", C_FIELD),
            ("alm-newton", "Augmented Lagrangian", C_PRICE),
            ("penalty-newton", "Quadratic penalty", C_FINE)]
    vals = [np.mean([e1[(s, k, "straight")]["iters"] for s in ("single", "slalom", "narrow")])
            for k, _, _ in rows]
    fig, ax = plt.subplots(figsize=(9.4, 4.1))
    _hbar(ax, [n for _, n, _ in rows], vals, [c for _, _, c in rows],
          xlabel="iterations to convergence (mean of 3 scenarios)", xlim=52)
    save(fig, "story_race")
    return dict(zip([k for k, _, _ in rows], vals))


def fig_robust(e2):
    inst = defaultdict(dict)
    for r in e2:
        inst[r["scenario"]][r["solver"]] = r
    solv = [k for k, d in inst.items() if any(x.get("success") for x in d.values())]
    rows = [("alm-newton", "ALM, Newton (baseline)", C_PRICE), ("sqp-exact", "M2 SQP", C_SIMPLE),
            ("ipopt", "IPOPT (reference)", C_REF), ("ipm-riccati", "M4 Riccati IP", C_RIC),
            ("panoc-alm", "M3 PANOC-ALM", C_PANOC), ("scp", "SCP (baseline)", C_SCP),
            ("ipm", "M1 Interior point", C_FIELD), ("penalty-newton", "Penalty (baseline)", C_FINE)]
    vals = [100 * np.mean([inst[k][v].get("success", False) for k in solv]) for v, _, _ in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _hbar(ax, [n for _, n, _ in rows], vals, [c for _, _, c in rows], fmt="{:.0f}%",
          xlabel="success rate on solvable instances", xlim=118)
    ax.set_xticks([0, 25, 50, 75, 100])
    save(fig, "story_robust")


def fig_gallery(cache):
    keys = [k for k in cache if k.startswith("gallery/") and cache[k]["ok"]][:8]
    fig, axes = plt.subplots(4, 2, figsize=(11, 6.4))
    for ax, k in zip(axes.ravel(), keys):
        m, s = k.split("/")[1].split("_")
        sc = scenarios.get(f"random_M{m}_s{s}")
        X, _ = unpack(cache[k]["z"], sc.N)
        road(ax, sc, margin=False)
        ax.plot(X[:, 0], X[:, 1], color="#7CE38B", lw=3, zorder=5)
        car(ax, X[-1, 0], X[-1, 1], X[-1, 2], color="#7CE38B", length=3.0, width=1.5)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.subplots_adjust(hspace=0.08, wspace=0.04)
    save(fig, "story_gallery")


def fig_pricetags(e1):
    """Each obstacle's price tag = how much the best plan would cost extra if it grew by 1 m."""
    sc = scenarios.get("slalom")
    X, _ = unpack(np.array(e1[("slalom", "ipm", "straight")]["z"]), sc.N)
    tags = ["+3.9", "0", "+4.7"]                       # results/e6_sensitivity.md
    fig, ax = plt.subplots(figsize=(10, 3.2))
    road(ax, sc)
    ax.set_ylim(sc.road[0] - 0.7, sc.road[1] + 3.3)
    ax.plot(X[:, 0], X[:, 1], color=CAR, lw=3.5, zorder=4)
    car(ax, X[-1, 0], X[-1, 1], X[-1, 2])
    for (ox, oy, r), tag in zip(sc.obstacles, tags):
        free = tag == "0"
        ax.annotate(f"{tag}" if not free else "0  (inactive)", xy=(ox, oy + r), xytext=(ox, 8.2),
                    ha="center", fontsize=17, fontweight="bold", color="white", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#8A8F98" if free else C_PRICE, ec="none"),
                    arrowprops=dict(arrowstyle="-", color="#8A8F98" if free else C_PRICE, lw=2.5))
    save(fig, "story_pricetags")


def fig_trap(cache):
    sc = scenarios.get("trap")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.2))
    rows = [(axes[0], "trap/straight", "#FF5A5A",
             "M1 interior point, lane-centre guess  →  locally infeasible point", "#C0392B"),
            (axes[1], "trap/panoc/straight", "#7CE38B",
             f"M3 PANOC-ALM, same guess  →  converged (f* = {cache['trap/panoc/straight']['f']:.2f})",
             "#2E7D32"),
            (axes[2], "trap/offset:3.0", "#7CE38B",
             f"M1 interior point, guess shifted 3 m  →  converged (f* = {cache['trap/offset:3.0']['f']:.2f})",
             "#2E7D32")]
    for ax, key, col, lab, tcol in rows:
        X, _ = unpack(cache[key]["z"], sc.N)
        road(ax, sc)
        ax.plot(X[:, 0], X[:, 1], color=col, lw=3.5, zorder=5)
        car(ax, X[-1, 0], X[-1, 1], X[-1, 2], color=col)
        if key == "trap/straight":                  # mark where the plan is still too close
            d = np.linalg.norm(X[1:, None, :2] - sc.obstacles[None, :, :2], axis=2) \
                - sc.obstacles[None, :, 2] - 1.0
            k, _ = np.unravel_index(np.argmin(d), d.shape)
            ax.text(X[k + 1, 0], X[k + 1, 1], "✕", color="white", fontsize=22, ha="center",
                    va="center", fontweight="bold", zorder=9, fontfamily="DejaVu Sans")
        ax.set_title(lab, color=tcol, loc="left", fontsize=15, fontfamily="DejaVu Sans")
    fig.subplots_adjust(hspace=0.32)
    save(fig, "story_trap")


def fig_scaling(e4):
    ns = [25, 50, 100, 200]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    w = 0.36
    for off, (k, lab, col) in zip((-w / 2, w / 2), [("ipm", "M1 Interior point", C_FIELD),
                                                    ("sqp-exact", "M2 SQP", C_SIMPLE)]):
        v = [next(r["iters"] for r in e4 if r["solver"] == k and r["scenario"] == "slalom"
                  and r["N"] == n) for n in ns]
        b = ax.bar(np.arange(4) + off, v, w, color=col, label=lab, zorder=3)
        ax.bar_label(b, fontsize=14, fontweight="bold", padding=2)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"n = {6 * n + 4:,}" for n in ns])
    ax.set(ylabel="iterations", ylim=(0, 42))
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    save(fig, "story_scaling")


# ---------------------------------------------------------------------------
# figures for the modern methods (M3 PANOC-ALM, M4 Riccati interior point)
# ---------------------------------------------------------------------------
def prep_sota():
    """Add the cases needed for the M3/M4 slides to the existing cache."""
    cache = pickle.load(open(CACHE, "rb"))
    nlp = build_nlp(scenarios.get("trap"))
    res = SOLVERS["panoc-alm"](nlp, initial_guess(nlp, "straight"))
    cache["trap/panoc/straight"] = dict(z=res.z, status=res.status, f=res.f)
    print("trap panoc", res.status, res.f)
    with open(CACHE, "wb") as fh:
        pickle.dump(cache, fh)


def fig_panoc(e1):
    """M3 vs its classical counterpart: many cheap iterations vs few expensive ones."""
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    for k, lab, col, ls in [("panoc-alm", "M3 PANOC-ALM", C_PANOC, "-"),
                            ("alm-newton", "ALM, Newton", C_FINE, "--")]:
        r = e1[("slalom", k, "straight")]
        H = r["history"]
        res = np.maximum([max(h["viol"], h["kkt"]) for h in H], 1e-12)
        c = r["counts"]
        ax.loglog(np.arange(1, len(H) + 1), res, color=col, lw=2.6, ls=ls,
                  label=f"{lab}:  {r['iters']:,} iterations, {c.get('hess', 0)} Hessians, "
                        f"{r['time']:.1f} s")
    ax.axhline(1e-6, color=INK, lw=1, ls=":")
    ax.text(1.2, 2e-6, "tolerance 10⁻⁶", fontsize=13, color=INK)
    ax.set(xlabel="iteration (log scale)", ylabel="max(violation, KKT)", ylim=(1e-9, 1e6),
           xlim=(1, 3e4))
    leg = ax.legend(frameon=False, loc="upper left", fontsize=14)
    for t, (_, _, col) in zip(leg.get_texts(), [(0, 0, C_PANOC), (0, 0, C_FINE)]):
        t.set_color(col)
        t.set_fontweight("bold")
    ax.grid(alpha=0.25)
    save(fig, "story_panoc")


def fig_kkt_spy():
    """Sparsity of the interior-point KKT matrix: natural vs stage-wise ordering."""
    sc = scenarios.slalom(N=8)
    nlp = build_nlp(sc)
    rng = np.random.default_rng(0)
    z = nlp.z0 + 0.1 * rng.standard_normal(nlp.n)
    H = nlp.hess_lag_sp(z, rng.standard_normal(nlp.m_eq), rng.random(nlp.m_in))
    Jg = nlp.jac_g_sp(z)
    import scipy.sparse as sp
    H = (H + Jg.T @ Jg).tocsr()
    Jh = nlp.jac_h_sp(z).tocsr()
    K = sp.bmat([[H, Jh.T], [Jh, None]]).tocsr()
    N, nx, nu = sc.N, 4, 2
    perm = []                                         # (lam_k, x_k, u_k) stage by stage
    for k in range(N + 1):
        perm += list(nlp.n + np.arange(nx * k, nx * (k + 1)))
        perm += list(range(nx * k, nx * (k + 1)))
        if k < N:
            perm += list(nx * (N + 1) + np.arange(nu * k, nu * (k + 1)))
    Kp = K[perm][:, perm]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.0))
    for ax, M, t in [(axes[0], K, "ordering (x, u, λ)"), (axes[1], Kp, "stage-wise ordering")]:
        ax.spy(M, markersize=2.2, color=C_RIC)
        ax.set_title(t, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(True)
            sp_.set_color("#B0B0B0")
    save(fig, "story_kkt_spy")


def fig_scaling_sota(e4):
    """Cost per iteration and total time vs problem size, N = 25 ... 800 (slalom)."""
    rows = [("ipm", "M1 Interior point (dense)", C_FIELD), ("sqp-exact", "M2 SQP", C_SIMPLE),
            ("panoc-alm", "M3 PANOC-ALM", C_PANOC), ("ipm-riccati", "M4 Riccati IP", C_RIC),
            ("ipopt", "IPOPT (C++, reference)", C_REF)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, lab, col in rows:
        rs = sorted([r for r in e4 if r["solver"] == k and r["scenario"] == "slalom"],
                    key=lambda r: r["N"])
        n = np.array([r["n"] for r in rs])
        ok = np.array([bool(r.get("success")) for r in rs])
        tpi = np.array([r["time"] / max(r["iters"], 1) for r in rs])
        tt = np.array([r["time"] for r in rs])
        axes[0].loglog(n, tpi, "-", color=col, lw=2.6, label=lab)
        axes[0].loglog(n[ok], tpi[ok], "o", color=col, ms=7)
        axes[0].loglog(n[~ok], tpi[~ok], "o", mfc="white", mec=col, ms=7, mew=2)
        axes[1].loglog(n[ok], tt[ok], "o-", color=col, lw=2.6, ms=7)
    axes[0].set(xlabel="number of variables n", ylabel="time per iteration [s]")
    axes[1].set(xlabel="number of variables n", ylabel="time to converge [s]")
    for ax in axes:
        ax.set_xticks([154, 304, 604, 1204, 2404, 4804])
        ax.set_xticklabels(["154", "304", "604", "1.2k", "2.4k", "4.8k"])
        ax.minorticks_off()
        ax.grid(alpha=0.25)
    axes[0].text(1500, 1.6, r"$\approx n^{2.4}$", fontsize=16, color=C_FIELD, fontweight="bold", va="center")
    axes[0].text(6000, 0.9, r"$\approx n^{1.1}$", fontsize=16, color=C_RIC, fontweight="bold", va="center")
    axes[0].text(6000, 3.3e-3, "≈ const", fontsize=16, color=C_PANOC, fontweight="bold",
                 va="center")
    axes[0].set_xlim(120, 1.6e4)
    axes[0].set_ylim(1e-3, 5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "story_scaling_sota")


def fig_localmin(e1):
    """Same problem, same start, three different KKT points (slalom)."""
    sc = scenarios.get("slalom")
    rows = [("ipm", "7.774  (M1, M3, M4, IPOPT, penalty, SCP)", "#4FC3F7"),
            ("alm-newton", "7.973  (ALM, Newton)", "#7CE38B"),
            ("sqp-exact", "8.208  (M2 SQP)", "#D7A6FF")]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    road(ax, sc)
    for k, lab, col in rows:
        X, _ = unpack(np.array(e1[("slalom", k, "straight")]["z"]), sc.N)
        ax.plot(X[:, 0], X[:, 1], color=col, lw=3.2, zorder=5, label=f"f* = {lab}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False, fontsize=15)
    save(fig, "story_localmin")


def draw_sota(only=None):
    if only == "localmin":
        fig_localmin({(r["scenario"], r["solver"], r["init"]): r for r in load("e1_benchmark.jsonl")})
        return
    if only == "fix":
        fig_scaling_sota(load("e4_scaling.jsonl"))
        fig_panoc({(r["scenario"], r["solver"], r["init"]): r for r in load("e1_benchmark.jsonl")})
        return
    cache = pickle.load(open(CACHE, "rb"))
    e1 = {(r["scenario"], r["solver"], r["init"]): r for r in load("e1_benchmark.jsonl")}
    fig_race_gif(cache)
    fig_robust(load("e2_random.jsonl"))
    fig_trap(cache)
    e4 = load("e4_scaling.jsonl")
    fig_scaling(e4)
    fig_scaling_sota(e4)
    fig_panoc(e1)
    fig_kkt_spy()


def draw():
    cache = pickle.load(open(CACHE, "rb"))
    e1 = {(r["scenario"], r["solver"], r["init"]): r for r in load("e1_benchmark.jsonl")}
    fig_drive(e1)
    fig_problem(e1)
    fig_hole()
    fig_think(cache)
    fig_fines_road(cache)
    fig_race_gif(cache)
    fig_landscapes()
    fig_simplify(cache)
    fig_canyon()
    fig_steps_canyon(load("e3_conditioning.jsonl"))
    print("race averages", fig_race(e1))
    fig_robust(load("e2_random.jsonl"))
    fig_gallery(cache)
    fig_pricetags(e1)
    fig_trap(cache)
    fig_scaling(load("e4_scaling.jsonl"))


if __name__ == "__main__":
    {"prep": prep, "draw": draw, "prep_sota": prep_sota, "draw_sota": draw_sota}[sys.argv[1]](*sys.argv[2:])
