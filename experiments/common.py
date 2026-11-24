"""Shared experiment plumbing: run one (scenario, solver, init) case in a worker
process and return a JSON-serialisable record."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker: fair timings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

import numpy as np  # noqa: E402

from avtraj import build_nlp, min_clearance, scenarios  # noqa: E402
from avtraj.solvers import SOLVERS, convex  # noqa: E402


def initial_guess(nlp, init):
    """'straight'   - zero-control rollout (drives through obstacles);
       'corridor'   - solution of the convex corridor QP around 'straight';
       'offset:<dy>' - straight rollout shifted sideways by dy for k >= 1
                       (violates the dynamics at k = 0; used to pick a homotopy class)."""
    if init == "straight":
        return nlp.z0.copy()
    if init.startswith("offset:"):
        z = nlp.z0.copy()
        N = nlp.meta["N"]
        z[4 + 1: 4 * (N + 1): 4] += float(init.split(":")[1])
        return z
    if init == "corridor":
        out = convex.corridor_qp(nlp, nlp.z0)
        return out["z"] if out["z"] is not None else nlp.z0.copy()
    raise ValueError(init)


def run_case(case):
    """case = dict(scenario, solver, init='straight', N=50, opts={}, keep_z=False)."""
    rec = dict(case)
    try:
        sc = scenarios.get(case["scenario"], N=case.get("N", 50))
        t0 = time.perf_counter()
        nlp = build_nlp(sc)
        rec["build_time"] = time.perf_counter() - t0
        z0 = initial_guess(nlp, case.get("init", "straight"))
        if "obstacle_scale" in case:            # sensitivity experiment: perturb a radius
            j, dr = case["obstacle_scale"]
            sc.obstacles[j, 2] += dr
            nlp = build_nlp(sc)
        if "z0" in case:
            z0 = np.asarray(case["z0"], dtype=float)
        res = SOLVERS[case["solver"]](nlp, z0, **case.get("opts", {}))
        rec.update(res.summary())
        rec["solver_name"], rec["solver"] = rec["solver"], case["solver"]
        rec["clearance"] = min_clearance(nlp, res.z)
        rec["n"], rec["m_eq"], rec["m_in"] = nlp.n, nlp.m_eq, nlp.m_in
        rec["history"] = [{k: _py(v) for k, v in h.items()} for h in res.history]
        if case.get("keep_z", False):
            rec["z"] = res.z.tolist()
            rec["mu"] = res.mu.tolist()
            rec["obs_rows"] = nlp.meta["obs_rows"].tolist()
    except Exception as e:  # keep the sweep alive, record the failure
        rec.update(status="exception", success=False, error=repr(e),
                   traceback=traceback.format_exc())
    return {k: _py(v) for k, v in rec.items()}


def _py(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, float) and not np.isfinite(v):
        return str(v)
    return v


def run_all(cases, out_file, workers=None):
    """Run cases in parallel and write one JSON record per line."""
    RESULTS.mkdir(exist_ok=True)
    workers = workers or min(len(cases), max(1, os.cpu_count() - 8))
    out = RESULTS / out_file
    t0 = time.time()
    with get_context("fork").Pool(workers) as pool, open(out, "w") as fh:
        for i, rec in enumerate(pool.imap_unordered(run_case, cases)):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(cases)} {time.time() - t0:6.0f}s] {rec['scenario']:18s} "
                  f"{rec['solver']:15s} {rec.get('init', ''):9s} {rec.get('status', ''):24s} "
                  f"f={rec.get('f', float('nan')):.4f} viol={rec.get('viol', float('nan')):.1e} "
                  f"it={rec.get('iters', -1)} t={rec.get('time', float('nan')):.2f}s", flush=True)
    return out


def load(out_file):
    with open(RESULTS / out_file) as fh:
        return [json.loads(line) for line in fh]
