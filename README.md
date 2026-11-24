# Constrained trajectory optimisation for obstacle avoidance

Trajectory planning for a car on a two-lane road, posed as a nonconvex NLP and solved with
interior-point, SQP, PANOC-ALM and Riccati interior-point methods (plus penalty, ALM, SCP
baselines and IPOPT as a reference).

## Install

```bash
conda create -n avtraj python=3.11 -y
conda activate avtraj
pip install -r requirements.txt
```

## Reproduce

```bash
bash run_all.sh                       # all experiments; writes results/ and figures/
```

or one experiment at a time (`--plot-only` redraws figures from saved results):

```bash
python experiments/e1_benchmark.py    # main comparison: 3 scenarios, all solvers, 2 initial guesses
python experiments/e2_random.py       # random obstacle fields
python experiments/e3_conditioning.py # penalty conditioning vs ALM
python experiments/e4_scaling.py      # horizon length N = 25 ... 800
python experiments/e5_ablation.py     # design-choice ablations
python experiments/e6_multipliers.py  # multipliers and local infeasibility
```

Extra figures: `python experiments/fig_schematic.py`, `python experiments/story_figures.py prep && python experiments/story_figures.py draw`, and `python experiments/method_pipelines.py` (needs pdflatex and PyMuPDF).

## Quick use

```python
from avtraj import build_nlp, scenarios
from avtraj.solvers import SOLVERS

res = SOLVERS["ipm"](build_nlp(scenarios.get("slalom")))  # or "sqp-exact", "panoc-alm", "ipm-riccati"
print(res.status, res.f, res.iters, res.time)
```
