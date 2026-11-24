#!/usr/bin/env bash
# Reproduce all results (results/) and figures (figures/).
#   bash run_all.sh            (about 10-20 minutes on a many-core machine)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONNOUSERSITE=1

if [ -d tests ]; then python -m pytest -q tests; fi
python experiments/e1_benchmark.py
python experiments/e2_random.py
python experiments/e3_conditioning.py
python experiments/e4_scaling.py
python experiments/e5_ablation.py
python experiments/e6_multipliers.py
echo "All results in results/, figures in figures/"
