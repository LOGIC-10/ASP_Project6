#!/usr/bin/env bash
# Run full high-fidelity pitch/tempo sweeps with caching and plotting.
# Usage:
#   nohup bash run_variations.sh > variation_run.log 2>&1 &
# Adjust WORKERS/NQ_* as needed.

set -euo pipefail

WORKERS=${WORKERS:-4}
NQ_CURVE=${NQ_CURVE:-6}
NQ_HEAT=${NQ_HEAT:-3}
MAT_PATH="Project6_musicDB.mat"

echo "[`date`] ensure fingerprint caches"
conda run -n asp python - <<PY
from variation_runner import ensure_fingerprint_caches
ensure_fingerprint_caches("${MAT_PATH}")
PY

echo "[`date`] run pitch/tempo/heatmap sweeps (workers=${WORKERS}, nq_curve=${NQ_CURVE}, nq_heat=${NQ_HEAT})"
# Launch three phases in parallel to shorten wall time. Each phase already parallelizes internally by --workers.
conda run -n asp python variation_runner.py --mode pitch --workers ${WORKERS} --nq_curve ${NQ_CURVE} &
PID_PITCH=$!
0conda run -n asp python variation_runner.py --mode tempo --workers ${WORKERS} --nq_curve ${NQ_CURVE} &
PID_TEMPO=$!
conda run -n asp python variation_runner.py --mode heatmap --workers ${WORKERS} --nq_heat ${NQ_HEAT} &
PID_HEAT=$!
wait ${PID_PITCH} ${PID_TEMPO} ${PID_HEAT}

echo "[`date`] plot from cache"
conda run -n asp python plot_variation_curves_hq.py --from-cache cache/variation_results.json

echo "[`date`] done"
