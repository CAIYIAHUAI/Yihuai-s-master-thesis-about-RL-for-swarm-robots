#!/usr/bin/env bash
# Re-eval the N=10 headline checkpoint with PyTorch deterministic algorithms
# enabled. Results land in runs/mappo/eval_deterministic.txt.
#
# The large-team (N=21 / N=28) deterministic results come from the single
# ring-inclusive model per size and are produced by scripts/train_scaling_n21_n28.sh
# into runs/mappo/eval_scaling.txt.
set -euo pipefail
cd "$(dirname "$0")/../swarm_formation"
PY="${PY:-python}"
RUNS=../runs/mappo
OUT=$RUNS/eval_deterministic.txt
START_TS=$(date +%s)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_ALLOC_CONF=expandable_segments:True

log() {
    local elapsed=$(( $(date +%s) - START_TS ))
    printf "[DET %02d:%02d:%02d] %s\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) "$*"
}

eval_one() {
    local label=$1 ckpt=$2 envs=$3 T=$4 shapes=$5
    {
        echo "========================================="
        echo "EVAL DET: $label  (T=$T)"
        echo "========================================="
        $PY evaluate.py --ckpt "$ckpt" \
            --shapes $shapes --n_seeds 5 --n_envs "$envs" --T "$T" --device cuda
    } 2>&1 | tee -a "$OUT"
    echo | tee -a "$OUT"
}

: > "$OUT"

log "N=10 v18d main"
eval_one "N=10 v18d main"      "$RUNS/self_shape_multishape_p2/best_eval.pt"  64 300 "t4 hex4 ring10 row5x2"

log "ALL DET EVAL COMPLETE"
