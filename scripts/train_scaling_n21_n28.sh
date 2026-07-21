#!/usr/bin/env bash
# Fixed-local scaling experiment (N=21, N=28).
#
# Trains ONE model per team size with all formations in the training pool
# (triangle, hexagon, rectangular grid, and the large ring) and stress_norm
# fixed_n_minus_1, then evaluates that single model per-shape. The large ring is
# included so its failure is measured by the same model that forms the other
# shapes, with no pool/normalization/cross-model confound.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/swarm_formation"

PY="${PY:-python}"
RUNS="../runs/mappo"
EVAL_OUT="$RUNS/eval_scaling.txt"
START_TS=$(date +%s)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

log() {
    local elapsed=$(( $(date +%s) - START_TS ))
    printf "[scaling %02d:%02d:%02d] %s\n" \
        $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) "$*"
}

mkdir -p "$RUNS"
: > "$EVAL_OUT"

run_one() {
    local tag="$1"          # n21 / n28
    local cfg="$2"          # config filename
    local bc_envs="$3"
    local bc_eval_envs="$4"
    local eval_envs="$5"
    local shapes="$6"

    log "[$tag] BC / DAgger"
    rm -rf "$RUNS/bc_${tag}"
    "$PY" train_bc.py --config "$cfg" --actor_type gat \
        --out_dir "$RUNS/bc_${tag}" --device cuda \
        --total_collections 200 --n_envs "$bc_envs" --rollout_steps 600 \
        --eval_envs "$bc_eval_envs" --eval_T 600 \
        > "$RUNS/bc_${tag}.log" 2>&1

    log "[$tag] PPO + anchor"
    rm -rf "$RUNS/mappo_${tag}"
    "$PY" train_mappo.py --config "$cfg" --actor_type gat \
        --actor_init "$RUNS/bc_${tag}/best.pt" \
        --bc_anchor  "$RUNS/bc_${tag}/best.pt" --bc_anchor_w 2.0 \
        --out_dir "$RUNS/mappo_${tag}" --device cuda \
        --total_updates 800 \
        > "$RUNS/mappo_${tag}.log" 2>&1

    log "[$tag] eval per-shape (incl. ring), T=600"
    {
        echo "========================================="
        echo "EVAL scaling: ${tag} (T=600)"
        echo "========================================="
        "$PY" evaluate.py --ckpt "$RUNS/mappo_${tag}/best_eval.pt" \
            --shapes $shapes --n_seeds 5 --n_envs "$eval_envs" --T 600 --device cuda
        echo
    } >> "$EVAL_OUT" 2>&1
}

run_one n21 ../configs/n21_all_shapes.yaml 16 32 64 "t6 hex21 ring21 row7x3"
run_one n28 ../configs/n28_all_shapes.yaml  8 16 16 "t7 hex28 ring28 row7x4"

log "ALL SCALING DONE -> $EVAL_OUT"
