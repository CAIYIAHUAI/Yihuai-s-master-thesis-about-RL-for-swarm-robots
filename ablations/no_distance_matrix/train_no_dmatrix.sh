#!/usr/bin/env bash
# "Without distance matrix" ablation (N=10, N=21, N=28).
#
# Re-runs the full BC -> PPO(+BC anchor) -> multi-seed-eval pipeline with the SAME
# hyperparameters as the with-distance baselines, changing exactly one thing: the
# robots' edge feature no longer contains d_desired (the target-shape distance
# matrix). edge_dim_selfshape=13 is set by the override configs in this folder. The
# hand-coded teacher still computes its DAgger labels from slot_pair_dist, so the
# teacher keeps the distances; only the robots are no longer given them.
#
# Mirrors scripts/train_scaling_n21_n28.sh + the N=10 headline recipe. Outputs land under
# ablations/no_distance_matrix/. Set ONLY="n10" (or "n21"/"n28") to run a single size.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
WD="ablations/no_distance_matrix"
CFG_DIR="$WD"
RUNS="$WD/runs"
EVAL_OUT="$WD/eval_no_dmatrix.txt"
START_TS=$(date +%s)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

log() {
    local elapsed=$(( $(date +%s) - START_TS ))
    printf "[no-dmatrix %02d:%02d:%02d] %s\n" \
        $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) "$*"
}

mkdir -p "$RUNS"
: > "$EVAL_OUT"

run_one() {
    local tag="$1"          # n10 / n21 / n28
    local cfg="$2"          # override config filename (in $CFG_DIR)
    local bc_cols="$3"
    local bc_envs="$4"
    local bc_rollout="$5"
    local bc_eval_envs="$6"
    local T="$7"
    local ppo_updates="$8"
    local eval_envs="$9"
    local shapes="${10}"

    log "[$tag] BC / DAgger (student edges = 13-dim, teacher keeps distances)"
    rm -rf "$RUNS/bc_${tag}"
    "$PY" swarm_formation/train_bc.py --config "$CFG_DIR/$cfg" --actor_type gat \
        --out_dir "$RUNS/bc_${tag}" --device cuda \
        --total_collections "$bc_cols" --n_envs "$bc_envs" --rollout_steps "$bc_rollout" \
        --eval_envs "$bc_eval_envs" --eval_T "$T" \
        > "$RUNS/bc_${tag}.log" 2>&1

    log "[$tag] PPO + BC anchor"
    rm -rf "$RUNS/mappo_${tag}"
    "$PY" swarm_formation/train_mappo.py --config "$CFG_DIR/$cfg" --actor_type gat \
        --actor_init "$RUNS/bc_${tag}/best.pt" \
        --bc_anchor  "$RUNS/bc_${tag}/best.pt" --bc_anchor_w 2.0 \
        --out_dir "$RUNS/mappo_${tag}" --device cuda \
        --total_updates "$ppo_updates" \
        > "$RUNS/mappo_${tag}.log" 2>&1

    log "[$tag] eval per-shape, T=$T"
    {
        echo "========================================="
        echo "EVAL no-dmatrix: ${tag} (T=$T)"
        echo "========================================="
        "$PY" swarm_formation/evaluate.py --ckpt "$RUNS/mappo_${tag}/best_eval.pt" \
            --shapes $shapes --n_seeds 5 --n_envs "$eval_envs" --T "$T" --device cuda
        echo
    } >> "$EVAL_OUT" 2>&1
}

# tag  cfg                          bc_cols envs rollout bc_eval_envs  T  ppo  eval_envs  shapes
[[ "${ONLY:-}" == "" || "${ONLY:-}" == "n10" ]] && \
    run_one n10 n10_all_shapes.yaml   200 64 300 64 300 500 64 "t4 hex4 ring10 row5x2"
[[ "${ONLY:-}" == "" || "${ONLY:-}" == "n21" ]] && \
    run_one n21 n21_all_shapes.yaml 200 16 600 32 600 800 64 "t6 hex21 ring21 row7x3"
[[ "${ONLY:-}" == "" || "${ONLY:-}" == "n28" ]] && \
    run_one n28 n28_all_shapes.yaml 200  8 600 16 600 800 16 "t7 hex28 ring28 row7x4"

log "ALL NO-DMATRIX DONE -> $EVAL_OUT"
