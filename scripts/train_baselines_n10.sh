#!/usr/bin/env bash
# Train the N=10 conference baselines and ablations.
#
# Set OVERWRITE=1 to remove an existing output directory before training.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/swarm_formation"

PY="${PY:-python}"
RUNS="../runs/mappo"
START_TS=$(date +%s)

log() {
    local elapsed=$(( $(date +%s) - START_TS ))
    printf "[N10 %02d:%02d:%02d] %s\n" \
        $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60)) "$*"
}

prepare_out_dir() {
    local dir="$1"
    if [[ -e "$dir" ]]; then
        if [[ "${OVERWRITE:-0}" == "1" ]]; then
            rm -rf "$dir"
        else
            echo "Output exists: $dir"
            echo "Set OVERWRITE=1 to retrain and replace it."
            exit 1
        fi
    fi
}

mkdir -p "$RUNS"

log "Tolstaya edge-aware BC"
prepare_out_dir "$RUNS/bc_tolstaya_edge"
"$PY" train_tolstaya.py --variant edge --config ../configs/n10_all_shapes.yaml \
    --out_dir "$RUNS/bc_tolstaya_edge" --device cuda \
    --total_collections 200 --n_envs 64 --rollout_steps 300 \
    --eval_envs 64 --eval_T 300 \
    > "$RUNS/bc_tolstaya_edge.log" 2>&1

log "Tolstaya pure-topology BC"
prepare_out_dir "$RUNS/bc_tolstaya_pure"
"$PY" train_tolstaya.py --variant pure --config ../configs/n10_all_shapes.yaml \
    --out_dir "$RUNS/bc_tolstaya_pure" --device cuda \
    --total_collections 200 --n_envs 64 --rollout_steps 300 \
    --eval_envs 64 --eval_T 300 \
    > "$RUNS/bc_tolstaya_pure.log" 2>&1

log "MeanAgg BC"
prepare_out_dir "$RUNS/bc_meanagg"
"$PY" train_bc.py --config ../configs/n10_all_shapes.yaml \
    --actor_type meanagg --out_dir "$RUNS/bc_meanagg" --device cuda \
    --total_collections 200 --n_envs 64 --rollout_steps 300 \
    --eval_envs 64 --eval_T 300 \
    > "$RUNS/bc_meanagg.log" 2>&1

log "MeanAgg PPO + BC anchor"
prepare_out_dir "$RUNS/mappo_meanagg"
"$PY" train_mappo.py --config ../configs/n10_all_shapes.yaml \
    --actor_type meanagg \
    --actor_init "$RUNS/bc_meanagg/best.pt" \
    --bc_anchor "$RUNS/bc_meanagg/best.pt" --bc_anchor_w 2.0 \
    --out_dir "$RUNS/mappo_meanagg" --device cuda \
    --total_updates 500 \
    > "$RUNS/mappo_meanagg.log" 2>&1

log "GAT BC + PPO without anchor"
prepare_out_dir "$RUNS/mappo_bc_no_anchor"
"$PY" train_mappo.py --config ../configs/n10_all_shapes.yaml \
    --actor_type gat \
    --actor_init "$RUNS/bc_multishape_scratch/best.pt" \
    --out_dir "$RUNS/mappo_bc_no_anchor" --device cuda \
    --total_updates 500 \
    > "$RUNS/mappo_bc_no_anchor.log" 2>&1

log "GAT PPO from scratch"
prepare_out_dir "$RUNS/mappo_from_scratch"
"$PY" train_mappo.py --config ../configs/n10_all_shapes.yaml \
    --actor_type gat \
    --out_dir "$RUNS/mappo_from_scratch" --device cuda \
    --total_updates 500 \
    > "$RUNS/mappo_from_scratch.log" 2>&1

log "All N=10 baseline training complete."
