#!/usr/bin/env bash
# Rebuild the final paper evaluation logs from kept checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/swarm_formation"

PY="${PY:-python}"
RUNS="../runs/mappo"
N10_OUT="$RUNS/eval_n10_baselines.txt"

eval_policy() {
    local out="$1"
    local label="$2"
    local ckpt="$3"
    local n_envs="$4"
    local T="$5"
    local shapes="$6"
    local extra="${7:-}"

    {
        echo
        echo "========================================="
        echo "EVAL: ${label} T=${T}"
        echo "========================================="
        "$PY" evaluate.py --ckpt "$ckpt" \
            --shapes $shapes --n_envs "$n_envs" --T "$T" --n_seeds 5 --device cuda \
            $extra
    } >> "$out"
}

: > "$N10_OUT"
eval_policy "$N10_OUT" "v18d main" "$RUNS/self_shape_multishape_p2/best_eval.pt" 64 300 "t4 hex4 ring10 row5x2"
eval_policy "$N10_OUT" "BC-only GAT" "$RUNS/bc_multishape_scratch/best.pt" 64 300 "t4 hex4 ring10 row5x2"
eval_policy "$N10_OUT" "MeanAgg PPO+anchor" "$RUNS/mappo_meanagg/best_eval.pt" 64 300 "t4 hex4 ring10 row5x2"
eval_policy "$N10_OUT" "BC + PPO no anchor" "$RUNS/mappo_bc_no_anchor/best_eval.pt" 64 300 "t4 hex4 ring10 row5x2"
eval_policy "$N10_OUT" "PPO from scratch" "$RUNS/mappo_from_scratch/best_eval.pt" 64 300 "t4 hex4 ring10 row5x2"
eval_policy "$N10_OUT" "MeanAgg BC-only" "$RUNS/bc_meanagg/best.pt" 64 300 "t4 hex4 ring10 row5x2"
eval_policy "$N10_OUT" "Tolstaya edge BC-only" "$RUNS/bc_tolstaya_edge/best.pt" 64 300 "t4 hex4 ring10 row5x2" "--actor_type tolstaya_edge"
eval_policy "$N10_OUT" "Tolstaya pure BC-only" "$RUNS/bc_tolstaya_pure/best.pt" 64 300 "t4 hex4 ring10 row5x2" "--actor_type tolstaya_pure"

# Large-team (N=21 / N=28) fixed-local scaling uses one ring-inclusive model per
# size; those eval logs are produced by scripts/train_scaling_n21_n28.sh into
# runs/mappo/eval_scaling.txt.

echo "Final eval logs written:"
echo "  $N10_OUT"
