# Decentralized Self-Shape Formation for Swarm Robots

Code for my master's thesis on reinforcement learning for swarm robot shape
formation under **strict local communication**. Each agent observes only local
geometry, local edge/task distances, and its own slot encoding — no global
target vector and no global positions.

The method is a three-stage pipeline sharing one GAT+GRU actor:

1. **Expert controller** — a hand-coded local distance-spring controller acts
   as the teacher.
2. **DAgger behavior cloning (BC)** — the actor learns the teacher's labels on
   states visited by the actor's own rollouts.
3. **MAPPO refinement** — PPO on a formation-stress reward, with the frozen BC
   actor as an action-space MSE anchor.

Team sizes N=10, N=21, and N=28 use the same method and a fixed communication
radius; only N-dependent experiment settings change.

## Repository Structure

```text
configs/                      All experiment hyperparameters
  n10_triangle.yaml           N=10 single-shape (T4 triangle) config
  n10_all_shapes.yaml         N=10 four-shape config
  n21_all_shapes.yaml         N=21 config; one model, all formations (incl. large ring)
  n28_all_shapes.yaml         N=28 config; one model, all formations (incl. large ring)

swarm_formation/              The implementation
  environment.py              Vectorized 2D unicycle env, local observations, resets
  observations.py             Node/edge feature definitions and observation builder
  templates.py                Shape template factories (triangle, hexagon, ring, grid)
  target.py                   Target map, coverage and matching utilities
  shape_metrics.py            Shape matching, collision metrics, formation-stress reward
  expert_controller.py        Hand-coded local spring-controller teacher
  model.py                    GAT+GRU actor, MeanAgg actor, centralized critic
  model_tolstaya.py           Tolstaya-style K-tap graph-filter baseline actors
  train_bc.py                 DAgger BC training entrypoint
  train_mappo.py              MAPPO training entrypoint (stress reward + BC anchor)
  train_tolstaya.py           Teacher-forced BC for the Tolstaya baselines
  evaluate.py                 Deterministic multi-seed policy/controller evaluation
  render.py                   Render an MP4 of a checkpoint forming any shape
  test_swarm_formation.py     Unit and smoke tests

scripts/                      One script per experiment group
  train_baselines_n10.sh      N=10 architecture baselines and ablations
  train_scaling_n21_n28.sh    N=21 / N=28 scaling experiment (train + eval)
  eval_n10.sh                 Rebuild the N=10 evaluation logs
  eval_deterministic.sh       Deterministic re-eval of the N=10 main checkpoint

ablations/
  no_distance_matrix/         "Without distance matrix" ablation (configs + runner)
```

All outputs (checkpoints, logs, evaluation text files, videos) are written
under `runs/` and are git-ignored: this repository contains only what is
needed to re-run the experiments.

## Setup

```bash
conda create -n swarm-formation python=3.10 -y
conda activate swarm-formation
pip install -r requirements.txt
```

Versions in `requirements.txt` are the exact tested versions (PyTorch 2.10.0
with CUDA 12.8). For PyTorch Geometric, follow the official wheel instructions
if your CUDA and Torch combination needs a specific index.

Verify the installation:

```bash
PYTHONPATH=swarm_formation python -c "import environment, observations, templates, target, shape_metrics, expert_controller, model, model_tolstaya, train_bc, train_mappo, train_tolstaya, evaluate, render"
pytest swarm_formation/test_swarm_formation.py -x -q
```

Short CPU smoke run:

```bash
python swarm_formation/train_mappo.py \
    --config configs/n10_triangle.yaml \
    --smoke --device cpu --out_dir runs/smoke
```

## Configs

| N | Config | Shapes | Eval T | Stress norm |
| ---: | --- | --- | ---: | --- |
| 10 | `configs/n10_triangle.yaml` | T4 triangle only | 300 | — |
| 10 | `configs/n10_all_shapes.yaml` | triangle, hexagon, circle, rectangular grid | 300 | fixed_n_minus_1 |
| 21 | `configs/n21_all_shapes.yaml` | triangle, hexagon, large ring, rectangular grid | 600 | fixed_n_minus_1 |
| 28 | `configs/n28_all_shapes.yaml` | triangle, hexagon, large ring, rectangular grid | 600 | fixed_n_minus_1 |

Each team size trains a single model on all of its formations (including the
large ring at N=21/28, which is included so its failure is measured by the same
model that forms the other shapes) and is evaluated per formation.

## Re-Running the Experiments

Run everything from the repository root. Each stage states the output
directory the later stages expect.

### 1. N=10 single-shape stage

```bash
python swarm_formation/train_bc.py \
    --config configs/n10_triangle.yaml \
    --out_dir runs/mappo/bc_v18_dagger \
    --device cuda \
    --total_collections 200

python swarm_formation/train_mappo.py \
    --config configs/n10_triangle.yaml \
    --out_dir runs/mappo/self_shape_v18d \
    --actor_init runs/mappo/bc_v18_dagger/best.pt \
    --bc_anchor runs/mappo/bc_v18_dagger/best.pt \
    --bc_anchor_w 2.0 \
    --device cuda
```

### 2. N=10 multi-shape main model

BC warm-started from the single-shape model, then MAPPO with the BC anchor:

```bash
python swarm_formation/train_bc.py \
    --config configs/n10_all_shapes.yaml \
    --out_dir runs/mappo/bc_multishape \
    --actor_init runs/mappo/self_shape_v18d/best_eval.pt \
    --device cuda \
    --total_collections 200

python swarm_formation/train_mappo.py \
    --config configs/n10_all_shapes.yaml \
    --out_dir runs/mappo/self_shape_multishape_p2 \
    --actor_init runs/mappo/bc_multishape/best.pt \
    --bc_anchor runs/mappo/bc_multishape/best.pt \
    --bc_anchor_w 2.0 \
    --total_updates 800 \
    --device cuda
```

`runs/mappo/self_shape_multishape_p2/best_eval.pt` is the main N=10 model.

### 3. N=10 BC-only model (also used by the baselines)

```bash
python swarm_formation/train_bc.py \
    --config configs/n10_all_shapes.yaml \
    --out_dir runs/mappo/bc_multishape_scratch \
    --device cuda \
    --total_collections 200
```

### 4. N=10 architecture baselines and ablations

Trains the Tolstaya K-tap baselines, the MeanAgg actor (BC and PPO+anchor),
BC+PPO without the anchor (initialized from step 3), and PPO from scratch:

```bash
scripts/train_baselines_n10.sh
```

### 5. N=21 / N=28 scaling

One model per team size, trained on all formations including the large ring,
then evaluated per shape (results land in `runs/mappo/eval_scaling.txt`):

```bash
scripts/train_scaling_n21_n28.sh
```

### 6. "Without distance matrix" ablation

Identical pipeline and hyperparameters, but the robots' edge features no
longer contain the target distance matrix (the teacher still uses it for
labels). Outputs land under `ablations/no_distance_matrix/runs/`:

```bash
bash ablations/no_distance_matrix/train_no_dmatrix.sh          # all sizes
ONLY=n10 bash ablations/no_distance_matrix/train_no_dmatrix.sh # one size
```

### 7. Evaluation

Rebuild the evaluation logs from the trained checkpoints:

```bash
scripts/eval_n10.sh            # all N=10 rows -> runs/mappo/eval_n10_baselines.txt
scripts/eval_deterministic.sh  # deterministic re-eval of the main N=10 model
```

Or evaluate any checkpoint directly:

```bash
python swarm_formation/evaluate.py \
    --ckpt runs/mappo/self_shape_multishape_p2/best_eval.pt \
    --shapes t4 hex4 ring10 row5x2 \
    --n_seeds 5 --n_envs 64 --T 300 --device cuda
```

Evaluate the hand-coded expert controller (no checkpoint):

```bash
python swarm_formation/evaluate.py \
    --controller --config configs/n21_all_shapes.yaml \
    --shapes t6 hex21 ring21 row7x3 \
    --n_seeds 5 --n_envs 64 --T 600 --device cuda
```

Registered shape names: `t4 hex4 ring10 row5x2` (N=10),
`t6 hex21 ring21 row7x3` (N=21), `t7 hex28 ring28 row7x4` (N=28).

### 8. Render a video

```bash
python swarm_formation/render.py \
    --ckpt runs/mappo/self_shape_multishape_p2/best_eval.pt \
    --shape hex4 --out runs/hex4.mp4 --device cuda
```

## License

MIT.
