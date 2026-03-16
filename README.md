# MAPPO Triangle-Fill: Kilobot Swarm Formation Control

Multi-Agent PPO (MAPPO) training for the VMAS `triangle_fill` scenario — a swarm of Kilobot-like agents learns to form a triangle formation using only local observations.

## Quick Start

**Prerequisites:** A Linux machine with an NVIDIA GPU (CUDA 12.8 compatible).

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/yihuai-master-thesis.git
cd yihuai-master-thesis
uv sync          # downloads Python 3.10, creates .venv, installs all dependencies
```

That's it! No need to manually install Python, create virtual environments, or manage CUDA libraries.

### 3. Train

```bash
uv run python train.py --config configs/v7.yaml
```

### 4. Evaluate / Render

```bash
uv run python evaluate.py --checkpoint runs/<run_name>/ckpt_final.pt
uv run python render_video.py --checkpoint runs/<run_name>/ckpt_final.pt
```

## Project Structure

```
├── train.py                 # MAPPO training loop
├── evaluate.py              # Evaluation script
├── eval_rollout.py          # Rollout evaluation utilities
├── visualize.py             # Visualization helpers
├── render_video.py          # Video rendering
├── scenarios/
│   └── triangle_fill.py     # VMAS scenario definition
├── utils/
│   └── triangle_reward/     # Reward computation (Sinkhorn, templates)
├── configs/                 # Training hyperparameter configs (YAML)
├── tests/                   # Unit tests
├── pyproject.toml           # Project metadata & dependencies
└── uv.lock                  # Locked dependency versions (reproducible)
```

## Running Tests

```bash
uv run pytest tests/
```

## CPU-only

If you don't have a GPU, change `device: cpu` in the config YAML, or pass `--device cpu`. Note: training will be significantly slower.
