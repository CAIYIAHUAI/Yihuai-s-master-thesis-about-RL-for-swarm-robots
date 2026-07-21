"""v18 DAgger BC using the hand-coded local controller teacher.

The collection loop uses DAgger: the env steps with the actor's own action
while the controller labels each visited state. Hidden state propagates within
and across chunks via the saved h_init trajectory, matching eval-time dynamics.

Optional beta-mixing: action_executed = beta * controller + (1-beta) * actor.
Beta anneals 1.0 → 0.0 over the first `dagger_warmup` collections. Helps the
early actor (which produces near-zero noise) avoid getting trapped in bad
states before BC labels can shape it.
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from environment import MeanShiftEnv, ARCH_VERSION, load_config
from train_mappo import _build_actor, build_selfshape_obs, _quick_eval_selfshape, load_partial_state_dict
from expert_controller import compute_controller_action
from observations import EDGE_FEATURE_DIM_SELFSHAPE, selfshape_node_dim


def collect_dagger(env: MeanShiftEnv, actor, template_pair_dist: torch.Tensor,
                    comm_radius: float, k_spring: float, F_norm: float,
                    T_horizon: int, beta: float) -> dict:
    """DAgger rollout: env steps with (beta*controller + (1-beta)*actor) actions.

    Records actor's own visited states, controller labels at each state, and
    the hidden state going INTO each step (so training chunks can resume from
    matching h instead of zero).
    """
    env.reset_all()
    env.set_cue_present_prob(0.0)
    cue_off = torch.zeros(env.B, device=env.device, dtype=torch.bool)
    latent_z = env.latent_z.clone()

    h = torch.zeros(env.B, env.N, actor.hidden, device=env.device, dtype=env.positions.dtype)
    obs_list, edge_index_list, edge_attr_list = [], [], []
    label_list, h_init_list = [], []
    with torch.no_grad():
        for _t in range(T_horizon):
            exp_out = env.expert_action(noise_gen=None, build_obs=False)
            obs, ei, ea = build_selfshape_obs(env, exp_out,
                                               cue_present=cue_off,
                                               latent_z=latent_z,
                                               heading_locked=True)
            # Controller label at this state
            label = compute_controller_action(env, template_pair_dist,
                                                comm_radius=comm_radius,
                                                k_spring=k_spring,
                                                linear_max_speed=F_norm)
            # Actor action
            _logits, vel_mu, h_next = actor(obs, ei, ea, h)
            actor_action = torch.tanh(vel_mu).clamp(-1.0, 1.0)
            # Save h going INTO this step (before forward)
            h_init_list.append(h.detach().clone())
            obs_list.append(obs); edge_index_list.append(ei); edge_attr_list.append(ea)
            label_list.append(label)
            # Execute beta-mixed action and step env
            action_exec = beta * label + (1.0 - beta) * actor_action
            action_exec = action_exec.clamp(-1.0, 1.0)
            _v, _info, _done = env.step(action_exec)
            h = h_next

    return {
        "obs_list": obs_list, "edge_index_list": edge_index_list,
        "edge_attr_list": edge_attr_list, "label_list": label_list,
        "h_init_list": h_init_list,
    }


def chunk_loss(actor, batch: dict, t0: int, t1: int,
                alpha_cos: float, low_speed_weight: float, speed_threshold: float):
    """BC loss for one chunk [t0, t1). h init = batch['h_init_list'][t0]."""
    h = batch["h_init_list"][t0].detach().clone()
    mu_preds, labels = [], []
    for t in range(t0, t1):
        obs = batch["obs_list"][t]
        ei = batch["edge_index_list"][t]
        ea = batch["edge_attr_list"][t]
        _logits, vel_mu, h = actor(obs, ei, ea, h)
        mu_preds.append(torch.tanh(vel_mu))
        labels.append(batch["label_list"][t])
    mu_pred = torch.stack(mu_preds, dim=0)                                  # [Tc, B, N, 2]
    label = torch.stack(labels, dim=0)                                      # [Tc, B, N, 2]

    # Per-sample weighting: down-weight low-speed teacher signals
    expert_speed = label.norm(dim=-1)
    is_low_speed = expert_speed < speed_threshold
    weight = torch.where(is_low_speed,
                          torch.full_like(expert_speed, low_speed_weight),
                          torch.ones_like(expert_speed))
    weight = weight / weight.mean().clamp_min(1e-6)

    mse_per = ((mu_pred - label) ** 2).sum(-1)                              # [Tc, B, N]
    mse = (mse_per * weight).mean()

    eps = 1e-6
    label_dir = label / label.norm(dim=-1, keepdim=True).clamp_min(eps)
    pred_dir  = mu_pred / mu_pred.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos_sim = (label_dir * pred_dir).sum(-1)
    valid = (~is_low_speed).float()
    cos_loss = ((1.0 - cos_sim) * valid).sum() / valid.sum().clamp_min(1.0)

    loss = mse + alpha_cos * cos_loss
    diag = {
        "mse": float(mse.item()),
        "cos_loss": float(cos_loss.item()),
        "mu_mean_abs": float(mu_pred.abs().mean().item()),
        "label_mean_abs": float(label.abs().mean().item()),
    }
    return loss, diag


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=REPO.parent / "configs" / "n10_triangle.yaml")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--actor_type", choices=["gat", "meanagg", "tolstaya_edge", "tolstaya_pure"],
                   default=None, help="Actor architecture. Defaults to cfg.actor_type or gat.")
    p.add_argument("--n_envs", type=int, default=64)
    p.add_argument("--rollout_steps", type=int, default=300)
    p.add_argument("--chunk_len", type=int, default=20)
    p.add_argument("--ppo_epochs", type=int, default=2,
                   help="Training passes over each collection (lower since collection is expensive)")
    p.add_argument("--total_collections", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha_cos", type=float, default=0.3)
    p.add_argument("--low_speed_weight", type=float, default=0.5)
    p.add_argument("--speed_threshold", type=float, default=0.05)
    p.add_argument("--k_spring", type=float, default=1.0)
    p.add_argument("--linear_max_speed_d", type=float, default=0.05)
    p.add_argument("--dagger_warmup", type=int, default=50,
                   help="Cols over which beta drops 1.0 -> 0.0 (controller-mix anneal)")
    p.add_argument("--dagger_min_beta", type=float, default=0.0,
                   help="Floor for beta after warmup (0 = pure DAgger)")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_envs", type=int, default=64)
    p.add_argument("--eval_T", type=int, default=300)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--actor_init", type=Path, default=None)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg["action_mode"] = "continuous"
    if args.actor_type is not None:
        cfg["actor_type"] = args.actor_type
    cfg.setdefault("actor_type", "gat")
    cfg.setdefault("n_agents", 10)
    cfg.setdefault("node_dim_selfshape", selfshape_node_dim(int(cfg["n_agents"])))
    cfg.setdefault("edge_dim_selfshape", EDGE_FEATURE_DIM_SELFSHAPE)
    cfg["envs"] = args.n_envs

    spacing = float(cfg["target_spacing"])
    comm_radius = float(cfg["comm_radius"])
    F_norm = args.linear_max_speed_d * spacing
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    actor = _build_actor(cfg, device, selfshape=True)
    if args.actor_init is not None:
        ckpt = torch.load(args.actor_init, map_location=device, weights_only=False)
        nf, np_ = load_partial_state_dict(actor, ckpt["actor_state"], verbose=True)
        print(f"[bc-dagger] warm-start from {args.actor_init}: {nf} full, {np_} partial")
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    from templates import make_template
    shape_names = list(cfg.get("shape_pool", ["t4"]))
    templates = {name: make_template(name, spacing, device=device)
                 for name in shape_names}

    env = MeanShiftEnv(cfg, n_envs=args.n_envs, device=str(device), seed=cfg.get("seed", 1))

    print(f"[bc-dagger] device={device} n_envs={args.n_envs} rollout={args.rollout_steps} "
          f"chunk={args.chunk_len} ppo_epochs={args.ppo_epochs} cols={args.total_collections} "
          f"warmup={args.dagger_warmup} min_beta={args.dagger_min_beta}")

    best_succ = -1.0
    n_chunks = args.rollout_steps // args.chunk_len
    chunk_starts = [c * args.chunk_len for c in range(n_chunks)]

    for collection in range(1, args.total_collections + 1):
        beta = max(args.dagger_min_beta, 1.0 - (collection - 1) / max(args.dagger_warmup, 1))
        # Collect (no grad)
        actor.eval()
        # Multi-shape: pass env.slot_pair_dist (per-env [B,N,N]) directly so
        # compute_controller_action labels use each env's sampled shape.
        batch = collect_dagger(env, actor, env.slot_pair_dist, comm_radius,
                                args.k_spring, F_norm, args.rollout_steps, beta=beta)
        actor.train()

        # Train: chunks sequential, multi-epoch
        diag_agg = {"mse": 0.0, "cos_loss": 0.0, "mu_mean_abs": 0.0, "label_mean_abs": 0.0}
        n_steps = 0
        for _epoch in range(args.ppo_epochs):
            perm = torch.randperm(n_chunks).tolist()
            for c in perm:
                t0, t1 = chunk_starts[c], chunk_starts[c] + args.chunk_len
                opt.zero_grad()
                loss, diag = chunk_loss(actor, batch, t0, t1,
                                          alpha_cos=args.alpha_cos,
                                          low_speed_weight=args.low_speed_weight,
                                          speed_threshold=args.speed_threshold)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                opt.step()
                for k in diag_agg: diag_agg[k] += diag[k]
                n_steps += 1
        for k in diag_agg: diag_agg[k] /= max(n_steps, 1)

        log = (f"[col {collection:04d}/{args.total_collections}] "
               f"beta={beta:.2f} mse={diag_agg['mse']:.4f} cos_loss={diag_agg['cos_loss']:.4f} "
               f"mu_abs={diag_agg['mu_mean_abs']:.3f} lbl_abs={diag_agg['label_mean_abs']:.3f}")

        if collection % args.eval_every == 0 or collection == args.total_collections:
            actor.eval()
            ev = _quick_eval_selfshape(actor, cfg, templates,
                                         n_envs=args.eval_envs,
                                         episode_length=args.eval_T,
                                         device=str(device), seed=collection,
                                         sampled=False)
            actor.train()
            log += (f"  | eval succ@0.4d={ev['shape_succ']:.3f} "
                    f"succ@0.25d={ev['shape_succ_strict']:.3f} "
                    f"match={ev['shape_match_d']:.3f}d "
                    f"max={ev['shape_match_d_max']:.3f}d "
                    f"coll={ev['coll']:.0f}")
            per_shape = ev.get("per_shape", {})
            if len(per_shape) > 1:
                shape_str = " ".join(
                    f"{n}={m['shape_succ']:.2f}/{m['shape_succ_strict']:.2f}"
                    for n, m in per_shape.items()
                )
                log += "  [" + shape_str + "]"
            if ev["shape_succ"] > best_succ:
                best_succ = ev["shape_succ"]
                torch.save({
                    "arch_version": ARCH_VERSION,
                    "actor_type": str(cfg.get("actor_type", "gat")),
                    "n_agents": int(cfg.get("n_agents", env.N)),
                    "actor_state": actor.state_dict(),
                    "config": cfg,
                    "best_eval_shape_succ": best_succ,
                }, args.out_dir / "best.pt")
                log += f"  ↑ best.pt"

        if collection % args.save_every == 0:
            torch.save({
                "arch_version": ARCH_VERSION,
                "actor_type": str(cfg.get("actor_type", "gat")),
                "n_agents": int(cfg.get("n_agents", env.N)),
                "actor_state": actor.state_dict(),
                "config": cfg,
            }, args.out_dir / f"ckpt_{collection:04d}.pt")

        print(log, flush=True)

    print(f"[bc-dagger] done. best_succ@0.4d={best_succ:.3f}")


if __name__ == "__main__":
    main()
