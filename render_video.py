import argparse
import torch
import imageio
from pathlib import Path
from vmas import make_env
try:
    from scenarios.triangle_fill import Scenario
except ImportError:
    from VMAS.scenarios.triangle_fill import Scenario
from models.actor import build_actor


def main():
    p = argparse.ArgumentParser(description="Render a video from a trained checkpoint.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--output", default="test_run.mp4", help="Output video path.")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    ckpt = torch.load(Path(args.ckpt), map_location=args.device)
    ckpt_args = ckpt.get("args", {}) or {}
    actor_hidden = int(ckpt_args.get("actor_hidden", 256))
    n_agents = int(ckpt_args.get("n_agents", 30))
    recurrent = bool(ckpt_args.get("recurrent", False))

    if args.max_episode_steps is None:
        args.max_episode_steps = int(ckpt_args.get("max_episode_steps", 1000))

    torch.manual_seed(args.seed)

    # Build scenario kwargs from checkpoint
    scenario_kwargs = {"n_agents": n_agents}
    for key in [
        "pile_center_mm", "pile_center_y_mm_range", "pile_halfwidth_mm",
        "obs_top_k_neighbors", "turn_v_frac", "normalize_obs",
        "formation_w", "formation_sinkhorn_tau", "formation_sinkhorn_iters",
        "formation_eps", "formation_template_seed",
        "safe_collision_w", "safe_action_w",
        "gnn_obs_self_only", "obs_include_goal_rel",
    ]:
        v = ckpt_args.get(key, None)
        if v is not None:
            scenario_kwargs[key] = v

    # Infer obs_dim for padding
    obs_dim_expected = ckpt_args.get("obs_dim", None)
    if obs_dim_expected is None:
        w = ckpt.get("actor_state", {}).get("fc_in.weight", None)
        if w is None:
            w = ckpt.get("actor_state", {}).get("net.0.weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            obs_dim_expected = int(w.shape[1])
    if obs_dim_expected is not None:
        scenario_kwargs["obs_pad_to_dim"] = int(obs_dim_expected)

    env = make_env(
        scenario=Scenario(),
        num_envs=1,
        device=args.device,
        continuous_actions=False,
        dict_spaces=False,
        max_steps=args.max_episode_steps,
        seed=args.seed,
        **scenario_kwargs,
    )

    obs_list = env.reset()
    obs_dim = int(obs_list[0].shape[-1])

    actor = build_actor(
        obs_dim=obs_dim,
        hidden=actor_hidden,
        recurrent=recurrent,
        gnn=bool(ckpt_args.get("gnn", False)),
        obs_top_k_neighbors=int(ckpt_args.get("obs_top_k_neighbors", 8)),
        gnn_hidden=int(ckpt_args.get("gnn_hidden", 64)),
        gnn_layers=int(ckpt_args.get("gnn_layers", 2)),
        gnn_radius=float(ckpt_args.get("gnn_radius", float(getattr(env.scenario, "comm_r", 0.0)))),
        gnn_top_k=int(ckpt_args.get("gnn_top_k", 0)),
        gnn_residual_init=float(ckpt_args.get("gnn_residual_init", 0.1)),
    ).to(env.device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

    hx = None
    if recurrent:
        hx = torch.zeros((1 * n_agents, actor_hidden), device=env.device)

    frames = []
    print("Capturing frames...")

    for step in range(args.max_episode_steps):
        with torch.no_grad():
            obs_all = torch.stack(obs_list, dim=1)  # [1, N, obs_dim]
            pos_all = None
            rot_all = None
            if actor.is_graph_actor:
                pos_all = torch.stack([a.state.pos for a in env.agents], dim=1)
                rot_all = torch.stack([a.state.rot for a in env.agents], dim=1)
            logits_all, hx = actor(obs_all, hx, pos_all, rot_all)
            logits_flat = logits_all.reshape(n_agents, -1)
            if args.deterministic:
                a_flat = torch.argmax(logits_flat, dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits_flat)
                a_flat = dist.sample()
            actions_for_env = [a_flat[i].unsqueeze(-1).unsqueeze(0) for i in range(n_agents)]

        obs_list, rews, dones, infos = env.step(actions_for_env)

        frame = env.render(
            mode="rgb_array",
            env_index=0,
            visualize_when_rgb=False,
        )
        if isinstance(frame, list) and len(frame) > 0:
            frame = frame[0]
        frames.append(frame)

        if bool(dones.item()):
            break

    print(f"Captured {len(frames)} frames. Saving to {args.output}...")
    imageio.mimsave(args.output, frames, fps=args.fps)
    print(f"Done! {args.output} saved.")


if __name__ == "__main__":
    main()
