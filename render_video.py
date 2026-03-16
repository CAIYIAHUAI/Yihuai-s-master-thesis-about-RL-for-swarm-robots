import argparse
import torch
import imageio
from pathlib import Path
from visualize import parse_args, _scenario_kwargs_from_ckpt, _infer_obs_dim_expected, Actor
from vmas import make_env
import sys
try:
    from scenarios.triangle_fill import Scenario
except ImportError:
    from VMAS.scenarios.triangle_fill import Scenario

def main():
    args = parse_args()
    args.device = "cpu"
    args.realtime = False
    
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    ckpt_args = ckpt.get("args", {}) or {}
    actor_hidden = int(ckpt_args.get("actor_hidden", 256))
    n_agents = int(ckpt_args.get("n_agents", 30))

    if args.max_episode_steps is None:
        args.max_episode_steps = int(ckpt_args.get("max_episode_steps", 1000))

    torch.manual_seed(args.seed)

    scenario_kwargs = _scenario_kwargs_from_ckpt(ckpt, args)

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

    actor = Actor(obs_dim=obs_dim, hidden=actor_hidden).to(env.device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

    frames = []
    print("Capturing frames...")

    for step in range(args.max_episode_steps):
        actions_for_env = []
        with torch.no_grad():
            for i in range(n_agents):
                logits = actor(obs_list[i])
                if args.deterministic:
                    a = torch.argmax(logits, dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                actions_for_env.append(a.unsqueeze(-1))

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

    print(f"Captured {len(frames)} frames. Saving to test_run.mp4...")
    imageio.mimsave("test_run.mp4", frames, fps=30)
    print("Done! test_run.mp4 saved.")

if __name__ == "__main__":
    main()
