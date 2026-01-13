import argparse
import os
import time
from typing import List, Optional

import torch

from vmas import make_env
from vmas.simulator.utils import save_video

from VMAS.scenarios.triangle_fill import Scenario


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout a random or fixed policy in triangle_fill.")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=float, default=None)
    p.add_argument("--video-path", default=None)
    p.add_argument("--no-random", action="store_true", help="Use a fixed action instead of random.")
    p.add_argument("--fixed-action", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print mean metrics every N steps (0 disables periodic printing).",
    )
    return p.parse_args()


def _mean_metrics(info0: dict) -> str:
    keys = ["inside_frac", "outside_mean", "collisions_mean", "cover_error"]
    parts = []
    for k in keys:
        v = info0.get(k, None)
        if isinstance(v, torch.Tensor) and v.numel() > 0:
            parts.append(f"{k}={v.float().mean().item():.4f}")
    return "  ".join(parts)


def main() -> None:
    args = _parse_args()

    if os.environ.get("TMPDIR") is None:
        print("WARNING: TMPDIR not set; if /tmp is unavailable, set TMPDIR=$PWD/.tmp before running.")

    torch.manual_seed(args.seed)

    scenario = Scenario()
    env = make_env(
        scenario=scenario,
        num_envs=args.num_envs,
        device=args.device,
        continuous_actions=False,
        dict_spaces=False,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    obs = env.reset()
    assert isinstance(obs, (list, tuple)), "dict_spaces=False should return a list/tuple of obs"

    frames: Optional[List] = [] if (args.render and args.save_video) else None

    start = time.time()
    last_info0 = None

    for step in range(args.steps):
        if args.no_random:
            actions = [
                torch.full((env.batch_dim, 1), args.fixed_action, device=env.device, dtype=torch.long)
                for _ in env.agents
            ]
        else:
            actions = [env.get_random_action(agent) for agent in env.agents]

        obs, rews, dones, infos = env.step(actions)

        info0 = infos[0]
        last_info0 = info0

        if step == 0:
            print("obs[0].shape:", obs[0].shape)
            print("rew[0].shape:", rews[0].shape)
            print("done.shape:", dones.shape)
            print("info keys:", list(info0.keys()))

        if args.print_every > 0 and (step + 1) % args.print_every == 0:
            print(f"step={step+1:5d}  mean_rew={rews[0].mean().item():+.4f}  " + _mean_metrics(info0))

        if args.render:
            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,
                visualize_when_rgb=True,
            )
            if frames is not None:
                frames.append(frame)

    elapsed = time.time() - start
    print(f"rollout finished: steps={args.steps} num_envs={args.num_envs} device={args.device} time={elapsed:.2f}s")
    if last_info0 is not None:
        print("final:", _mean_metrics(last_info0))

    if frames is not None and len(frames) > 0:
        fps = args.video_fps
        if fps is None:
            fps = 1.0 / env.scenario.world.dt
        if args.video_path is not None:
            video_name = args.video_path
            base, _ = os.path.splitext(video_name)
            save_video(base, frames, fps=int(round(fps)))
            print("saved video:", base + ".mp4")
        else:
            save_video("triangle_fill_rollout", frames, fps=int(round(fps)))
            print("saved video: triangle_fill_rollout.mp4")


if __name__ == "__main__":
    main()
