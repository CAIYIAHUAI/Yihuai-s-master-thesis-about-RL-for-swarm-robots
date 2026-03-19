import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.triangle_reward import (  # noqa: E402
    build_template_knn_signatures,
    build_triangle_template,
    per_agent_lattice_loss,
)


def test_build_template_knn_signatures_is_scale_invariant():
    tmpl = build_triangle_template(30, side_length=1.0)
    sig_a = build_template_knn_signatures(tmpl, k=6)
    sig_b = build_template_knn_signatures(tmpl * 0.045, k=6)

    assert sig_a.shape == (30, 6)
    assert torch.allclose(sig_a, sig_b, atol=1e-6, rtol=1e-6)


def test_per_agent_lattice_loss_prefers_template_structure():
    torch.manual_seed(0)
    tmpl = build_triangle_template(30, side_length=1.0)
    template_sigs = build_template_knn_signatures(tmpl, k=6)

    perfect = (tmpl * 0.045).unsqueeze(0)
    random_cloud = torch.randn_like(perfect) * 0.045

    perfect_loss = per_agent_lattice_loss(torch.cdist(perfect, perfect), template_sigs, k=6)
    random_loss = per_agent_lattice_loss(torch.cdist(random_cloud, random_cloud), template_sigs, k=6)

    assert perfect_loss.mean().item() < random_loss.mean().item()


def test_per_agent_progress_reward_uses_zero_delta_on_first_step():
    from vmas import make_env
    from scenarios.triangle_fill import Scenario

    torch.manual_seed(0)
    env = make_env(
        scenario=Scenario(),
        num_envs=4,
        device="cpu",
        continuous_actions=False,
        dict_spaces=False,
        max_steps=8,
        seed=0,
        n_agents=30,
        formation_w=0.25,
        spacing_w=2.0,
        lattice_w=1.0,
        lattice_k=6,
        progress_reward=True,
        share_reward=False,
        success_bonus=0.0,
        safe_collision_w=0.5,
        safe_action_w=0.0,
        formation_sinkhorn_iters=30,
        torch_compile=False,
        fast_collisions=False,
    )

    env.reset()
    stop_actions = [
        torch.zeros((env.batch_dim, 1), device=env.device, dtype=torch.long)
        for _ in env.agents
    ]
    _, rews, _, infos = env.step(stop_actions)

    scenario = env.scenario
    _, collision_pen, action_cost, _ = scenario._compute_metrics()
    rew_tensor = torch.stack(rews, dim=1)
    expected = -scenario.safe_collision_w * collision_pen - scenario.safe_action_w * action_cost

    assert torch.allclose(rew_tensor, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(infos[0]["local_spacing_progress_mean"], torch.zeros_like(infos[0]["formation_loss"]))
    assert torch.allclose(infos[0]["local_lattice_progress_mean"], torch.zeros_like(infos[0]["formation_loss"]))
    assert torch.allclose(infos[0]["global_shape_progress_mean"], torch.zeros_like(infos[0]["formation_loss"]))
