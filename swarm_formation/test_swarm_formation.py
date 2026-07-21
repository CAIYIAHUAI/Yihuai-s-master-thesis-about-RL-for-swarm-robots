"""Sanity / smoke tests for swarm_formation.

Run from the package directory:
    pytest -x test_swarm_formation.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
def load_cfg() -> dict:
    return yaml.safe_load((REPO.parent / "configs" / "n10_triangle.yaml").read_text())


# ---------------------------------------------------------------------------
def test_pyg_imports_and_smoke():
    """PyG must be installed and a tiny GATv2Conv must run on the chosen device."""
    import torch_geometric  # noqa: F401
    from torch_geometric.nn import GATv2Conv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    conv = GATv2Conv(8, 8, heads=2, edge_dim=4, concat=False).to(dev)
    x = torch.randn(5, 8, device=dev)
    ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], device=dev)
    ea = torch.randn(5, 4, device=dev)
    y = conv(x, ei, ea)
    assert y.shape == (5, 8)


def test_t4_target_generation():
    from target import T4_N_AGENTS, make_t4_template
    p = make_t4_template(spacing=0.045)
    assert p.shape == (T4_N_AGENTS, 2)
    # Check row counts: r=0 -> 1, r=1 -> 2, r=2 -> 3, r=3 -> 4
    ys = p[:, 1]
    unique_y = torch.unique(ys)
    assert unique_y.numel() == 4
    # Side length: distance between (r=3, k=0) and (r=3, k=1) must equal d
    # row r=3 starts at index 6 (after 1+2+3) — find a pair in the bottom row.
    bottom_row = p[(ys == ys.min())]
    assert bottom_row.shape[0] == 4
    pair = (bottom_row[1] - bottom_row[0]).norm()
    assert torch.allclose(pair, torch.tensor(0.045), atol=1e-6)


def test_config_time_scale_asserts():
    from observations import assert_time_scales
    cfg = load_cfg()
    assert_time_scales(cfg)
    bad = dict(cfg); bad["forward_step"] = 0.99
    with pytest.raises(AssertionError):
        assert_time_scales(bad)


def test_density_target_centric_communicated():
    """rho[i, k] should sum messages from i and its comm-neighbors over targets."""
    from observations import _density, _self_mask
    cfg = load_cfg()
    # Two robots near a single target.
    pos = torch.tensor([[[0.0, 0.0], [0.01, 0.0]]])     # within comm_radius
    targets = torch.tensor([[[0.005, 0.0], [10.0, 10.0]]])
    claim = torch.zeros(1, 2, 2)
    not_self = ~_self_mask(2, "cpu").unsqueeze(0)
    d = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
    adj = (d < float(cfg["comm_radius"])) & not_self
    adj_with_self = adj | torch.eye(2, dtype=torch.bool).unsqueeze(0)
    rho, _m_self, _density_cutoff, _q_visible = _density(pos, targets, claim, adj_with_self, cfg)
    sigma = float(cfg["kernel_sigma"])
    self_msg_00 = math.exp(-((pos[0, 0] - targets[0, 0]).norm().item() ** 2)
                            / (2 * sigma * sigma))
    assert rho[0, 0, 0] > self_msg_00 - 1e-6


def test_density_cutoff():
    """A target outside density_radius produces zero density even if it is in Q_i."""
    from observations import _density
    cfg = load_cfg()
    far = float(cfg["density_radius"]) + 0.01
    pos = torch.tensor([[[0.0, 0.0]]])
    targets = torch.tensor([[[far, 0.0]]])
    claim = torch.zeros(1, 1, 1)
    adj_with_self = torch.eye(1, dtype=torch.bool).unsqueeze(0)
    rho, _m_self, density_cutoff, q_visible = _density(pos, targets, claim, adj_with_self, cfg)
    assert rho.item() == 0.0
    assert not density_cutoff.any()
    # but Q_i still includes this target because target_sense_radius > density_radius
    assert q_visible.any()


def test_target_sense_radius_separated_from_density_radius():
    """Lock down: a target between density_radius and target_sense_radius is in Q_i
    but contributes zero density. This was the user-flagged invariant."""
    from observations import _density
    cfg = load_cfg()
    assert float(cfg["target_sense_radius"]) > float(cfg["density_radius"]), \
        "config: target_sense_radius must be > density_radius"
    mid = 0.5 * (float(cfg["density_radius"]) + float(cfg["target_sense_radius"]))
    pos = torch.tensor([[[0.0, 0.0]]])
    targets = torch.tensor([[[mid, 0.0]]])
    claim = torch.zeros(1, 1, 1)
    adj_with_self = torch.eye(1, dtype=torch.bool).unsqueeze(0)
    rho, _m_self, density_cutoff, q_visible = _density(pos, targets, claim, adj_with_self, cfg)
    assert q_visible.item()
    assert not density_cutoff.item()
    assert rho.item() == 0.0


def test_low_competition_target_attracts_robot():
    """Two robots in the comm graph; one is far from target A and close to target B.
    The other is close to A and far from B. Each robot's v_rho should point toward
    its respective unclaimed target — confirms the share^p competitive weight."""
    from observations import expert_step, ExpertState
    cfg = load_cfg()
    cfg = dict(cfg); cfg["expert_velocity_noise"] = 0.0
    # robot 0 close to target 0, robot 1 close to target 1; both within comm_radius
    pos = torch.tensor([[[0.0, 0.0], [0.04, 0.0]]])
    rot = torch.zeros(1, 2)
    last_v = torch.zeros(1, 2, 2)
    last_a = torch.zeros(1, 2, dtype=torch.long)
    targets = torch.tensor([[[0.01, 0.0], [0.05, 0.0]]])
    state = ExpertState.zeros(1, 2, 2)
    stuck = torch.zeros(1, 2, dtype=torch.long)
    out = expert_step(pos, rot, last_v, last_a, targets, state, cfg, stuck,
                      noise_gen=None, build_obs=False)
    # Robot 0 should be pulled toward target 0 (positive x direction).
    assert out.velocity[0, 0, 0] > 0
    # Robot 1 should also be pulled toward target 1 (positive x direction).
    assert out.velocity[0, 1, 0] > 0
    # Without competitive selection both robots would be pulled to the centroid
    # (between target 0 and target 1), so robot 0 would head toward x≈0.03, but with
    # share^p robot 0 owns target 0 and primarily moves there; robot 1 owns target 1.
    # Their resultant velocity x-components must therefore differ when noise=0.
    assert not torch.allclose(out.velocity[0, 0], out.velocity[0, 1])


def test_hop_gradient_sign():
    """Lock the hop-gradient sign: it must point AWAY from blocked sources."""
    from observations import expert_step, ExpertState
    cfg = load_cfg()
    cfg = dict(cfg); cfg["expert_velocity_noise"] = 0.0
    # 3 robots in a line. Spacing chosen so 0—1 and 1—2 are within comm_radius (0.135)
    # but 0—2 is NOT (so robot 2 must reach blocked source 0 via robot 1).
    # Hop counts: 0=0, 1=1, 2=2.  Escape direction at robot 2 should point in +x.
    pos = torch.tensor([[[0.0, 0.0], [0.10, 0.0], [0.20, 0.0]]])
    rot = torch.zeros(1, 3)
    last_v = torch.zeros(1, 3, 2)            # below blocked threshold
    last_a = torch.zeros(1, 3, dtype=torch.long)
    targets = torch.tensor([[[0.0, 1.0], [0.0, 1.5], [0.0, 2.0]]])  # all outside Q_i
    state = ExpertState.zeros(1, 3, 3)
    state.blocked_steps[0, 0] = int(cfg["blocked_steps_threshold"])  # robot 0 already blocked
    stuck = torch.zeros(1, 3, dtype=torch.long)
    out = expert_step(pos, rot, last_v, last_a, targets, state, cfg, stuck,
                      noise_gen=None, build_obs=False)
    # Robot 2 (highest hop) should have hop_gradient pointing in +x (escape direction);
    # body frame is identity since rotation=0, so x-component > 0.
    assert out.hop_count[0, 0] == 0
    assert out.hop_count[0, 1] == 1
    assert out.hop_count[0, 2] == 2
    # The body-frame x-component (forward) of hop_gradient at robot 2 should be > 0.
    assert out.hop_gradient[0, 2, 0] > 0.5


def test_claim_hysteresis():
    """Inside r_claim => increment; outside r_release => zero; in-between => decay."""
    from observations import _update_claim
    cfg = load_cfg()
    prev = torch.tensor([[[3.0, 0.0]]])           # one robot, two targets
    d = torch.tensor([[[float(cfg["r_claim"]) - 0.001,
                         float(cfg["r_release"]) + 0.001]]])
    new = _update_claim(prev, d, cfg)
    assert new[0, 0, 0] > prev[0, 0, 0]            # in-band increment
    assert new[0, 0, 1] == 0.0                     # released

    # In-between: between r_claim and r_release => decay
    prev = torch.tensor([[[5.0]]])
    d = torch.tensor([[[(float(cfg["r_claim"]) + float(cfg["r_release"])) / 2.0]]])
    new = _update_claim(prev, d, cfg)
    assert new[0, 0, 0] == 5.0 * float(cfg["claim_decay"])


def test_symmetry_break_noise_determinism():
    """Same seed → same expert output. Different seed → different output."""
    from observations import expert_step, ExpertState
    cfg = load_cfg()
    cfg = dict(cfg); cfg["expert_velocity_noise"] = 0.001
    pos = torch.tensor([[[0.0, 0.0], [0.0, 0.005], [0.005, 0.0],
                          [0.005, 0.005], [0.01, 0.0], [0.01, 0.005],
                          [0.01, 0.01], [0.005, 0.01], [0.0, 0.01], [0.015, 0.0]]])
    rot = torch.zeros(1, 10)
    last_v = torch.zeros(1, 10, 2)
    last_a = torch.zeros(1, 10, dtype=torch.long)
    targets = torch.tensor([[[0.0, 0.0], [0.045, 0.0], [-0.045, 0.0],
                               [0.0, -0.039], [-0.022, -0.039], [0.022, -0.039],
                               [0.0, -0.078], [-0.045, -0.078], [0.045, -0.078],
                               [-0.022, -0.039]]])
    state = ExpertState.zeros(1, 10, 10)
    stuck = torch.zeros(1, 10, dtype=torch.long)

    g1 = torch.Generator(); g1.manual_seed(7)
    out1 = expert_step(pos, rot, last_v, last_a, targets, state, cfg, stuck,
                       noise_gen=g1, build_obs=False)
    g2 = torch.Generator(); g2.manual_seed(7)
    out2 = expert_step(pos, rot, last_v, last_a, targets, state, cfg, stuck,
                       noise_gen=g2, build_obs=False)
    assert torch.allclose(out1.velocity, out2.velocity)

    g3 = torch.Generator(); g3.manual_seed(8)
    out3 = expert_step(pos, rot, last_v, last_a, targets, state, cfg, stuck,
                       noise_gen=g3, build_obs=False)
    assert not torch.allclose(out1.velocity, out3.velocity)


def test_blocked_escape_velocity_nonzero():
    """A blocked robot in a tight cluster should get a nonzero escape velocity."""
    from observations import expert_step, ExpertState
    cfg = load_cfg()
    cfg = dict(cfg); cfg["expert_velocity_noise"] = 0.0
    # Pile 10 robots together; previous velocity ~ 0 so they're all "slow".
    pos = torch.zeros(1, 10, 2)
    pos[0, :, 0] = torch.linspace(0, 0.001, 10)
    rot = torch.zeros(1, 10)
    last_v = torch.zeros(1, 10, 2)              # zero -> below blocked_speed_threshold
    last_a = torch.zeros(1, 10, dtype=torch.long)
    # targets spread out so escape direction is well-defined
    targets = torch.tensor([[[0.0, -0.045 * r] for r in range(10)]])
    # Pre-fill blocked_steps so robots are already considered blocked.
    state = ExpertState.zeros(1, 10, 10)
    state.blocked_steps += int(cfg["blocked_steps_threshold"])
    stuck = torch.zeros(1, 10, dtype=torch.long)
    out = expert_step(pos, rot, last_v, last_a, targets, state, cfg, stuck,
                      noise_gen=None, build_obs=False)
    assert out.blocked_flag.all()
    speeds = out.velocity.norm(dim=-1)
    assert (speeds > 0.0).all()


def test_hop_count_bfs():
    """Linear chain: hop counts equal BFS distance from the source."""
    from observations import _hop_count
    N = 5
    adj = torch.zeros(1, N, N, dtype=torch.bool)
    for i in range(N - 1):
        adj[0, i, i + 1] = True
        adj[0, i + 1, i] = True
    sources = torch.zeros(1, N, dtype=torch.bool); sources[0, 0] = True
    hop = _hop_count(adj, sources, hop_max=10)
    assert torch.equal(hop, torch.tensor([[0, 1, 2, 3, 4]]))


def test_qi_empty_uses_shape_centroid_guidance():
    """If no targets are within density_radius, expert should pull toward shape centroid."""
    from observations import expert_step, ExpertState
    cfg = load_cfg()
    cfg = dict(cfg); cfg["expert_velocity_noise"] = 0.0
    # Single robot far from any target.
    pos = torch.tensor([[[1.0, 0.0]]])
    rot = torch.tensor([[0.0]])
    last_v = torch.zeros(1, 1, 2)
    last_a = torch.zeros(1, 1, dtype=torch.long)
    targets = torch.tensor([[[0.0, 0.0], [0.045, 0.0]]])
    state = ExpertState.zeros(1, 1, 2)
    stuck = torch.zeros(1, 1, dtype=torch.long)
    out = expert_step(pos, rot, last_v, last_a, targets, state, cfg, stuck,
                      noise_gen=None, build_obs=False)
    # No target visible => v_ρ ≈ 0; v_a points toward shape_centroid (negative-x).
    assert out.velocity[0, 0, 0] < 0.0


def test_velocity_to_action_directions():
    """Action mapping respects body-frame heading_error sign."""
    from observations import velocity_to_action, ACT_LEFT, ACT_RIGHT, ACT_FORWARD, ACT_STOP
    cfg = load_cfg()
    rot = torch.zeros(1, 4)
    # vel: [forward, left, right, stop]
    vel = torch.tensor([
        [[1.0, 0.0],
         [0.0, 1.0],
         [0.0, -1.0],
         [0.0, 0.0]]
    ]) * float(cfg["v_max"])
    dist_nearest = torch.zeros(1, 4)        # near => use near threshold
    action, _, _ = velocity_to_action(vel, rot, dist_nearest, cfg)
    assert action[0, 0].item() == ACT_FORWARD
    assert action[0, 1].item() == ACT_LEFT
    assert action[0, 2].item() == ACT_RIGHT
    assert action[0, 3].item() == ACT_STOP


def test_label_margin_lowers_weight():
    """Heading error close to the threshold should be down-weighted."""
    from observations import label_sample_weight, ACT_LEFT
    cfg = load_cfg()
    margin = float(cfg["label_margin"])
    threshold = torch.tensor([[0.5]])
    he = torch.tensor([[0.5 + 0.5 * margin]])     # within margin band
    a = torch.tensor([[ACT_LEFT]])
    w = label_sample_weight(he, threshold, a, cfg)
    assert w[0, 0].item() < 1.0


def test_local_stop_proxy():
    """Robot near a target with stable claim and zero velocity → STOP candidate."""
    from observations import local_stop_proxy
    cfg = load_cfg()
    pos = torch.tensor([[[0.0, 0.0]]])
    targets = torch.tensor([[[0.001, 0.0]]])     # well within arrival_radius
    claim = torch.full((1, 1, 1), float(cfg["claim_steps"]))
    vel = torch.zeros(1, 1, 2)
    min_neighbor = torch.tensor([[1.0]])
    s = local_stop_proxy(pos, targets, claim, vel, min_neighbor, cfg)
    assert s[0, 0]


def test_pyg_graph_batching_shape():
    """build_edges must produce PyG-batched (edge_index, edge_attr) of correct shape."""
    from observations import build_edges, EDGE_FEATURE_DIM
    cfg = load_cfg()
    B, N = 3, 4
    pos = torch.randn(B, N, 2) * 0.01
    rot = torch.zeros(B, N)
    last_v = torch.zeros(B, N, 2)
    not_self = ~torch.eye(N, dtype=torch.bool).unsqueeze(0)
    d = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
    adj = (d < float(cfg["comm_radius"])) & not_self
    ei, ea = build_edges(pos, rot, last_v, adj, cfg)
    E_expected = int(adj.sum().item())
    assert ei.shape == (2, E_expected)
    assert ea.shape == (E_expected, EDGE_FEATURE_DIM)
    # All node indices should be in [0, B*N).
    assert int(ei.max().item()) < B * N
    assert int(ei.min().item()) >= 0


def test_gat_gru_forward_shape():
    from observations import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, build_edges
    from model import GATGRUActor
    cfg = load_cfg()
    B, N = 2, 5
    pos = torch.randn(B, N, 2) * 0.01
    rot = torch.zeros(B, N)
    last_v = torch.zeros(B, N, 2)
    not_self = ~torch.eye(N, dtype=torch.bool).unsqueeze(0)
    adj = (((pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)) < float(cfg["comm_radius"])) & not_self
    ei, ea = build_edges(pos, rot, last_v, adj, cfg)
    obs = torch.randn(B, N, NODE_FEATURE_DIM)
    actor = GATGRUActor(node_dim=NODE_FEATURE_DIM, edge_dim=EDGE_FEATURE_DIM,
                        gat_hidden=16, gat_heads=2, gru_hidden=32)
    h0 = torch.zeros(B, N, 32)
    logits, vel_aux, h1 = actor(obs, ei, ea, h0)
    assert logits.shape == (B, N, 4)
    assert vel_aux.shape == (B, N, 2)
    assert h1.shape == (B, N, 32)


def test_hidden_reset_on_done():
    from model import reset_hidden_for_done
    h = torch.ones(3, 4, 5)
    done = torch.tensor([True, False, True])
    h2 = reset_hidden_for_done(h, done)
    assert (h2[0] == 0).all()
    assert (h2[1] == 1).all()
    assert (h2[2] == 0).all()


def test_actor_locality_no_global_signals():
    """Node features must have shape NODE_FEATURE_DIM and contain no goal/slot/GW."""
    from observations import NODE_FEATURE_DIM, build_node_features
    cfg = load_cfg()
    B, N = 1, 3
    rot = torch.zeros(B, N)
    last_v = torch.zeros(B, N, 2)
    last_a = torch.zeros(B, N, dtype=torch.long)
    stuck = torch.zeros(B, N, dtype=torch.long)
    blocked = torch.zeros(B, N, dtype=torch.bool)
    hop = torch.zeros(B, N, dtype=torch.long)
    min_nb = torch.full((B, N), 0.05)
    coll_risk = torch.zeros(B, N, dtype=torch.long)
    nodes = build_node_features(
        rotations=rot, last_velocity=last_v, last_action=last_a,
        stuck_counter=stuck, blocked_flag=blocked, hop=hop,
        min_neighbor_dist=min_nb, collision_risk=coll_risk, cfg=cfg,
    )
    assert nodes.shape == (B, N, NODE_FEATURE_DIM)


def test_compute_matching_returns_per_robot_mean():
    """compute_matching must return *mean* per-robot distance (not sum); locks the
    semantics that thresholds like '< 0.4 d' are per-robot."""
    from target import compute_matching
    # 2 robots, 2 targets: robot 0 at (1,0) → target 0 at (0,0): dist 1.
    #                       robot 1 at (3,0) → target 1 at (5,0): dist 2.
    # Mean per-robot: (1 + 2) / 2 = 1.5. Sum would be 3.
    pos = torch.tensor([[[1.0, 0.0], [3.0, 0.0]]])
    targets = torch.tensor([[[0.0, 0.0], [5.0, 0.0]]])
    perm, dist = compute_matching(pos, targets)
    assert dist.shape == (1,)
    assert math.isclose(float(dist[0]), 1.5, abs_tol=1e-5), \
        f"expected mean per-robot 1.5, got {float(dist[0])}"
    assert perm.tolist() == [[0, 1]]


def test_expert_rollout_improves_matching_and_coverage():
    """Expert rollout must measurably improve formation quality, not just be finite.
    Compares the matching distance and coverage at episode end vs at episode start."""
    from environment import MeanShiftEnv
    from target import compute_coverage, compute_matching
    cfg = load_cfg()
    env = MeanShiftEnv(cfg, n_envs=8, device="cpu", seed=0)
    noise_gen = torch.Generator(device=env.device); noise_gen.manual_seed(1)
    _, m0 = compute_matching(env.positions, env.target_map)
    c0 = compute_coverage(env.positions, env.target_map,
                          radius=float(cfg["arrival_radius"]))
    for _ in range(150):
        out = env.expert_action(noise_gen=noise_gen, build_obs=False)
        env.step(out.action)
    _, m1 = compute_matching(env.positions, env.target_map)
    c1 = compute_coverage(env.positions, env.target_map,
                          radius=float(cfg["arrival_radius"]))
    # Strict improvement: matching distance must drop by at least 3×, coverage must
    # rise meaningfully. Catches future regressions in the expert silently.
    assert float(m1.mean()) < float(m0.mean()) / 3.0, \
        f"expert did not converge: m0={float(m0.mean()):.4f} m1={float(m1.mean()):.4f}"
    assert float(c1.mean()) > float(c0.mean()) + 0.4, \
        f"expert did not raise coverage: c0={float(c0.mean()):.2f} c1={float(c1.mean()):.2f}"


def test_bc_tiny_overfit():
    """Tiny-data BC should drive policy CE accuracy ≥ 0.95 on the overfit set."""
    from observations import EDGE_FEATURE_DIM, NODE_FEATURE_DIM, N_ACTIONS
    from model import GATGRUActor
    from train_mappo import reconstruct_pyg

    torch.manual_seed(0)
    cfg = load_cfg()
    B, N, T = 2, 5, 8
    F_node, F_edge = NODE_FEATURE_DIM, EDGE_FEATURE_DIM
    obs_nodes = torch.randn(B, T, N, F_node)
    edge_attr_full = torch.randn(B, T, N, N, F_edge)
    edge_mask = torch.zeros(B, T, N, N, dtype=torch.bool)
    # Sparse, asymmetric mask
    edge_mask[..., 0, 1] = True
    edge_mask[..., 1, 2] = True
    edge_mask[..., 2, 0] = True
    expert_action = torch.randint(0, N_ACTIONS, (B, T, N))
    sample_weight = torch.ones(B, T, N)
    dones = torch.zeros(B, T, dtype=torch.bool)

    actor = GATGRUActor(node_dim=F_node, edge_dim=F_edge,
                        gat_hidden=32, gat_heads=2, gru_hidden=32)
    opt = torch.optim.AdamW(actor.parameters(), lr=3e-3)
    for step in range(120):
        h = torch.zeros(B, N, actor.hidden)
        loss = 0.0
        for t in range(T):
            ei, ea = reconstruct_pyg(edge_mask[:, t], edge_attr_full[:, t])
            logits, _v, h_next = actor(obs_nodes[:, t], ei, ea, h)
            loss = loss + torch.nn.functional.cross_entropy(
                logits.reshape(-1, N_ACTIONS), expert_action[:, t].reshape(-1),
            )
            h = h_next
        loss = loss / T
        opt.zero_grad()
        loss.backward()
        opt.step()
    # Now check accuracy
    with torch.no_grad():
        h = torch.zeros(B, N, actor.hidden)
        correct = 0; total = 0
        for t in range(T):
            ei, ea = reconstruct_pyg(edge_mask[:, t], edge_attr_full[:, t])
            logits, _v, h_next = actor(obs_nodes[:, t], ei, ea, h)
            pred = logits.argmax(dim=-1)
            correct += int((pred == expert_action[:, t]).sum().item())
            total += pred.numel()
            h = h_next
        acc = correct / total
    assert acc >= 0.95, f"tiny overfit only reached {acc:.3f}"


def test_mappo_2_update_smoke():
    """Two MAPPO updates with --smoke should run without errors."""
    import subprocess
    import sys as _sys
    cmd = [
        _sys.executable, str(REPO / "train_mappo.py"),
        "--smoke",
        "--out_dir", tempfile.mkdtemp(prefix="ms_mappo_"),
        "--device", "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"mappo smoke failed:\nstdout={res.stdout}\nstderr={res.stderr}"


def test_random_target_disc_bounds():
    """random_disc target centroids must stay inside target_offset_max."""
    from environment import MeanShiftEnv
    cfg = load_cfg()
    cfg = dict(cfg)
    cfg["target_center_mode"] = "random_disc"
    cfg["target_offset_max"] = 0.135
    env = MeanShiftEnv(cfg, n_envs=64, device="cpu", seed=0)
    target_centroid = env.target_map.mean(dim=1)
    radius = target_centroid.norm(dim=-1)
    assert (radius <= float(cfg["target_offset_max"]) + 1e-6).all()


def test_random_theta_per_env():
    """target_theta_random samples independent target orientations per env."""
    from environment import MeanShiftEnv
    cfg = load_cfg()
    cfg = dict(cfg)
    cfg["target_center_mode"] = "origin"
    cfg["target_theta_random"] = True
    env = MeanShiftEnv(cfg, n_envs=8, device="cpu", seed=0)
    centered = env.target_map - env.target_map.mean(dim=1, keepdim=True)
    reference = centered[0].unsqueeze(0).expand_as(centered[1:])
    max_delta = (centered[1:] - reference).abs().amax(dim=(1, 2))
    assert (max_delta > 1e-4).any()


def test_robot_origin_jitter_bounded():
    """origin_jitter keeps a small fixed spawn blob without initial hard collisions."""
    from environment import MeanShiftEnv
    cfg = load_cfg()
    cfg = dict(cfg)
    cfg["robot_init_mode"] = "origin_jitter"
    cfg["robot_init_jitter"] = 0.015
    env = MeanShiftEnv(cfg, n_envs=16, device="cpu", seed=0)
    max_radius = env.positions.norm(dim=-1).max()
    assert float(max_radius) <= 5.0 * float(cfg["robot_init_jitter"])
    d_ij = (env.positions.unsqueeze(2) - env.positions.unsqueeze(1)).norm(dim=-1)
    not_self = ~torch.eye(env.N, dtype=torch.bool).unsqueeze(0)
    min_pair = d_ij[not_self.expand_as(d_ij)].min()
    assert float(min_pair) >= float(cfg["collision_hard"])


def test_expert_handles_random_target():
    """Expert should still solve most random target + theta episodes."""
    from environment import MeanShiftEnv
    cfg = load_cfg()
    cfg = dict(cfg)
    cfg["target_center_mode"] = "random_disc"
    cfg["target_offset_max"] = 0.135
    cfg["target_theta_random"] = True
    cfg["robot_init_mode"] = "origin_jitter"
    cfg["episode_length"] = 100
    env = MeanShiftEnv(cfg, n_envs=8, device="cpu", seed=1)
    noise_gen = torch.Generator(device=env.device); noise_gen.manual_seed(2)
    info = None
    for _ in range(100):
        out = env.expert_action(noise_gen=noise_gen, build_obs=False)
        _, info, _done = env.step(out.action)
    assert info is not None
    succ = float(info.success.float().mean().item())
    assert succ >= 0.7, f"random target expert sanity succ={succ:.2f}"


# ===========================================================================
# Self-shape mode tests (R1-R17 from plan / 诊断补充)
# ===========================================================================
def _make_perfect_t4_positions(spacing: float = 0.045, B: int = 4):
    from shape_metrics import make_centered_t4_template
    template = make_centered_t4_template(spacing)
    return template.unsqueeze(0).expand(B, -1, -1).clone(), template


# --- R4: shape metrics ----------------------------------------------------
def test_shape_profile_perfect_t4_zero():
    """Perfect T4 (any translation/rotation/permutation) → profile_error ≈ 0."""
    from shape_metrics import shape_profile_error, make_centered_t4_template
    spacing = 0.045
    template = make_centered_t4_template(spacing)
    perfect = template.unsqueeze(0)
    trans = (template + torch.tensor([0.5, -0.3])).unsqueeze(0)
    theta = math.pi / 5
    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
    rot = (R @ template.T).T.unsqueeze(0)
    perm = template[torch.randperm(10)].unsqueeze(0)
    for pos in (perfect, trans, rot, perm):
        err = shape_profile_error(pos, template).item()
        assert err < 1e-4, f"perfect T4 profile error {err} > 1e-4"


def test_shape_profile_random_blob_nonzero():
    from shape_metrics import shape_profile_error, make_centered_t4_template
    template = make_centered_t4_template(0.045)
    torch.manual_seed(0)
    blob = torch.randn(1, 10, 2) * 0.05
    assert shape_profile_error(blob, template).item() > 0.05


def test_isospectral_disambiguation():
    """R4: combined sorted-distance + Laplacian-eigvalsh metric must reject
    deformations that one term alone might miss. Squeezed T4 fails both terms.
    """
    from shape_metrics import shape_profile_error, make_centered_t4_template
    template = make_centered_t4_template(0.045)
    perm = torch.randperm(10)
    iso_positions = template[perm].unsqueeze(0)
    err_iso = shape_profile_error(iso_positions, template, alpha=0.5).item()
    assert err_iso < 1e-4
    squashed = (template * 0.7).unsqueeze(0)
    err_squashed = shape_profile_error(squashed, template, alpha=0.5).item()
    assert err_squashed > 0.1, f"squashed should give large err, got {err_squashed}"


# --- R6: comm_radius covers T4 diameter -----------------------------------
def test_comm_radius_covers_t4_diameter():
    """R6: with bumped comm_radius (4d=0.180), perfect T4 must be all-connected."""
    from shape_metrics import isolated_agent_count, make_centered_t4_template
    cfg = load_cfg()
    spacing = float(cfg["target_spacing"])
    comm_radius = float(cfg["comm_radius"])
    assert comm_radius > 3.0 * spacing, \
        f"comm_radius {comm_radius} must exceed 3d={3*spacing} for T4 diameter"
    template = make_centered_t4_template(spacing)
    positions = template.unsqueeze(0)
    iso = isolated_agent_count(positions, comm_radius=comm_radius, min_neighbors=1)
    assert iso.item() == 0


def test_connectivity_penalty_local():
    """R6b: isolated_agent_count is per-agent."""
    from shape_metrics import isolated_agent_count
    pos = torch.tensor([[[10.0, 10.0]] + [[0.0, 0.0]] * 9])
    iso = isolated_agent_count(pos, comm_radius=0.18, min_neighbors=1)
    assert iso.item() == 1.0


# --- R7+R15: edge raw z_src -----------------------------------------------
def test_edge_feature_includes_raw_z_src():
    """R7+R15: edge attr last 4 dims are the source agent's raw z (NOT diff)."""
    from observations import build_edges, EDGE_FEATURE_DIM, EDGE_FEATURE_DIM_SELFSHAPE
    from environment import MeanShiftEnv
    cfg = load_cfg()
    env = MeanShiftEnv(cfg, n_envs=2, device="cpu", seed=0)
    torch.manual_seed(0)
    z = torch.randn(2, 10, 4)
    adj = ((env.positions.unsqueeze(2) - env.positions.unsqueeze(1)).norm(dim=-1)
           < float(cfg["comm_radius"]))
    adj = adj & ~torch.eye(10, dtype=torch.bool).unsqueeze(0).expand(2, -1, -1)
    ei9, ea9 = build_edges(env.positions, env.rotations, env.velocity, adj, cfg, latent_z=None)
    ei13, ea13 = build_edges(env.positions, env.rotations, env.velocity, adj, cfg, latent_z=z)
    assert ea9.shape[-1] == EDGE_FEATURE_DIM == 9
    # Without slot info, build_edges appends only raw z_src (4 dims) → 13-dim.
    # v17b' added d_desired as an OPTIONAL 14th dim when slot info is also provided
    # (EDGE_FEATURE_DIM_SELFSHAPE = 14 is the maximum with all features).
    assert ea13.shape[-1] == 13, "edges with latent_z only should be 13-dim (no d_desired)"
    assert EDGE_FEATURE_DIM_SELFSHAPE == 14, "constant tracks maximum (geom + z_src + d_desired)"
    assert torch.allclose(ea9, ea13[:, :9])
    nz = adj.nonzero(as_tuple=False)
    b0, src0, dst0 = nz[0].tolist()
    assert torch.allclose(ea13[0, 9:13], z[b0, src0])
    diff = z[b0, src0] - z[b0, dst0]
    assert not torch.allclose(ea13[0, 9:13], diff)
    # v17b': when slot info is also provided, output grows to 14
    slot_assign = torch.zeros(2, 10, dtype=torch.long)
    slot_assign[:] = torch.arange(10)
    slot_pd = torch.rand(10, 10) * 0.05
    _, ea14 = build_edges(env.positions, env.rotations, env.velocity, adj, cfg,
                           latent_z=z, slot_assignment=slot_assign, slot_pair_dist=slot_pd)
    assert ea14.shape[-1] == 14
    assert torch.allclose(ea14[:, :13], ea13)


# --- NODE_FEATURE_DIM layout (latent appended after the base block) -------
def test_node_features_layout_append_at_end():
    from observations import (build_node_features, NODE_FEATURE_DIM,
                        NODE_FEATURE_DIM_SELFSHAPE, LATENT_Z_DIM)
    from environment import MeanShiftEnv
    cfg = load_cfg()
    env = MeanShiftEnv(cfg, n_envs=2, device="cpu", seed=0)
    out = env.expert_action(noise_gen=None, build_obs=False)
    z = torch.randn(2, 10, LATENT_Z_DIM)
    common_kw = dict(
        rotations=env.rotations,
        last_velocity=env.velocity, last_action=env.last_action,
        stuck_counter=env.stuck_counter, blocked_flag=out.blocked_flag,
        hop=out.hop_count,
        min_neighbor_dist=torch.full((2, 10), 0.05),
        collision_risk=out.collision_risk, cfg=cfg,
    )
    base = build_node_features(latent_z=None, **common_kw)
    new = build_node_features(latent_z=z, **common_kw)
    assert base.shape[-1] == NODE_FEATURE_DIM == 8
    assert NODE_FEATURE_DIM_SELFSHAPE == 22          # 8 base + 4 latent + 10 slot (N=10)
    assert new.shape[-1] == NODE_FEATURE_DIM + LATENT_Z_DIM   # base + latent, no slot
    assert torch.allclose(base, new[..., :NODE_FEATURE_DIM])
    assert torch.allclose(new[..., NODE_FEATURE_DIM:NODE_FEATURE_DIM + LATENT_Z_DIM], z)


# --- R1: per-episode latent z (no robot ID across episodes) ---------------
def test_latent_z_resampled_each_episode_constant_within():
    from environment import MeanShiftEnv
    cfg = load_cfg()
    env = MeanShiftEnv(cfg, n_envs=8, device="cpu", seed=42)
    z0 = env.latent_z.clone()
    for _ in range(5):
        env.step(torch.zeros(8, 10, dtype=torch.long))
    assert torch.allclose(env.latent_z, z0)
    env.reset_all()
    assert not torch.allclose(env.latent_z, z0)


# --- R3: continuous curriculum + self-paced safeguard ---------------------
def test_curriculum_progress_continuous():
    from train_mappo import compute_curriculum
    cfg = load_cfg()
    cfg.setdefault("curriculum", {}).update({"T_warmup": 200, "T_total": 1500})
    schedules = []
    for u in range(0, 1600, 50):
        s = compute_curriculum(u, T_warmup=200, T_total=1500, cfg=cfg)
        schedules.append(s)
    for k, expect_increasing in [("s", True), ("shape_w", True),
                                 ("cue_prob", False), ("target_w", False),
                                 ("kl_old_w", False)]:
        vals = [sd[k] for sd in schedules]
        for a, b in zip(vals, vals[1:]):
            if expect_increasing:
                assert b >= a - 1e-9, f"{k} not monotonic increasing: {a} → {b}"
            else:
                assert b <= a + 1e-9, f"{k} not monotonic decreasing: {a} → {b}"
    s0 = compute_curriculum(0, 200, 1500, cfg)
    s_end = compute_curriculum(1500, 200, 1500, cfg)
    assert s0["s"] == 0 and s0["cue_prob"] == 1.0 and s0["shape_w"] == 0
    # FIX 4: s_max may cap final s below 1.0 (default 0.5 in current config).
    s_max = float(cfg.get("curriculum", {}).get("s_max", 1.0))
    assert s_end["s"] == s_max


# --- R11: critic features keep target-aware dims --------------------------
def test_critic_features_target_aware_in_warmup():
    from train_mappo import (build_critic_features, build_critic_features_selfshape,
                              CRITIC_FEATURE_DIM, CRITIC_FEATURE_DIM_SELFSHAPE)
    from environment import MeanShiftEnv
    from shape_metrics import make_centered_t4_template
    cfg = load_cfg()
    env = MeanShiftEnv(cfg, n_envs=2, device="cpu", seed=0)
    template = make_centered_t4_template(env.spacing)
    prof, _, iso, bp = env.compute_shape_metrics(template)
    z = env.latent_z
    base_feats, _ = build_critic_features(env, t_global=10, episode_length=300)
    new_feats, _ = build_critic_features_selfshape(
        env, t_global=10, episode_length=300, latent_z=z,
        profile_error=prof, isolated_count=iso, bond_penalty=bp,
    )
    assert base_feats.shape[-1] == CRITIC_FEATURE_DIM == 12
    assert new_feats.shape[-1] == CRITIC_FEATURE_DIM_SELFSHAPE == 20
    assert torch.allclose(new_feats[..., :12], base_feats)
    assert torch.allclose(new_feats[..., 12:16], z)


def test_v12_rank_slot_assignment():
    """Rank-based slot routing yields a valid permutation per env."""
    from environment import MeanShiftEnv, load_config
    cfg = load_config(REPO.parent / "configs" / "n10_triangle.yaml")
    env = MeanShiftEnv(cfg, n_envs=8, device="cpu", seed=0)
    sa = env.slot_assignment                       # [8, 10] long
    assert sa.shape == (8, 10)
    # Each row must be a permutation of [0, 1, ..., 9]
    for b in range(8):
        sorted_b = sa[b].sort().values
        assert torch.equal(sorted_b, torch.arange(10)), \
            f"env {b} slot_assignment is not a valid permutation: {sa[b].tolist()}"


# --- v18: strict-locality + stress reward ---------------------------------
def test_v18_no_target_vec_body_in_obs():
    """v18e: build_selfshape_obs returns 22-dim, slot block (12:22) is
    invariant to swarm-wide translation (no centroid leakage)."""
    from train_mappo import build_selfshape_obs
    from environment import MeanShiftEnv
    cfg = load_cfg()
    env = MeanShiftEnv(cfg, n_envs=2, device="cpu", seed=0)
    out = env.expert_action(noise_gen=None, build_obs=False)
    obs1, _, _ = build_selfshape_obs(env, out, cue_present=env.cue_present,
                                      latent_z=env.latent_z, heading_locked=True)
    assert obs1.shape[-1] == 22, f"v18e obs dim must be 22, got {obs1.shape[-1]}"
    assert not torch.isnan(obs1).any()
    # Translate all positions by same offset → slot one-hot block (47:57) unchanged
    offset = torch.tensor([0.123, -0.456])
    env.positions = env.positions + offset.view(1, 1, 2)
    out2 = env.expert_action(noise_gen=None, build_obs=False)
    obs2, _, _ = build_selfshape_obs(env, out2, cue_present=env.cue_present,
                                      latent_z=env.latent_z, heading_locked=True)
    assert torch.allclose(obs1[..., 12:22], obs2[..., 12:22]), \
        "slot one-hot block should be translation-invariant (no centroid leak)"


def test_v18_formation_stress_reward_basic():
    """Place agents EXACTLY at template (rotated by random θ) → stress = 0.
    Perturb one agent → stress > 0 monotonically."""
    from shape_metrics import (formation_stress_reward, make_centered_t4_template,
                                pairwise_dist)
    spacing = 0.045
    template = make_centered_t4_template(spacing)                     # [10, 2]
    pair_dist = pairwise_dist(template)                                # [10, 10]
    N = 10
    # Perfect placement (slot k → template[k]) under random rotation
    theta = 0.7
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]])
    pos_perfect = (template @ R.T).unsqueeze(0)                        # [1, 10, 2]
    slots = torch.arange(N).unsqueeze(0)                               # [1, 10]
    env_score, per_agent = formation_stress_reward(
        pos_perfect, slots, pair_dist, comm_radius=10.0, spacing=spacing)
    assert env_score.shape == (1,)
    assert per_agent.shape == (1, 10)
    assert env_score.item() >= -1e-6, f"perfect placement should give stress=0, got {env_score.item()}"
    # Perturb agent 0 by 1d → stress should be clearly positive (negated < 0)
    pos_pert = pos_perfect.clone()
    pos_pert[0, 0] = pos_pert[0, 0] + torch.tensor([spacing, 0.0])
    env_pert, _ = formation_stress_reward(pos_pert, slots, pair_dist,
                                            comm_radius=10.0, spacing=spacing)
    assert env_pert.item() < env_score.item() - 1e-3, \
        f"perturbation should decrease (negative) stress reward; perfect={env_score.item()} pert={env_pert.item()}"


def test_v18_formation_stress_translation_invariance():
    """Stress is invariant under global translation (only depends on pairwise dist)."""
    from shape_metrics import (formation_stress_reward, make_centered_t4_template,
                                pairwise_dist)
    spacing = 0.045
    template = make_centered_t4_template(spacing)
    pair_dist = pairwise_dist(template)
    pos = template.unsqueeze(0) + torch.randn(1, 10, 2) * 0.5 * spacing
    slots = torch.arange(10).unsqueeze(0)
    s_a, _ = formation_stress_reward(pos, slots, pair_dist, comm_radius=10.0,
                                      spacing=spacing)
    pos_translated = pos + torch.tensor([5.0, -3.0]).view(1, 1, 2)
    s_b, _ = formation_stress_reward(pos_translated, slots, pair_dist,
                                      comm_radius=10.0, spacing=spacing)
    assert torch.allclose(s_a, s_b, atol=1e-5)


def test_v18_formation_stress_rotation_invariance():
    """Stress is invariant under global rotation (pairwise distances are SE(2)-invariant)."""
    from shape_metrics import (formation_stress_reward, make_centered_t4_template,
                                pairwise_dist)
    spacing = 0.045
    template = make_centered_t4_template(spacing)
    pair_dist = pairwise_dist(template)
    pos = template.unsqueeze(0) + torch.randn(1, 10, 2) * 0.3 * spacing
    slots = torch.arange(10).unsqueeze(0)
    s_a, _ = formation_stress_reward(pos, slots, pair_dist, comm_radius=10.0,
                                      spacing=spacing)
    theta = 1.2
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]])
    pos_rotated = pos @ R.T
    s_b, _ = formation_stress_reward(pos_rotated, slots, pair_dist,
                                      comm_radius=10.0, spacing=spacing)
    assert torch.allclose(s_a, s_b, atol=1e-5)


def test_multishape_env_default_pool_is_t4():
    """Without cfg.shape_pool, env defaults to ["t4"] (preserves legacy behavior)."""
    from environment import MeanShiftEnv
    cfg = load_cfg()
    cfg.pop("shape_pool", None)
    env = MeanShiftEnv(cfg, n_envs=4, device="cpu", seed=0)
    assert env._shape_names == ["t4"]
    assert env.slot_pair_dist.shape == (4, 10, 10)


def test_multishape_env_samples_shapes():
    """With shape_pool=[4 shapes] and many envs, reset_all produces >=2 unique ids."""
    from environment import MeanShiftEnv
    cfg = load_cfg()
    cfg["shape_pool"] = ["t4", "hex4", "ring10", "row5x2"]
    env = MeanShiftEnv(cfg, n_envs=64, device="cpu", seed=42)
    env.reset_all()
    unique_ids = torch.unique(env.shape_id).numel()
    assert unique_ids >= 2, f"expected diverse shape sampling, got {unique_ids} unique ids"
    assert env.slot_pair_dist.shape == (64, 10, 10)


def test_multishape_target_map_matches_shape_id():
    """env.target_map of a shape should match that shape's pairwise distances."""
    from environment import MeanShiftEnv
    from shape_metrics import pairwise_dist
    cfg = load_cfg()
    cfg["shape_pool"] = ["t4", "hex4", "ring10", "row5x2"]
    env = MeanShiftEnv(cfg, n_envs=64, device="cpu", seed=42)
    env.reset_all()
    for b in range(64):
        sid = int(env.shape_id[b].item())
        actual_pdist = pairwise_dist(env.target_map[b])
        expected_pdist = env._cached_pair_dists[sid]
        assert torch.allclose(actual_pdist, expected_pdist, atol=1e-4), \
            f"env {b} shape_id {sid}: target_map pdist mismatch"


def test_multishape_formation_stress_batched_equivalence():
    """formation_stress_reward with [10,10] and broadcast [B,10,10] return identical
    values when all envs use the same shape."""
    from shape_metrics import (formation_stress_reward,
                                make_centered_t4_template, pairwise_dist)
    torch.manual_seed(0)
    spacing = 0.045
    template = make_centered_t4_template(spacing)
    pair_2d = pairwise_dist(template)
    B = 8
    pair_3d = pair_2d.unsqueeze(0).expand(B, -1, -1).contiguous()
    pos = template.unsqueeze(0).expand(B, -1, -1) + torch.randn(B, 10, 2) * 0.2 * spacing
    slots = torch.stack([torch.randperm(10) for _ in range(B)], dim=0)
    s2, p2 = formation_stress_reward(pos, slots, pair_2d, comm_radius=10.0, spacing=spacing)
    s3, p3 = formation_stress_reward(pos, slots, pair_3d, comm_radius=10.0, spacing=spacing)
    assert torch.allclose(s2, s3, atol=1e-6), f"env-scalar diff: {(s2-s3).abs().max()}"
    assert torch.allclose(p2, p3, atol=1e-6), f"per-agent diff: {(p2-p3).abs().max()}"


def test_multishape_formation_stress_distinguishes_shapes():
    """When envs have different shape pair_dists, stress values differ."""
    from shape_metrics import formation_stress_reward, pairwise_dist
    from templates import make_template
    spacing = 0.045
    tpl_t4 = make_template("t4", spacing)
    tpl_hex = make_template("hex4", spacing)
    pair_t4 = pairwise_dist(tpl_t4)
    pair_hex = pairwise_dist(tpl_hex)
    B = 4
    pair_3d = torch.stack([pair_t4, pair_t4, pair_hex, pair_hex], dim=0)
    # All envs have positions matching T4 — t4 envs should have lower stress.
    pos = tpl_t4.unsqueeze(0).expand(B, -1, -1).contiguous()
    slots = torch.arange(10).unsqueeze(0).expand(B, -1)
    _, per_agent = formation_stress_reward(pos, slots, pair_3d,
                                            comm_radius=10.0, spacing=spacing)
    stress_t4 = (-per_agent[:2]).mean().item()
    stress_hex = (-per_agent[2:]).mean().item()
    assert stress_t4 < stress_hex, \
        f"T4 stress {stress_t4} should be < hex stress {stress_hex} when positions match T4"


def test_multishape_compute_shape_metrics_multi():
    """compute_shape_metrics_multi returns per-shape correct metrics."""
    from environment import MeanShiftEnv
    from templates import make_template
    cfg = load_cfg()
    cfg["shape_pool"] = ["t4", "hex4", "ring10", "row5x2"]
    env = MeanShiftEnv(cfg, n_envs=16, device="cpu", seed=7)
    env.reset_all()
    templates = {name: make_template(name, env.spacing) for name in env._shape_names}
    prof, _, iso, bp = env.compute_shape_metrics_multi(templates)
    assert prof.shape == (16,) and bp.shape == (16,) and iso.shape == (16,)
    for b in range(16):
        assert torch.isfinite(prof[b]) and prof[b] >= 0
        assert torch.isfinite(bp[b]) and bp[b] >= 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-v"]))
