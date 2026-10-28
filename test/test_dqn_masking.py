"""Checks for the masked DQN allocator"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test.test_allocation_env import make_env  
from training.dqn import (  
    DQNConfig,
    MaskedDQNAgent,
    NStepAccumulator,
    apply_mask_torch,
    masked_argmax,
    masked_random_action,
)
from training.evaluation import DQNPolicy, run_episode  
from training.features import ObservationEncoder  


def test_masked_argmax_ignores_invalid_best_action():
    q = np.array([0.0, 5.0, 1.0])
    mask = np.array([True, False, True])
    assert masked_argmax(q, mask) == 2


def test_masked_random_only_returns_valid_actions():
    rng = np.random.default_rng(0)
    mask = np.array([True, False, True, False])
    draws = {masked_random_action(mask, rng) for _ in range(200)}
    assert draws == {0, 2}


def test_idle_biased_exploration():
    rng = np.random.default_rng(0)
    mask = np.array([True, True, False, True])
    draws = [masked_random_action(mask, rng, idle_prob=0.9) for _ in range(5000)]
    assert set(draws) == {0, 1, 3}
    assert 0.87 < draws.count(0) / len(draws) < 0.93
    only_vessels = np.array([False, True, True])
    assert masked_random_action(only_vessels, rng, idle_prob=0.9) in (1, 2)


def test_forced_step_returns_idle_without_network():
    agent = MaskedDQNAgent(4, 3, DQNConfig(seed=0, eps_start=1.0))
    only_idle = np.array([True, False, False])
    for _ in range(50):
        assert agent.act(np.zeros(4, np.float32), only_idle) == 0


def test_masked_target_uses_only_next_valid_actions():
    """Bootstrap must ignore a huge Q-value on an action invalid in s'."""
    agent = MaskedDQNAgent(2, 3, DQNConfig(seed=0, dueling=False, double=False, batch_size=1, learning_starts=1))
    with torch.no_grad():
        for net in (agent.online, agent.target):
            for layer in net.body:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.zero_()
                    layer.bias.fill_(1.0)
            net.head.weight.zero_()
            net.head.bias.copy_(torch.tensor([1.0, 100.0, 2.0]))
    q = agent.target(torch.zeros(1, 2), None)
    masked = apply_mask_torch(q, torch.tensor([[True, False, True]]))
    assert float(masked.max()) == 2.0  # 100 on action 1 is excluded


def test_nstep_returns_and_terminal_discount():
    acc = NStepAccumulator(n=3, gamma=0.5)
    m = np.array([True])
    out = []
    out += acc.push("s0", m, 0, 1.0, "s1", m, False, False)
    out += acc.push("s1", m, 0, 1.0, "s2", m, False, False)
    out += acc.push("s2", m, 0, 1.0, "s3", m, False, False)
    assert len(out) == 1
    _, _, _, ret, discount, nxt, _ = out[0]
    assert ret == 1.0 + 0.5 + 0.25 and discount == 0.125 and nxt == "s3"

    out = acc.push("s3", m, 0, 4.0, "s4", m, True, False)  # terminal flushes all
    assert len(out) == 3 and all(t[4] == 0.0 for t in out)
    assert out[-1][3] == 4.0


def test_truncation_still_bootstraps():
    acc = NStepAccumulator(n=1, gamma=0.9)
    m = np.array([True])
    (transition,) = acc.push("s0", m, 0, 0.0, "s1", m, False, True)
    assert np.isclose(transition[4], 0.9)


def test_agent_trains_and_never_takes_invalid_action_in_env():
    env = make_env()
    try:
        encoder = ObservationEncoder.for_env(env)
        agent = MaskedDQNAgent(
            encoder.dim,
            env.num_vessels + 1,
            DQNConfig(seed=0, batch_size=4, learning_starts=4, eps_decay_steps=10),
        )
        updates = 0
        for episode in range(5):
            obs, info = env.reset(seed=43)
            x = encoder.encode(obs, info["fleet_state"], env.vessel_ids)
            while True:
                mask = info["action_mask"]
                action = agent.act(x, mask)
                assert mask[action]
                obs, reward, term, trunc, info = env.step(action)
                assert info["invalid_action"] is False
                x2 = encoder.encode(obs, info["fleet_state"], env.vessel_ids)
                if agent.observe(x, mask, action, reward, x2, info["action_mask"], term, trunc):
                    updates += 1
                x = x2
                if term or trunc:
                    break
        assert updates > 0

        totals = run_episode(env, DQNPolicy(agent, encoder), seed=43)
        assert totals["invalid_actions"] == 0
    finally:
        env.close()


def test_save_and_load_round_trip(tmp_dir=None):
    import tempfile
    agent = MaskedDQNAgent(6, 3, DQNConfig(seed=1, hidden=(16,)))
    x = np.random.default_rng(0).random(6).astype(np.float32)
    mask = np.array([True, True, True])
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "agent.pt"
        agent.save(path, extra={"note": "ok"})
        loaded = MaskedDQNAgent.load(path)
    assert np.allclose(agent.q_values(x, mask), loaded.q_values(x, mask))
    assert loaded.extra["note"] == "ok"


def test_encoder_shape_and_risk_range():
    env = make_env()
    try:
        obs, info = env.reset(seed=43)
        for risk in (True, False):
            enc = ObservationEncoder.for_env(env, risk_features=risk)
            x = enc.encode(obs, info["fleet_state"], env.vessel_ids)
            assert x.shape == (enc.dim,)
            assert np.all(np.isfinite(x))
        risk = ObservationEncoder.for_env(env).risk_matrix(info["fleet_state"], env.vessel_ids)
        assert risk.shape == (2, 4) and np.all((risk >= 0) & (risk <= 1))
        # Two ships 300 m apart on identical parallel courses:
        # proximity = dcpa = 300 / (2 * warning distance), tcpa = 0.
        expected = min(1.0, 300.0 / (2.0 * env.warning_distance_m))
        assert np.isclose(risk[0, 2], expected, atol=1e-6)
        assert np.isclose(risk[0, 0], expected, atol=1e-6)
        assert risk[0, 1] == 0.0
    finally:
        env.close()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print(f"All {len(tests)} DQN masking tests PASSED!")
