"""Checks for the benefit gate (training/benefit.py).

Run with:  python -m test.test_benefit_gate
Uses the small HoldController test environment, so no BC checkpoint is needed.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test.test_allocation_env import make_env  # noqa: E402
from training.benefit import BenefitConfig, BenefitEstimator, gain_features  # noqa: E402
from training.dqn import DQNConfig, MaskedDQNAgent  # noqa: E402
from training.evaluation import DQNPolicy, prepare_input, run_episode  # noqa: E402
from training.features import ObservationEncoder  # noqa: E402


def test_gated_mask_is_subset_of_tracker_mask():
    env = make_env()
    try:
        obs, info = env.reset(seed=43)
        estimator = BenefitEstimator(BenefitConfig(horizon=5))
        gains, idle_return, mask = estimator(env, info)
        tracker_mask = np.asarray(info["action_mask"], dtype=bool)
        assert mask.shape == tracker_mask.shape == gains.shape
        assert mask[0]  # idle always allowed
        assert not np.any(mask & ~tracker_mask)  # the gate only ever removes actions
        assert np.all(np.isfinite(gains)) and np.isfinite(idle_return)
        cost = env.reward_weights.intervention
        # A gated-out ship is one that is not predicted to pay.
        for a in range(1, len(mask)):
            if tracker_mask[a] and not mask[a]:
                assert gains[a] <= estimator.config.margin + 1e-9
            assert gains[a] >= -cost - 1e3
    finally:
        env.close()


def test_no_gate_keeps_tracker_mask():
    env = make_env()
    try:
        _, info = env.reset(seed=43)
        _, _, mask = BenefitEstimator(BenefitConfig(horizon=5, gate=False))(env, info)
        assert np.array_equal(mask, np.asarray(info["action_mask"], dtype=bool))
    finally:
        env.close()


def test_busy_operator_skips_rollouts():
    env = make_env()
    try:
        env.reset(seed=43)
        _, _, _, _, info = env.step(1)  # operator now busy -> only idle valid
        estimator = BenefitEstimator(BenefitConfig(horizon=5))
        gains, idle_return, mask = estimator(env, info)
        assert mask.tolist() == [True] + [False] * env.num_vessels
        assert estimator.rollouts == 0 and idle_return == 0.0
    finally:
        env.close()


def test_gain_features_shape_and_range():
    feats = gain_features(np.array([0.0, 50.0, -1.0, 2.0]), idle_return=-30.0, w_near_miss=5.0)
    assert feats.shape == (4,) and feats.dtype == np.float32
    assert np.all(np.abs(feats) <= 1.0)


def test_gated_dqn_runs_without_invalid_actions():
    env = make_env()
    try:
        encoder = ObservationEncoder.for_env(env, benefit_features=True)
        estimator = BenefitEstimator(BenefitConfig(horizon=5))
        obs, info = env.reset(seed=43)
        x, mask = prepare_input(encoder, estimator, obs, info, env)
        assert x.shape == (encoder.dim,)
        agent = MaskedDQNAgent(encoder.dim, env.num_vessels + 1, DQNConfig(seed=0, hidden=(16,)))
        totals = run_episode(env, DQNPolicy(agent, encoder, estimator), seed=43)
        assert totals["invalid_actions"] == 0
    finally:
        env.close()


def test_policy_rebuilds_gate_from_checkpoint_extra():
    import tempfile

    env = make_env()
    try:
        encoder = ObservationEncoder.for_env(env, benefit_features=True)
        agent = MaskedDQNAgent(encoder.dim, env.num_vessels + 1, DQNConfig(seed=0, hidden=(16,)))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "agent.pt"
            agent.save(path, extra={"benefit": BenefitConfig(horizon=5).to_dict()})
            loaded = MaskedDQNAgent.load(path)
        policy = DQNPolicy(loaded, encoder)
        assert policy.estimator is not None and policy.estimator.config.horizon == 5
    finally:
        env.close()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
    print(f"All {len(tests)} benefit gate tests PASSED!")
