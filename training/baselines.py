"""Reference allocation policies used to judge the DQN during evaluation"""

import numpy as np

from .dqn import masked_random_action
from .features import ObservationEncoder


def idle_policy(observation, info, env):
    """Never intervene (lower bound: what the BC fleet does on its own)."""
    return 0


class RandomValidPolicy:
    """Pick uniformly among valid actions, including idle."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, observation, info, env):
        return masked_random_action(info["action_mask"], self.rng)


def greedy_nearest_policy(observation, info, env):
    """Whenever the operator is free, assist the ship closest to a neighbour.

    This is the smoke-test policy from test/run_simple_allocation.py. It spends
    budget as fast as possible.
    """
    mask = info["action_mask"]
    candidates = [
        (float(observation["nearest_distance"][i]), i + 1)
        for i in range(env.num_vessels)
        if observation["ships"][i][6] > 0.5 and mask[i + 1]
    ]
    return min(candidates)[1] if candidates else 0


class RiskThresholdPolicy:
    """Assist the highest-risk ship only if its risk crosses a threshold.

    Risk = predicted closest-approach distance (constant-velocity CPA) within
    the horizon.  Intervene when it is below ``threshold_m``; otherwise save
    the budget.  This mirrors the proposal's risk-threshold baseline.
    """

    def __init__(self, threshold_m: float = 500.0, horizon_s: float = 300.0):
        self.threshold_m = threshold_m
        self.horizon_s = horizon_s
        self._encoder = None

    def __call__(self, observation, info, env):
        mask = info["action_mask"]
        if not mask[1:].any():
            return 0
        if self._encoder is None or self._encoder.num_vessels != env.num_vessels:
            self._encoder = ObservationEncoder.for_env(env, risk_horizon_s=self.horizon_s)
        risk = self._encoder.risk_matrix(info["fleet_state"], env.vessel_ids)
        dcpa_m = risk[:, 0] * 2.0 * env.warning_distance_m
        best, best_action = np.inf, 0
        for i in range(env.num_vessels):
            if mask[i + 1] and dcpa_m[i] < best:
                best, best_action = dcpa_m[i], i + 1
        return best_action if best < self.threshold_m else 0


def make_baselines(seed: int = 0):
    return {
        "idle": idle_policy,
        "random_valid": RandomValidPolicy(seed),
        "greedy_nearest": greedy_nearest_policy,
        "risk_threshold": RiskThresholdPolicy(),
    }


__all__ = [
    "RandomValidPolicy",
    "RiskThresholdPolicy",
    "greedy_nearest_policy",
    "idle_policy",
    "make_baselines",
]
