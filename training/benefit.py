import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.mcts import (  # noqa: E402
    ModelMCTSConfig,
    build_state,
    model_step,
    params_from_env,
    predicted_cpa,
)


@dataclass
class BenefitConfig:
    horizon: int = 30
    gamma: float = 0.99
    activation_m: float = 1000.0
    margin: float = 0.0
    gate: bool = True
    risk_horizon_s: float = 300.0

    def to_dict(self) -> Dict:
        return asdict(self)


class BenefitEstimator:

    def __init__(self, config: Optional[BenefitConfig] = None):
        self.config = config or BenefitConfig()
        self._shared: Dict = {}
        self._mcts_cfg = ModelMCTSConfig(
            activation_m=self.config.activation_m,
            risk_horizon_s=self.config.risk_horizon_s,
        )
        self.calls = 0
        self.rollouts = 0

    def _rollout(self, params, state, first_action: int) -> float:
        s = state.copy()
        total, discount = 0.0, 1.0
        action = first_action
        for _ in range(self.config.horizon):
            reward, done = model_step(params, s, action)
            total += discount * reward
            discount *= self.config.gamma
            action = 0
            if done:
                break
        self.rollouts += 1
        return total

    def __call__(self, env, info) -> Tuple[np.ndarray, float, np.ndarray]:
        """Return (gains, idle_return, gated_mask), each indexed like the action space."""
        cfg = self.config
        mask = np.asarray(info["action_mask"], dtype=bool).copy()
        n = env.num_vessels
        cost = float(env.reward_weights.intervention)
        gains = np.full(n + 1, -cost, dtype=np.float64)
        gains[0] = 0.0
        idle_return = 0.0

        if mask[1:].any():
            self.calls += 1
            params = params_from_env(env, self._mcts_cfg, self._shared)
            env._mcts_params = params  # build_state reads it
            state = build_state(env, info)
            cpa = predicted_cpa(params, state)
            candidates = [a for a in range(1, n + 1) if mask[a] and cpa[a - 1] < cfg.activation_m]
            if candidates:
                idle_return = self._rollout(params, state, 0)
                for a in candidates:
                    gains[a] = self._rollout(params, state, a) - idle_return

        if cfg.gate:
            mask[1:] &= gains[1:] > cfg.margin
        return gains, idle_return, mask


def gain_features(gains: np.ndarray, idle_return: float, w_near_miss: float) -> np.ndarray:
    """Per-ship squashed gains followed by the squashed idle return (N + 1 values)."""
    scale = max(1e-6, float(w_near_miss))
    per_ship = np.tanh(np.asarray(gains[1:], dtype=np.float64) / scale)
    overall = math.tanh(idle_return / (3.0 * scale))
    return np.concatenate([per_ship, [overall]]).astype(np.float32)


__all__ = [
    "BenefitConfig",
    "BenefitEstimator",
    "gain_features",
]
