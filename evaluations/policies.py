"""Reference policies and threshold calibration for the benchmark

Every policy follows the same call signature as `training/baselines.py`:

    policy(observation, info, env) -> int

so it plugs into `training.evaluation.evaluate`
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.baselines import RiskThresholdPolicy  
from training.evaluation import evaluate, summarise  
from training.features import ObservationEncoder  


def risk_arrays(encoder: ObservationEncoder, info, env):
    """Per-vessel predicted closest-approach distance (m) and time (s)."""
    risk = encoder.risk_matrix(info["fleet_state"], env.vessel_ids)
    dcpa_m = risk[:, 0] * 2.0 * env.warning_distance_m
    tcpa_s = risk[:, 1] * encoder.risk_horizon_s
    return dcpa_m, tcpa_s


class HighestRiskGreedyPolicy:
    """Assist the vessel with the highest immediate risk whenever possible.

    Risk is the predicted closest-approach distance (constant-velocity
    assumption) to any active neighbour; ties are broken by the
    closest-approach time.  
    With `threshold_m=None` (default) the operator dispatches to the
    vessel with the smallest predicted DCPA whenever an action is valid.
    Set `threshold_m` to a distance to dispatch only when the smallest
    predicted DCPA is below that threshold.
    
    """

    def __init__(self, horizon_s: float = 300.0, threshold_m=None):
        self.horizon_s = float(horizon_s)
        self.threshold_m = threshold_m
        self._encoder = None

    def _encoder_for(self, env):
        if self._encoder is None or self._encoder.num_vessels != env.num_vessels:
            self._encoder = ObservationEncoder.for_env(env, risk_horizon_s=self.horizon_s)
        return self._encoder

    def __call__(self, observation, info, env):
        mask = np.asarray(info["action_mask"], dtype=bool)
        valid = [i + 1 for i in range(env.num_vessels) if mask[i + 1]]
        if not valid:
            return 0
        dcpa_m, tcpa_s = risk_arrays(self._encoder_for(env), info, env)
        best = min(valid, key=lambda a: (dcpa_m[a - 1], tcpa_s[a - 1]))
        if self.threshold_m is not None and dcpa_m[best - 1] >= self.threshold_m:
            return 0
        return best


def calibrate_risk_gate(
    envs,
    thresholds=(100.0, 250.0, 500.0, 750.0, 1000.0),
    horizons=(150.0, 300.0, 600.0),
):
    """Pick the risk-gate threshold and horizon that score best on ``envs``.

    ``envs`` must be calibration scenarios that are NOT used for the final
    comparison.  Returns ``(policy, table)`` where ``table`` is a list of dict
    rows (one per threshold/horizon pair) and ``policy`` is a fresh
    ``RiskThresholdPolicy`` using the best pair.
    """
    table = []
    best_key, best_row = None, None
    for horizon in horizons:
        for threshold in thresholds:
            policy = RiskThresholdPolicy(threshold_m=float(threshold), horizon_s=float(horizon))
            summary = summarise(evaluate(envs, policy))
            row = {
                "threshold_m": float(threshold),
                "horizon_s": float(horizon),
                "reward_mean": summary["reward_mean"],
                "collision_events_mean": summary["collision_events_mean"],
                "near_miss_events_mean": summary["near_miss_events_mean"],
                "interventions_mean": summary["interventions_mean"],
            }
            table.append(row)
            key = (row["reward_mean"], -row["interventions_mean"])
            if best_key is None or key > best_key:
                best_key, best_row = key, row
    policy = RiskThresholdPolicy(
        threshold_m=best_row["threshold_m"], horizon_s=best_row["horizon_s"]
    )
    return policy, table


__all__ = ["HighestRiskGreedyPolicy", "calibrate_risk_gate", "risk_arrays"]
