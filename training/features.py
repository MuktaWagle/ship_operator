"""Flatten FleetAllocationEnv observations into one vector for the DQN.

``FleetAllocationEnv`` returns a ``spaces.Dict`` with three groups
(``ships``, ``nearest_distance``, ``operator``).  An MLP Q-network needs a
single fixed-length vector, so this module concatenates them in a fixed
order.

Optionally (``risk_features=True``, the default) it appends four extra
per-vessel features computed from the raw fleet state:

    dcpa        predicted closest-approach distance to any active neighbour,
                assuming constant velocity, / (2 * warning distance)
    tcpa        time until that closest approach / risk horizon
    proximity   current nearest-neighbour distance / (2 * warning distance)
    goal_dist   distance to own goal / map diagonal

All four are clipped to [0, 1].  The original observation only gives raw
positions and headings, so the network would otherwise have to learn
pairwise geometry from 15 x 8 numbers.  The extra features make "who is
about to get close to whom" directly visible.  Turn them off to train on
the unchanged Task 3 observation (useful as an ablation for the report).

Layout of the encoded vector for N vessels::

    [ship_0 (8 [+4]), ship_1, ..., ship_{N-1}, nearest_distance (N), operator (8)]
"""

from typing import Any, Dict, Optional

import numpy as np


SHIP_FEATURES = 8
RISK_FEATURES = 4
OPERATOR_FEATURES = 8


class ObservationEncoder:
    """Turn a FleetAllocationEnv observation into a flat float32 vector."""

    def __init__(
        self,
        num_vessels: int,
        *,
        risk_features: bool = True,
        warning_distance_m: float = 500.0,
        risk_horizon_s: float = 300.0,
        map_diagonal_m: float = 10250.0,
    ):
        if num_vessels < 1:
            raise ValueError("num_vessels must be positive.")
        if warning_distance_m <= 0 or risk_horizon_s <= 0 or map_diagonal_m <= 0:
            raise ValueError("Distance and horizon scales must be positive.")

        self.num_vessels = int(num_vessels)
        self.risk_features = bool(risk_features)
        self.warning_distance_m = float(warning_distance_m)
        self.risk_horizon_s = float(risk_horizon_s)
        self.map_diagonal_m = float(map_diagonal_m)

    @classmethod
    def for_env(cls, env, *, risk_features: bool = True, risk_horizon_s: float = 300.0):
        """Build an encoder whose scales match a FleetAllocationEnv."""
        return cls(
            env.num_vessels,
            risk_features=risk_features,
            warning_distance_m=env.warning_distance_m,
            risk_horizon_s=risk_horizon_s,
            map_diagonal_m=env.map_diagonal,
        )

    @property
    def per_ship_dim(self) -> int:
        return SHIP_FEATURES + (RISK_FEATURES if self.risk_features else 0)

    @property
    def dim(self) -> int:
        return self.num_vessels * (self.per_ship_dim + 1) + OPERATOR_FEATURES

    def encode(
        self,
        observation: Dict[str, np.ndarray],
        fleet_state: Optional[Dict[str, Any]] = None,
        vessel_ids=None,
    ) -> np.ndarray:
        """Return the flat vector.

        ``fleet_state`` and ``vessel_ids`` are required when risk features are
        on: pass ``info["fleet_state"]`` and ``env.vessel_ids``.
        """
        ships = np.asarray(observation["ships"], dtype=np.float32)
        nearest = np.asarray(observation["nearest_distance"], dtype=np.float32)
        operator = np.asarray(observation["operator"], dtype=np.float32)

        if ships.shape != (self.num_vessels, SHIP_FEATURES):
            raise ValueError(
                f"Expected ships shape {(self.num_vessels, SHIP_FEATURES)}, "
                f"got {ships.shape}."
            )

        if self.risk_features:
            if fleet_state is None or vessel_ids is None:
                raise ValueError(
                    "risk_features=True needs fleet_state and vessel_ids."
                )
            risk = self.risk_matrix(fleet_state, vessel_ids)
            ships = np.concatenate([ships, risk], axis=1)

        return np.concatenate([
            ships.reshape(-1),
            nearest.reshape(-1),
            operator.reshape(-1),
        ]).astype(np.float32)

    def risk_matrix(self, fleet_state: Dict[str, Any], vessel_ids) -> np.ndarray:
        """Per-vessel [dcpa, tcpa, proximity, goal_dist], each in [0, 1].

        Inactive vessels get ones (no risk, far away, nothing left to do).
        """
        n = self.num_vessels
        result = np.ones((n, RISK_FEATURES), dtype=np.float32)
        ships = fleet_state["ships"]

        active = [
            index for index, ship_id in enumerate(vessel_ids)
            if ships[ship_id]["status"] == "active"
        ]
        if not active:
            return result

        positions = np.zeros((n, 2))
        velocities = np.zeros((n, 2))
        for index, ship_id in enumerate(vessel_ids):
            ship = ships[ship_id]
            positions[index] = (ship["x"], ship["y"])
            velocities[index] = (
                ship["speed"] * np.cos(ship["heading"]),
                ship["speed"] * np.sin(ship["heading"]),
            )
            if index in active:
                goal = np.asarray(ship["goal"], dtype=float)
                result[index, 3] = np.clip(
                    np.linalg.norm(goal - positions[index]) / self.map_diagonal_m,
                    0.0,
                    1.0,
                )

        scale = 2.0 * self.warning_distance_m
        for i in active:
            best_dcpa, best_tcpa, best_now = np.inf, self.risk_horizon_s, np.inf
            for j in active:
                if i == j:
                    continue
                rel_p = positions[j] - positions[i]
                rel_v = velocities[j] - velocities[i]
                now = float(np.linalg.norm(rel_p))
                speed_sq = float(rel_v @ rel_v)
                if speed_sq < 1e-9:
                    t_cpa = 0.0
                else:
                    t_cpa = float(np.clip(-(rel_p @ rel_v) / speed_sq, 0.0, self.risk_horizon_s))
                d_cpa = float(np.linalg.norm(rel_p + rel_v * t_cpa))
                best_now = min(best_now, now)
                if d_cpa < best_dcpa:
                    best_dcpa, best_tcpa = d_cpa, t_cpa
            if np.isfinite(best_dcpa):
                result[i, 0] = min(1.0, best_dcpa / scale)
                result[i, 1] = best_tcpa / self.risk_horizon_s
                result[i, 2] = min(1.0, best_now / scale)

        return result


__all__ = ["ObservationEncoder"]
