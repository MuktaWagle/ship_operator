"""Drop-in replacement for BCController that adds a planning operator.

`BCController.take_over` only knows fixed manoeuvres (left / right / stop). This
subclass adds `direction="avoid"`: while that takeover is active, the chosen
planner computes the vessel's turn and speed every step from the live fleet
state. Timers, one-operator checks, release and BC navigation are inherited, so
`FleetAllocationEnv`, the tracker and the demos work unchanged.

Use it by swapping the controller and selecting the new mode:

    controller = PlanningBCController(fleet, policy=policy, planner=VelocityObstaclePlanner())
    env = FleetAllocationEnv(fleet, controller, tracker, ..., assistance_direction="right")
    env.assistance_direction = "avoid"   # or relax the check in FleetAllocationEnv.__init__

The `assistance_direction` check in `FleetAllocationEnv.__init__` only allows
left / right / stop, so either set the attribute afterwards (as above) or add
"avoid" to that tuple (one-line change).
"""
from __future__ import annotations

import numpy as np

from bc_navigation.controller import BCController

from .planners import Limits, Planner


class PlanningBCController(BCController):
    def __init__(self, fleet, policy=None, planner: Planner | None = None):
        # BCController.__init__ calls self.reset(), which needs self.planner.
        self.planner = planner
        super().__init__(fleet, policy=policy)
        manifest = self.policy.manifest
        self.limits = Limits(
            dt=float(fleet.dt),
            max_turn=float(manifest["max_turn_radians"]),
            max_speed=float(manifest["max_speed_ms"]),
        )

    def reset(self):
        super().reset()
        if self.planner is not None:
            self.planner.reset()

    def take_over(self, ship_id, direction="right", duration=30, turn_rate_deg=1.0):
        if direction != "avoid":
            return super().take_over(ship_id, direction, duration, turn_rate_deg)

        if self.planner is None:
            raise ValueError("direction='avoid' needs a planner.")
        if ship_id not in self.fleet.active_ship_ids:
            raise ValueError("Select an active ship.")
        busy = [
            other
            for other in self.fleet.active_ship_ids
            if self.mode[other] == "operator" and self.remaining_seconds(other) > 0
        ]
        if any(other != ship_id for other in busy):
            raise ValueError("Operator already controls another ship.")
        if (
            not np.isfinite(duration)
            or duration <= 0
            or not np.isclose(duration / self.fleet.dt, round(duration / self.fleet.dt))
        ):
            raise ValueError("Duration must be a positive multiple of dt.")

        ship = self.fleet.get_state()["ships"][ship_id]
        self.operator[ship_id] = {
            "direction": "avoid",
            "turn": 0.0,
            "speed": float(np.clip(ship["speed"], 0.0, self.limits.max_speed)),
            "until": self.fleet.current_time + duration,
        }
        self.mode[ship_id] = "operator"
        self._record("takeover", ship_id, direction="avoid", duration=float(duration))

    def actions(self, state):
        # The base class handles timers, release and BC for everyone, and gives
        # an "avoid" ship a harmless straight-line placeholder that we replace.
        commands = super().actions(state)
        for ship_id in self.fleet.active_ship_ids:
            if (
                self.mode[ship_id] == "operator"
                and self.operator.get(ship_id, {}).get("direction") == "avoid"
            ):
                ship = state["ships"][ship_id]
                others = [
                    state["ships"][other]
                    for other in self.fleet.active_ship_ids
                    if other != ship_id
                ]
                steps_left = max(1, int(np.ceil(self.remaining_seconds(ship_id) / self.fleet.dt - 1e-9)))
                turn, speed = self.planner.plan(ship_id, ship, others, self.limits, steps_left)
                commands[ship_id] = self._movement(ship_id, ship, turn, speed)
        return commands