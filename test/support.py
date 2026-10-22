"""Small controller doubles used by tests that do not exercise navigation."""

import numpy as np


class HoldController:
    """Minimal controller for isolating fleet/operator bookkeeping tests."""

    def __init__(self, fleet):
        self.fleet = fleet
        self.reset()

    def reset(self):
        self.mode = {
            ship_id: "autonomous"
            for ship_id in self.fleet.vessel_ids
        }
        self.operator = {}

    def take_over(self, ship_id, direction="right", duration=30, turn_rate_deg=1.0):
        del turn_rate_deg
        self.mode[ship_id] = "operator"
        self.operator[ship_id] = {
            "direction": direction,
            "remaining": float(duration),
        }

    def release(self, ship_id):
        self.mode[ship_id] = "autonomous"
        self.operator.pop(ship_id, None)

    def remaining_seconds(self, ship_id):
        return max(0.0, self.operator.get(ship_id, {}).get("remaining", 0.0))

    def actions(self, state):
        del state
        actions = {}
        for ship_id in self.fleet.active_ship_ids:
            if self.mode[ship_id] == "operator":
                self.operator[ship_id]["remaining"] -= self.fleet.dt
                if self.remaining_seconds(ship_id) <= 0:
                    self.release(ship_id)
            actions[ship_id] = np.zeros(3, dtype=np.float32)
        return actions
