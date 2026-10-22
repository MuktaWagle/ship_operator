"""Frozen BC navigation plus explicit timed operator control."""

import hashlib
from pathlib import Path

import numpy as np

from .features import MODEL_FEATURE_NAMES, features, augment_features
from .model import BCPolicy


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "bc_navigation_epoch025.pt"
EXPECTED_SHA256 = "2d396e209396c8dbc03d877e1809546021856b67cde8f743bb7d7f82553c1c15"


class FrozenBCPolicy:
    """Load and run the selected, frozen BC navigation checkpoint."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        import torch

        checkpoint = Path(checkpoint)
        self.torch = torch
        self.sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.sha256 != EXPECTED_SHA256:
            raise ValueError("Expected the frozen, held-out-tested epoch-25 model.")

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            saved["architecture"] != "10-64-64-2_relu_tanh_sigmoid_v3"
            or saved["epoch"] != 25
        ):
            raise ValueError("Incorrect policy architecture or epoch.")
        if saved["model_feature_names"] != MODEL_FEATURE_NAMES:
            raise ValueError("Incorrect policy feature order.")

        self.manifest = saved["manifest"]
        self.manifest_sha256 = saved["manifest_sha256"]
        self.model = BCPolicy()
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        torch.set_num_threads(1)

    def predict(self, ship):
        position = np.array([ship["x"], ship["y"]])
        goal = np.array(ship["goal"])
        original = features(position, ship["heading"], ship["speed"], goal)
        model_input = augment_features(original[None, :], self.manifest)

        with self.torch.inference_mode():
            action = self.model(
                self.torch.from_numpy(model_input)
            ).numpy()[0]

        return (
            float(action[0] * self.manifest["max_turn_radians"]),
            float(action[1] * self.manifest["max_speed_ms"]),
        )


class BCController:
    """Drive all ships with BC except during a timed operator takeover."""

    def __init__(self, fleet, policy=None):
        self.fleet = fleet
        self.policy = policy if policy is not None else FrozenBCPolicy()
        manifest = self.policy.manifest

        for env in fleet.environments.values():
            checks = [
                (fleet.dt, manifest["dt"]),
                (env.max_x, manifest["map_width_m"]),
                (env.max_y, manifest["map_height_m"]),
                (env.goal_threshold, manifest["arrival_radius_m"]),
            ]
            if any(not np.isclose(actual, expected) for actual, expected in checks):
                raise ValueError("Fleet geometry or dt differs from BC training.")

        self.reset()

    def reset(self):
        self.mode = {
            ship_id: "autonomous"
            for ship_id in self.fleet.vessel_ids
        }
        self.operator = {}
        self.events = []

    def _record(self, event, ship_id, **extra):
        self.events.append({
            "timestamp": float(self.fleet.current_time),
            "ship_id": int(ship_id),
            "event": event,
            **extra,
        })

    def take_over(self, ship_id, direction="right", duration=30, turn_rate_deg=1.0):
        if ship_id not in self.fleet.active_ship_ids:
            raise ValueError("Select an active ship.")

        busy = [
            other
            for other in self.fleet.active_ship_ids
            if self.mode[other] == "operator" and self.remaining_seconds(other) > 0
        ]
        if any(other != ship_id for other in busy):
            raise ValueError("Operator already controls another ship.")
        if direction not in ("left", "right", "stop"):
            raise ValueError("Use left, right or stop.")
        if (
            not np.isfinite(duration)
            or duration <= 0
            or not np.isclose(duration / self.fleet.dt, round(duration / self.fleet.dt))
        ):
            raise ValueError("Duration must be a positive multiple of dt.")
        if not np.isfinite(turn_rate_deg) or turn_rate_deg <= 0:
            raise ValueError("Turn rate must be positive.")

        turn = (
            0.0
            if direction == "stop"
            else np.radians(turn_rate_deg) * self.fleet.dt
            * (1 if direction == "left" else -1)
        )
        if abs(turn) > self.policy.manifest["max_turn_radians"] + 1e-8:
            raise ValueError("Manual turn exceeds the BC turn bound.")

        ship = self.fleet.get_state()["ships"][ship_id]
        speed = float(np.clip(
            ship["speed"] if ship["speed"] > 0 else 3.0,
            0,
            self.policy.manifest["max_speed_ms"],
        ))
        self.operator[ship_id] = {
            "direction": direction,
            "turn": turn,
            "speed": speed,
            "until": self.fleet.current_time + duration,
        }
        self.mode[ship_id] = "operator"
        self._record("takeover", ship_id, direction=direction, duration=float(duration))

    def release(self, ship_id, reason="manual"):
        if ship_id not in self.fleet.active_ship_ids:
            raise ValueError("Cannot release an inactive ship.")
        self.operator.pop(ship_id, None)
        self.mode[ship_id] = "autonomous"
        self._record("release", ship_id, reason=reason)

    def remaining_seconds(self, ship_id):
        return max(
            0.0,
            self.operator.get(ship_id, {}).get("until", self.fleet.current_time)
            - self.fleet.current_time,
        )

    def _movement(self, ship_id, ship, turn, speed):
        manifest = self.policy.manifest
        if (
            not np.isfinite([turn, speed]).all()
            or abs(turn) > manifest["max_turn_radians"] + 1e-6
            or not 0 <= speed <= manifest["max_speed_ms"] + 1e-6
        ):
            raise ValueError("Invalid BC turn/speed command.")

        heading = ship["heading"] + turn
        command = np.array([
            speed * self.fleet.dt * np.cos(heading),
            speed * self.fleet.dt * np.sin(heading),
            turn,
        ])
        action_space = self.fleet.environments[ship_id].action_space
        if np.any(command < action_space.low) or np.any(command > action_space.high):
            raise ValueError("BC action exceeds fleet action space.")
        return command

    def actions(self, state):
        if state["timestamp"] != self.fleet.current_time:
            raise ValueError("Stale fleet state.")

        commands = {}
        for ship_id in self.fleet.active_ship_ids:
            ship = state["ships"][ship_id]

            if self.mode[ship_id] == "operator":
                if self.remaining_seconds(ship_id) <= 0:
                    self.release(ship_id, reason="timer")
                else:
                    command = self.operator[ship_id]
                    commands[ship_id] = (
                        np.zeros(3)
                        if command["direction"] == "stop"
                        else self._movement(
                            ship_id,
                            ship,
                            command["turn"],
                            command["speed"],
                        )
                    )
                    continue

            turn, speed = self.policy.predict(ship)
            commands[ship_id] = self._movement(ship_id, ship, turn, speed)

        return commands
