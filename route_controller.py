import numpy as np


def angle_difference(target, current):
    return (target - current + np.pi) % (2 * np.pi) - np.pi


class RouteController:
    """Follow a route and recover after a timed operator command."""

    def __init__(self, fleet):
        self.fleet = fleet
        self.routes = {
            ship_id: fleet.trajectories[index][:, :2].copy()
            for index, ship_id in enumerate(fleet.vessel_ids)
        }
        self.reset()

    def reset(self):
        # Call this after resetting the fleet.
        self.next_waypoint = {
            ship_id: 1 for ship_id in self.fleet.vessel_ids
        }
        self.mode = {
            ship_id: "autonomous"
            for ship_id in self.fleet.vessel_ids
        }
        self.operator = {}
        self.recovery_target = {}
        self.recovery_index = {}

        # Fixed demonstration cruise speed per ship.
        self.cruise_speed = {
            ship_id: float(np.clip(
                np.median(self.fleet.trajectories[index][:, 2]),
                3.0,
                8.0,
            ))
            for index, ship_id in enumerate(self.fleet.vessel_ids)
        }

    def take_over(self, ship_id, direction="right", duration=30,
                  turn_rate_deg=1.0):
        """Start a timed left turn, right turn, or stationary hold."""
        if ship_id not in self.fleet.active_ship_ids:
            raise ValueError("Select an active ship.")

        direction = str(direction).strip().lower()

        if direction not in ("left", "right", "stop"):
            raise ValueError(
                "direction must be 'left', 'right', or 'stop'."
            )

        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be positive and finite.")

        if not np.isclose(
            duration / self.fleet.dt,
            round(duration / self.fleet.dt),
        ):
            raise ValueError("duration must be a multiple of fleet.dt.")

        env = self.fleet.environments[ship_id]

        if direction == "stop":
            turn = 0.0
        else:
            if not np.isfinite(turn_rate_deg) or turn_rate_deg <= 0:
                raise ValueError(
                    "turn_rate_deg must be positive and finite."
                )

            turn = np.radians(turn_rate_deg) * self.fleet.dt
            turn *= 1 if direction == "left" else -1

        if not env.action_space.low[2] <= turn <= env.action_space.high[2]:
            raise ValueError("Requested turn exceeds the action bounds.")

        self.operator[ship_id] = {
            "direction": direction,
            "turn": turn,
            "until": self.fleet.current_time + duration,
        }

        self.mode[ship_id] = "operator"

        print(
            f"TAKEOVER ship {ship_id}: {direction} "
            f"for {duration:g} simulated seconds."
        )

    def release(self, ship_id):
        """Choose the nearest point on the remaining route."""
        if ship_id not in self.fleet.active_ship_ids:
            raise ValueError("Cannot release an inactive ship.")

        ship = self.fleet.get_state()["ships"][ship_id]
        position = np.array([ship["x"], ship["y"]])
        route = self.routes[ship_id]

        first_segment = max(0, self.next_waypoint[ship_id] - 1)
        best = None

        for index in range(first_segment, len(route) - 1):
            start = route[index]
            change = route[index + 1] - start
            length_squared = float(change @ change)

            fraction = (
                0.0 if length_squared <= 1e-12
                else float(np.clip(
                    (position - start) @ change / length_squared,
                    0.0, 1.0,
                ))
            )

            # Do not rejoin behind the last recorded route progress.
            # The first unfinished segment begins at the last waypoint
            # already passed by this controller.
            point = start + fraction * change
            distance = float(np.linalg.norm(position - point))

            if best is None or distance < best[0]:
                best = (distance, point.copy(), index + 1)

        self.operator.pop(ship_id, None)
        self.recovery_target[ship_id] = best[1]
        self.recovery_index[ship_id] = best[2]
        self.mode[ship_id] = "recovering"

        print(
            f"RELEASE ship {ship_id}: "
            f"rejoin target is {best[0]:.1f} m away."
        )

    def remaining_seconds(self, ship_id):
        command = self.operator.get(ship_id)
        if command is None:
            return 0.0
        return max(0.0, command["until"] - self.fleet.current_time)

    def _movement(self, ship_id, ship, turn, distance):
        """Convert steering and travel distance into simulator actions."""
        env = self.fleet.environments[ship_id]
        turn = float(np.clip(
            turn,
            env.action_space.low[2],
            env.action_space.high[2],
        ))

        heading = ship["heading"] + turn
        direction = np.array([np.cos(heading), np.sin(heading)])

        # Conservatively keep both displacement components in bounds.
        distance_limit = float(min(
            env.action_space.high[0],
            env.action_space.high[1],
            -env.action_space.low[0],
            -env.action_space.low[1],
        ))
        distance = min(distance, distance_limit)
        dx, dy = direction * distance

        return np.array([dx, dy, turn], dtype=float)

    def _towards(self, ship_id, ship, target):
        position = np.array([ship["x"], ship["y"]])
        offset = target - position
        distance = float(np.linalg.norm(offset))

        if distance <= 1e-9:
            return np.zeros(3)

        desired_heading = np.arctan2(offset[1], offset[0])
        error = angle_difference(desired_heading, ship["heading"])

        # Prototype setting: at most 30 degrees per ten-second step.
        turn_limit = min(
            np.radians(3.0) * self.fleet.dt,
            float(self.fleet.environments[ship_id].action_space.high[2]),
        )
        turn = float(np.clip(error, -turn_limit, turn_limit))

        # Rotate first when badly misaligned. This is simplified
        # kinematics, not a physically realistic ship manoeuvre.
        travel = 0.0
        if abs(error - turn) <= 1e-6:
            travel = min(
                distance,
                self.cruise_speed[ship_id] * self.fleet.dt,
            )

        return self._movement(ship_id, ship, turn, travel)

    def actions(self, state):
        """Choose commands for every currently active vessel."""
        commands = {}

        for ship_id in self.fleet.active_ship_ids:
            ship = state["ships"][ship_id]
            position = np.array([ship["x"], ship["y"]])

            if self.mode[ship_id] == "operator":
                if self.remaining_seconds(ship_id) <= 0:
                    self.release(ship_id)
                else:
                    command = self.operator[ship_id]

                    if command["direction"] == "stop":
                        # Hold position and heading.
                        # FleetSimulator sets speed to zero on this step.
                        commands[ship_id] = np.zeros(3, dtype=float)
                    else:
                        commands[ship_id] = self._movement(
                            ship_id,
                            ship,
                            command["turn"],
                            self.cruise_speed[ship_id] * self.fleet.dt,
                        )

                    continue

            if self.mode[ship_id] == "recovering":
                target = self.recovery_target[ship_id]

                if np.linalg.norm(position - target) > 1.0:
                    commands[ship_id] = self._towards(
                        ship_id, ship, target
                    )
                    continue

                self.next_waypoint[ship_id] = (
                    self.recovery_index[ship_id]
                )
                self.mode[ship_id] = "autonomous"
                print(f"REJOINED ship {ship_id}: following route again.")

            route = self.routes[ship_id]
            index = self.next_waypoint[ship_id]

            while (
                index < len(route) - 1
                and np.linalg.norm(position - route[index]) <= 1.0
            ):
                index += 1

            self.next_waypoint[ship_id] = index
            commands[ship_id] = self._towards(
                ship_id, ship, route[index]
            )

        return commands