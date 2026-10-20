import numpy as np

from ship_env import ShipEnvironment

def closest_distance_during_step(a_start, a_end, b_start, b_end):
    """Minimum centre separation during straight movement within a step."""
    a_start = np.asarray(a_start, dtype=float)
    a_end = np.asarray(a_end, dtype=float)
    b_start = np.asarray(b_start, dtype=float)
    b_end = np.asarray(b_end, dtype=float)

    relative_start = a_start - b_start
    relative_change = (a_end - a_start) - (b_end - b_start)

    denominator = np.dot(relative_change, relative_change)

    if denominator <= 1e-12:
        fraction = 0.0
    else:
        fraction = float(np.clip(
            -np.dot(relative_start, relative_change) / denominator,
            0.0,
            1.0,
        ))

    distance = np.linalg.norm(
        relative_start + fraction * relative_change
    )

    return float(distance), fraction
class SharedShipEnvironment(ShipEnvironment):
    """Use shared simulated histories for controlled neighbors."""

    def __init__(self, *args, live_history, **kwargs):
        self.live_history = live_history
        super().__init__(*args, **kwargs)

        # This first implementation supports our current demo settings.
        if self.normalize_xy or self.scale_act or self.use_drift_fea:
            raise ValueError(
                "This demo requires normalize_xy=False, "
                "scale_act=False and use_drift_fea=False."
            )

    def _get_agent_observation(self, trajectory, time, target_timestamp):
        # Identify which vessel this requested trajectory belongs to.
        agent_id = next(
            (
                i for i, candidate in enumerate(self.ship_trajectories)
                if candidate is trajectory
            ),
            None,
        )

        # Keep the original data for the ego reference trajectory
        # and for background vessels.
        if agent_id == self.ego_pos or agent_id not in self.live_history:
            return super()._get_agent_observation(
                trajectory, time, target_timestamp
            )

        history = self.live_history[agent_id]
        rows = []

        # Match the original observation history length and time spacing.
        for offset in range(self.observation_history_length, -1, -1):
            timestamp = target_timestamp - offset * self.second_perts
            row = history.get(timestamp, self.padded_val)
            rows.append(np.array(row, dtype=float).copy())

        return np.array(rows)
class FleetSimulator:
    """Manage vessels sharing one scenario and simulation clock."""

    def __init__(
        self,
        trajectories,
        timestamps,
        vessel_ids,
        region_of_interest,
        max_steps=1000,
        enable_graphics=False,
        dt=10,
    ):
        if not (
            len(trajectories) == len(timestamps) == len(vessel_ids)
        ):
            raise ValueError(
                "Each vessel needs one trajectory, timestamp array and ID."
            )

        if not vessel_ids:
            raise ValueError("The fleet must contain at least one vessel.")

        if len(set(vessel_ids)) != len(vessel_ids):
            raise ValueError("Vessel IDs must be unique.")

        if dt <= 0 or max_steps < 1:
            raise ValueError("dt and max_steps must be positive.")

        self.trajectories = [
            np.asarray(trajectory, dtype=float).copy()
            for trajectory in trajectories
        ]
        self.timestamps = [
            np.asarray(times).copy()
            for times in timestamps
        ]
        self.vessel_ids = list(vessel_ids)

        for trajectory, times in zip(
            self.trajectories, self.timestamps
        ):
            if (
                trajectory.ndim != 2
                or trajectory.shape[1] != 4
                or len(trajectory) < 2
            ):
                raise ValueError(
                    "Each trajectory needs at least two rows "
                    "with x, y, speed and heading."
                )

            if times.ndim != 1 or len(trajectory) != len(times):
                raise ValueError(
                    "Trajectory rows and timestamps must match."
                )

            if not np.all(np.isfinite(trajectory)):
                raise ValueError("Trajectory values must be finite.")

            if not np.all(np.isfinite(times)):
                raise ValueError("Timestamps must be finite.")

            if not np.allclose(np.diff(times), dt):
                raise ValueError("All timestamps must use the fleet dt.")

        starts = [times[0] for times in self.timestamps]

        if not all(start == starts[0] for start in starts):
            raise ValueError(
                "This version requires all vessels to start together."
            )

        self.start_time = starts[0]
        self.dt = dt
        self.max_steps = max_steps
        self.live_history = {}
        self.environments = {}
        self.replay_actions = {}
        self.last_pair_distances = {}
        self.pair_minima = {}
        self.status = {}
        self.step_count = 0
        self.current_time = self.start_time
        self._has_reset = False

        # All fleet members remain potential neighbours, even if a
        # controller extends a journey beyond its recorded duration.
        all_neighbors = [
            [j for j in range(len(vessel_ids)) if j != i]
            for i in range(len(vessel_ids))
        ]

        for index, ship_id in enumerate(self.vessel_ids):
            self.environments[ship_id] = SharedShipEnvironment(
                self.trajectories,
                self.timestamps,
                all_neighbors,
                region_of_interest,
                ego_pos=index,
                n_neighbor_agents=max(1, len(vessel_ids) - 1),
                live_history=self.live_history,
                normalize_xy=False,
                scale_act=False,
                use_drift_fea=False,
                max_steps=max_steps,
                second_perts=dt,
                enable_graphics=enable_graphics,
            )

    def reset(self, seed=42):
        """Restart every vessel and clear the previous episode."""
        self._has_reset = False
        self.live_history.clear()
        self.replay_actions.clear()
        self.last_pair_distances.clear()
        self.pair_minima.clear()
        self.step_count = 0
        self.current_time = self.start_time
        self.status = {
            ship_id: "active"
            for ship_id in self.vessel_ids
        }

        # Publish every initial state before any vessel observes others.
        for index, trajectory in enumerate(self.trajectories):
            self.live_history[index] = {
                self.start_time: trajectory[0].copy()
            }

        for index, ship_id in enumerate(self.vessel_ids):
            env = self.environments[ship_id]

            _, info = env.reset(
                seed=seed,
                options={"ego_pos": index},
            )

            actions = info.get("actions")

            # The original environment expects valid reference actions
            # for some of its internal metrics.
            if actions is None or len(actions) == 0:
                raise ValueError(
                    f"Ship {ship_id} has no valid reference actions."
                )

            self.replay_actions[ship_id] = actions.copy()

            if env._check_goal_reached():
                self.status[ship_id] = "arrived"

        self._has_reset = True
        return self.get_state()

    def get_state(self):
        """Return independent copies of the fleet's current state."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before get_state().")

        return {
            "timestamp": self.current_time,
            "step": self.step_count,
            "ships": {
                ship_id: {
                    **env.current_state,
                    "goal": tuple(env.goal_position),
                    "status": self.status[ship_id],
                }
                for ship_id, env in self.environments.items()
            },
        }
    @property
    def active_ship_ids(self):
        """Return the IDs of ships that still need commands."""
        if not self._has_reset:
            raise RuntimeError("Call reset() first.")

        return [
            ship_id
            for ship_id in self.vessel_ids
            if self.status[ship_id] == "active"
        ]

    @property
    def finished(self):
        """Return True when no ships remain active."""
        return len(self.active_ship_ids) == 0
    def step(self, actions):
        """Advance all active vessels by one shared time interval."""
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")

        active_ids = self.active_ship_ids

        if not active_ids:
            raise RuntimeError(
                "No active vessels remain. Call reset() for a new episode."
            )

        # Every active vessel needs exactly one command.
        if set(actions) != set(active_ids):
            raise ValueError(
                f"Provide actions for exactly these active ships: "
                f"{active_ids}"
            )

        # Validate ALL commands before changing any vessel.
        commands = {}

        for ship_id in active_ids:
            command = np.asarray(actions[ship_id], dtype=float)

            if command.shape != (3,) or not np.all(np.isfinite(command)):
                raise ValueError(
                    f"Ship {ship_id}: action must contain three finite "
                    "numbers: dx, dy, dheading."
                )

            env = self.environments[ship_id]

            if (
                np.any(command < env.action_space.low)
                or np.any(command > env.action_space.high)
            ):
                raise ValueError(
                    f"Ship {ship_id}: action exceeds simulator bounds."
                )

            if env.current_state["timestamp"] != self.current_time:
                raise RuntimeError("Vessel clocks are not synchronized.")

            commands[ship_id] = command.copy()
        positions_before = {
            ship_id: np.array([
                self.environments[ship_id].current_state["x"],
                self.environments[ship_id].current_state["y"],
            ], dtype=float)
            for ship_id in active_ids
        }
        next_time = self.current_time + self.dt

        # Publish every next state before any vessel builds its observation.
        # These equations match the original unscaled movement model.
        for ship_id in active_ids:
            env = self.environments[ship_id]
            state = env.current_state
            dx, dy, dheading = commands[ship_id]

            self.live_history[env.ego_pos][next_time] = np.array([
                state["x"] + dx,
                state["y"] + dy,
                np.hypot(dx, dy) / self.dt,
                (state["heading"] + dheading) % (2 * np.pi),
            ])

        # Previously finished vessels no longer participate in traffic.
        for ship_id in self.vessel_ids:
            if ship_id not in commands:
                env = self.environments[ship_id]
                self.live_history[env.ego_pos][next_time] = (
                    np.array(env.padded_val, dtype=float).copy()
                )

        for ship_id in active_ids:
            env = self.environments[ship_id]

            _, _, terminated, truncated, info = env.step(
                commands[ship_id]
            )

            state = env.current_state

            np.testing.assert_allclose(
                self.live_history[env.ego_pos][next_time],
                [
                    state["x"],
                    state["y"],
                    state["speed"],
                    state["heading"],
                ],
                rtol=0,
                atol=1e-6,
            )

            if state["timestamp"] != next_time:
                raise RuntimeError("A vessel advanced to the wrong time.")

            # Give leaving the region priority over arrival.
            if not info["within_region"]:
                self.status[ship_id] = "out of bounds"
            elif terminated:
                self.status[ship_id] = "arrived"
            elif truncated:
                self.status[ship_id] = "time limit"

        # Measure every pair that participated in this interval.
        # This includes ships that arrived during this step.
        self.last_pair_distances = {}

        for i, ship_a_id in enumerate(active_ids):
            for ship_b_id in active_ids[i + 1:]:
                a = self.environments[ship_a_id].current_state
                b = self.environments[ship_b_id].current_state

                distance, fraction = closest_distance_during_step(
                    positions_before[ship_a_id],
                    [a["x"], a["y"]],
                    positions_before[ship_b_id],
                    [b["x"], b["y"]],
                )

                pair = tuple(sorted((ship_a_id, ship_b_id)))

                measurement = {
                    "distance_m": distance,
                    "elapsed_seconds": (
                        self.step_count + fraction
                    ) * self.dt,
                }

                self.last_pair_distances[pair] = measurement.copy()

                previous = self.pair_minima.get(pair)

                if (
                    previous is None
                    or distance < previous["distance_m"]
                ):
                    self.pair_minima[pair] = measurement.copy()


        self.step_count += 1
        self.current_time = next_time

        return self.get_state()
    def get_distance_metrics(self):
        """Return copies of measurements from completed steps."""
        if not self._has_reset:
            raise RuntimeError("Call reset() first.")

        return {
            "last_step": {
                pair: result.copy()
                for pair, result in self.last_pair_distances.items()
            },
            "episode_minima": {
                pair: result.copy()
                for pair, result in self.pair_minima.items()
            },
        }
    def close(self):
        """Release the original simulator's display resources."""
        for env in self.environments.values():
            env.close()
        self._has_reset = False