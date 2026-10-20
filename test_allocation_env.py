"""Checks for the fleet allocation state-action-reward environment."""

import numpy as np

from allocation_env import FleetAllocationEnv, RewardWeights
from fleet_simulator import FleetSimulator
from operator_tracker import OperatorBudgetTracker, OperatorStatus
from route_controller import RouteController


REGION = {
    "LON": (103.82, 103.88),
    "LAT": (1.15, 1.22),
}


def make_env():
    trajectories = [
        np.array([
            [1000.0, 1000.0, 10.0, 0.0],
            [1100.0, 1000.0, 10.0, 0.0],
            [1200.0, 1000.0, 10.0, 0.0],
            [1300.0, 1000.0, 10.0, 0.0],
            [1400.0, 1000.0, 10.0, 0.0],
            [1500.0, 1000.0, 10.0, 0.0],
        ]),
        np.array([
            [1000.0, 1300.0, 10.0, 0.0],
            [1100.0, 1300.0, 10.0, 0.0],
            [1200.0, 1300.0, 10.0, 0.0],
            [1300.0, 1300.0, 10.0, 0.0],
            [1400.0, 1300.0, 10.0, 0.0],
            [1500.0, 1300.0, 10.0, 0.0],
        ]),
    ]
    timestamps = [
        np.array([0, 10, 20, 30, 40, 50]),
        np.array([0, 10, 20, 30, 40, 50]),
    ]

    fleet = FleetSimulator(
        trajectories,
        timestamps,
        vessel_ids=[101, 102],
        region_of_interest=REGION,
        max_steps=4,
        dt=10,
    )
    controller = RouteController(fleet)
    tracker = OperatorBudgetTracker(
        vessel_ids=[101, 102],
        total_budget=20,
        duration=10,
        delay=10,
        dt=10,
        max_simulation_time=40,
    )
    return FleetAllocationEnv(
        fleet,
        controller,
        tracker,
        warning_distance_m=250,
        collision_distance_m=50,
        reward_weights=RewardWeights(
            completed_journey=1.0,
            collision=10.0,
            near_miss=1.0,
            intervention=1.0,
        ),
    )


def test_reset_observation_and_action_mask():
    env = make_env()
    try:
        observation, info = env.reset(seed=43)

        assert env.observation_space.contains(observation)
        assert observation["ships"].shape == (2, 8)
        assert observation["operator"].shape == (8,)
        assert np.array_equal(info["action_mask"], [True, True, True])
        assert observation["ships"][0, 6] == 1.0
        assert observation["ships"][0, 7] == 0.0
    finally:
        env.close()


def test_dispatch_and_intervention_reward_are_step_local():
    env = make_env()
    try:
        env.reset(seed=43)

        _, reward, terminated, truncated, info = env.step(1)

        assert not terminated
        assert not truncated
        assert info["dispatched"] is True
        assert info["reward_components"]["interventions"] == 1
        assert reward == -1.0
        assert env.tracker.status == OperatorStatus.ASSISTING
        assert info["action_mask"][0]
        assert not info["action_mask"][1]
        assert not info["action_mask"][2]

        _, reward, _, _, info = env.step(0)
        assert info["reward_components"]["interventions"] == 0
        assert reward == 0.0
    finally:
        env.close()


def test_invalid_assignment_is_reported_without_crashing():
    env = make_env()
    try:
        env.reset(seed=43)
        env.step(1)

        _, reward, _, _, info = env.step(2)
        assert info["invalid_action"] is True
        assert info["dispatched"] is False
        assert reward == 0.0
    finally:
        env.close()


if __name__ == "__main__":
    test_reset_observation_and_action_mask()
    test_dispatch_and_intervention_reward_are_step_local()
    test_invalid_assignment_is_reported_without_crashing()
    print("All allocation environment tests PASSED!")
