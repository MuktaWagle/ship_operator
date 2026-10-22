"""Tests for operator state and budget tracker"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operator_tracker import OperatorBudgetTracker, OperatorStatus
from fleet_simulator import FleetSimulator
from test.support import HoldController


def test_initialization():
    """Test initial parameters, defaults, and input validations."""
    vessel_ids = [101, 102, 103]
    tracker = OperatorBudgetTracker(
        vessel_ids=vessel_ids,
        total_budget=300,
        duration=60,
        delay=10,
        dt=10,
    )

    state = tracker.get_state()
    assert state["budget"] == 300.0
    assert state["total_budget"] == 300.0
    assert state["budget_ratio"] == 1.0
    assert state["status"] == OperatorStatus.AVAILABLE
    assert state["status_name"] == "AVAILABLE"
    assert state["assigned_ship_id"] is None
    assert state["prep_time_remaining"] == 0.0
    assert state["assist_time_remaining"] == 0.0
    assert state["simulation_time"] == 0.0
    assert state["step"] == 0
    assert state["interventions_initiated"] == 0
    assert state["total_interventions"] == 0
    assert state["total_active_time"] == 0.0
    assert state["can_assign"] is True
    assert tracker.is_available is True
    assert tracker.is_preparing is False
    assert tracker.is_assisting is False

    try:
        OperatorBudgetTracker(vessel_ids=[])
        assert False, "Should raise ValueError on empty vessel list."
    except ValueError:
        pass

    try:
        OperatorBudgetTracker(vessel_ids=[101, 101])
        assert False, "Should raise ValueError on duplicate vessel IDs."
    except ValueError:
        pass

    try:
        OperatorBudgetTracker(vessel_ids=[101], duration=25, dt=10)
        assert False, "Should raise ValueError when duration is not a multiple of dt."
    except ValueError:
        pass

    print("PASS: initialization and parameter validation.")


def test_action_mapping_and_masking():
    """Test action indexing, ship ID mapping, and action masking."""
    vessel_ids = [101, 102, 103]
    tracker = OperatorBudgetTracker(
        vessel_ids=vessel_ids,
        total_budget=100,
        duration=50,
        delay=10,
        dt=10,
    )

    assert tracker.action_to_ship_id(0) is None
    assert tracker.action_to_ship_id(1) == 101
    assert tracker.action_to_ship_id(2) == 102
    assert tracker.action_to_ship_id(3) == 103

    assert tracker.ship_id_to_action(None) == 0
    assert tracker.ship_id_to_action(101) == 1
    assert tracker.ship_id_to_action(102) == 2
    assert tracker.ship_id_to_action(103) == 3

    # Action mask when Available and all active:
    mask = tracker.get_action_mask(active_ship_ids=[101, 102, 103])
    assert mask.shape == (4,)
    assert np.all(mask == [True, True, True, True])

    # Action mask when ship 102 has retired/arrived:
    mask = tracker.get_action_mask(active_ship_ids=[101, 103])
    assert np.array_equal(mask, [True, True, False, True])

    # When operator is busy
    tracker.dispatch(action=1, active_ship_ids=[101, 103])
    assert tracker.status == OperatorStatus.PREPARING
    mask_busy = tracker.get_action_mask(active_ship_ids=[101, 103])
    assert np.array_equal(mask_busy, [True, False, False, False])
    assert tracker.is_action_valid(0) is True
    assert tracker.is_action_valid(1) is False

    print("PASS: action mapping and action masking.")


def test_delay_and_assistance_transitions():
    """Test the full lifecycle: Available -> Preparing -> Assisting -> Available."""
    vessel_ids = [101, 102]
    tracker = OperatorBudgetTracker(
        vessel_ids=vessel_ids,
        total_budget=200,
        duration=30,  # 30 seconds (3 steps of 10s)
        delay=20,     # 20 seconds (2 steps of 10s)
        dt=10,
    )

    # Step 0: Dispatch intervention to ship 102
    dispatched = tracker.dispatch(action=2, active_ship_ids=[101, 102])
    assert dispatched is True
    assert tracker.status == OperatorStatus.PREPARING
    assert tracker.is_preparing is True
    assert tracker.assigned_ship_id == 102
    assert tracker.preparation_timer == 20.0
    assert tracker.assistance_timer == 30.0
    assert tracker.interventions_initiated_this_step == 1
    assert tracker.total_interventions == 1

    # Step 1: Advance by dt=10 (still preparing)
    res = tracker.step(active_ship_ids=[101, 102])
    assert tracker.status == OperatorStatus.PREPARING
    assert tracker.preparation_timer == 10.0
    assert tracker.assistance_timer == 30.0
    assert tracker.remaining_budget == 200.0  # budget not consumed during prep
    assert tracker.interventions_initiated_this_step == 0
    assert tracker.simulation_time == 10.0
    assert tracker.step_count == 1

    # Step 2: Advance by dt=10 (prep finishes -> transitions to Assisting)
    res = tracker.step(active_ship_ids=[101, 102])
    assert tracker.status == OperatorStatus.ASSISTING
    assert tracker.is_assisting is True
    assert tracker.preparation_timer == 0.0
    assert tracker.assistance_timer == 30.0
    assert "ASSISTANCE_STARTED_102" in str(res["transition_event"])

    # Step 3: Advance by dt=10 (assisting step 1: consumes 10s budget)
    res = tracker.step(active_ship_ids=[101, 102])
    assert tracker.status == OperatorStatus.ASSISTING
    assert tracker.assistance_timer == 20.0
    assert tracker.remaining_budget == 190.0
    assert tracker.total_active_assistance_time == 10.0

    # Step 4: Advance by dt=10 (assisting step 2: consumes 10s budget)
    res = tracker.step(active_ship_ids=[101, 102])
    assert tracker.status == OperatorStatus.ASSISTING
    assert tracker.assistance_timer == 10.0
    assert tracker.remaining_budget == 180.0
    assert tracker.total_active_assistance_time == 20.0

    # Step 5: Advance by dt=10 (assisting step 3: duration expires -> transitions to Available)
    res = tracker.step(active_ship_ids=[101, 102])
    assert tracker.status == OperatorStatus.AVAILABLE
    assert tracker.is_available is True
    assert tracker.assigned_ship_id is None
    assert tracker.remaining_budget == 170.0
    assert tracker.total_active_assistance_time == 30.0
    assert "DURATION_EXPIRED_102" in str(res["transition_event"])

    print("PASS: delay buffer and assistance state transitions.")


def test_immediate_activation_zero_delay():
    """Test intervention when delay delta = 0."""
    tracker = OperatorBudgetTracker(
        vessel_ids=[101],
        total_budget=100,
        duration=20,
        delay=0,
        dt=10,
    )

    tracker.dispatch(action=1)
    assert tracker.status == OperatorStatus.ASSISTING
    assert tracker.is_assisting is True
    assert tracker.preparation_timer == 0.0
    assert tracker.assistance_timer == 20.0

    tracker.step()
    assert tracker.status == OperatorStatus.ASSISTING
    assert tracker.remaining_budget == 90.0

    tracker.step()
    assert tracker.status == OperatorStatus.AVAILABLE
    assert tracker.remaining_budget == 80.0

    print("PASS: immediate activation with zero delay.")


def test_budget_exhaustion():
    """Test behavior when assistance budget is completely exhausted."""
    tracker = OperatorBudgetTracker(
        vessel_ids=[101, 102],
        total_budget=30,  # 30 seconds
        duration=30,
        delay=0,
        dt=10,
    )

    # First intervention uses all 30s
    tracker.dispatch(action=1)
    for _ in range(3):
        tracker.step()

    assert tracker.remaining_budget == 0.0
    assert tracker.status == OperatorStatus.AVAILABLE
    assert tracker.has_sufficient_budget is False

    # Budget is exhausted: mask should only allow action 0
    mask = tracker.get_action_mask(active_ship_ids=[101, 102])
    assert np.array_equal(mask, [True, False, False])

    # Trying to dispatch should fail
    assert tracker.is_action_valid(1) is False
    try:
        tracker.dispatch(1, strict=True)
        assert False, "Should not dispatch when budget is exhausted."
    except ValueError:
        pass

    # Non-strict dispatch returns False
    assert tracker.dispatch(1, strict=False) is False

    print("PASS: budget exhaustion handling and masking.")


def test_early_release_on_ship_arrival():
    """Test that if the assisted vessel arrives or retires, operator is freed."""
    tracker = OperatorBudgetTracker(
        vessel_ids=[101, 102],
        total_budget=100,
        duration=40,
        delay=10,
        dt=10,
    )

    # Dispatch to ship 101
    tracker.dispatch(action=1, active_ship_ids=[101, 102])
    tracker.step(active_ship_ids=[101, 102])  # Prep finished -> Assisting

    assert tracker.status == OperatorStatus.ASSISTING
    assert tracker.assigned_ship_id == 101

    # Suppose on the next step ship 101 arrives/retires (active ships now only [102])
    res = tracker.step(active_ship_ids=[102])

    assert tracker.status == OperatorStatus.AVAILABLE
    assert tracker.assigned_ship_id is None
    assert "EARLY_RELEASE_RETIRED_101" in str(res["transition_event"])

    # Operator is immediately available for ship 102
    assert tracker.is_available is True
    mask = tracker.get_action_mask(active_ship_ids=[102])
    assert np.array_equal(mask, [True, False, True])

    print("PASS: early release on vessel arrival/retirement.")


def test_observation_vector():
    """Test RL observation vector format and values."""
    tracker = OperatorBudgetTracker(
        vessel_ids=[101, 102],
        total_budget=100,
        duration=40,
        delay=20,
        dt=10,
        max_simulation_time=1000,
    )

    vec = tracker.get_observation_vector(normalize=True)
    assert vec.shape == (7,)
    assert vec.dtype == np.float32
    # [norm_budget=1.0, avail=1, prep=0, assist=0, norm_prep=0, norm_assist=0, norm_time=0]
    expected = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(vec, expected)

    # Dispatch to enter Preparing
    tracker.dispatch(1)
    vec = tracker.get_observation_vector(normalize=True)
    # [norm_budget=1.0, avail=0, prep=1, assist=0, norm_prep=1.0, norm_assist=1.0, norm_time=0]
    expected = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    assert np.allclose(vec, expected)

    print("PASS: observation vector generation.")


def test_integration_with_simulator_and_controller():
    """Integration test: driving FleetSimulator with OperatorBudgetTracker."""
    region = {
        "LON": (103.82, 103.88),
        "LAT": (1.15, 1.22),
    }

    # Two test trajectories
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
            [1000.0, 2000.0, 10.0, 0.0],
            [1100.0, 2000.0, 10.0, 0.0],
            [1200.0, 2000.0, 10.0, 0.0],
            [1300.0, 2000.0, 10.0, 0.0],
            [1400.0, 2000.0, 10.0, 0.0],
            [1500.0, 2000.0, 10.0, 0.0],
        ]),
    ]

    times = [
        np.array([0, 10, 20, 30, 40, 50]),
        np.array([0, 10, 20, 30, 40, 50]),
    ]

    fleet = FleetSimulator(
        trajectories,
        times,
        vessel_ids=[101, 102],
        region_of_interest=region,
        max_steps=5,
    )

    try:
        state = fleet.reset(seed=42)
        controller = HoldController(fleet)
        tracker = OperatorBudgetTracker(
            vessel_ids=fleet.vessel_ids,
            total_budget=50,
            duration=20,  # 20s = 2 steps
            delay=10,     # 10s = 1 step
            dt=fleet.dt,
        )

        step_idx = 0
        while not fleet.finished and step_idx < 4:
            # Policy decision (e.g. at step 0, dispatch intervention to ship 101)
            mask = tracker.get_action_mask(fleet.active_ship_ids)
            if step_idx == 0:
                assert mask[1]
                tracker.dispatch(action=1, active_ship_ids=fleet.active_ship_ids)

            # Check if tracker transitioned to active assistance
            if tracker.is_assisting and controller.mode[tracker.assigned_ship_id] != "operator":
                # Operator takes over ship 101 with right turn
                controller.take_over(
                    tracker.assigned_ship_id,
                    direction="right",
                    duration=tracker.duration,
                    turn_rate_deg=1.0,
                )

            # Generate actions from controller
            actions = controller.actions(state)
            state = fleet.step(actions)

            # Advance tracker with active ships
            tracker.step(active_ship_ids=fleet.active_ship_ids)
            step_idx += 1

        assert tracker.total_interventions == 1
        assert tracker.total_active_assistance_time == 20.0
        assert tracker.remaining_budget == 30.0
        print("PASS: integration with FleetSimulator and test controller.")

    finally:
        fleet.close()


if __name__ == "__main__":
    test_initialization()
    test_action_mapping_and_masking()
    test_delay_and_assistance_transitions()
    test_immediate_activation_zero_delay()
    test_budget_exhaustion()
    test_early_release_on_ship_arrival()
    test_observation_vector()
    test_integration_with_simulator_and_controller()
    print("\nAll Operator State and Budget Tracker tests PASSED!")
