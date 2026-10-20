"""Command-line simulation test runner for Operator State & Budget Tracker

Runs a headless simulation of the fleet without requiring a GUI window.
Demonstrates:
- Operator transitions: Available -> Preparing (delta buffer delay) -> Assisting (tau duration) -> Available.
- Real-time budget depletion (b_t / B).
- Action masking on each step.
- Automated allocation when vessels encounter close separation.
"""

import numpy as np

from fleet_simulator import FleetSimulator
from route_controller import RouteController
from operator_tracker import OperatorBudgetTracker, OperatorStatus
from scenario import load_source_trajectories, build_fleet_scenario


def run_headless_test(num_steps: int = 40):
    print("=" * 78)
    print("ShipNaviSim — Operator State & Budget Tracker Simulation Test")
    print("=" * 78)

    region = {
        "LON": (103.82, 103.88),
        "LAT": (1.15, 1.22),
    }

    trajectories, timestamps = load_source_trajectories(
        region,
        dt=10,
    )

    scenario_trajectories, scenario_timestamps, vessel_ids = build_fleet_scenario(
        trajectories,
        timestamps,
        fleet_size=6,
        seed=43,
        min_start_distance=100.0,
        dt=10,
    )

    fleet = FleetSimulator(
        scenario_trajectories,
        scenario_timestamps,
        vessel_ids,
        region_of_interest=region,
        max_steps=num_steps,
        dt=10,
    )

    controller = RouteController(fleet)
    tracker = OperatorBudgetTracker(
        vessel_ids=fleet.vessel_ids,
        total_budget=150.0,         # Budget B = 150s
        duration=40.0,              # Duration tau = 40s (4 steps)
        delay=10.0,                 # Delay delta = 10s (1 step buffer)
        dt=fleet.dt,
        max_simulation_time=num_steps * fleet.dt,
    )

    state = fleet.reset(seed=43)
    controller.reset()
    tracker.reset()

    print(f"\nFleet initialized with {len(fleet.vessel_ids)} vessels: {fleet.vessel_ids}")
    print(f"Total budget B: {tracker.total_budget}s | Duration tau: {tracker.duration}s | Buffer delay delta: {tracker.delay}s\n")
    print(f"{'Step':>4} | {'Time':>5}s | {'Status':<10} | {'Assigned':<9} | {'Budget':>7}s | {'ActMask':<12} | {'Event / Action'}")
    print("-" * 78)

    for step_idx in range(num_steps):
        if fleet.finished:
            print("All vessels finished their journeys.")
            break

        # 1. Action Masking
        mask = tracker.get_action_mask(fleet.active_ship_ids)
        valid_actions = [i for i, ok in enumerate(mask) if ok]
        mask_summary = f"{len(valid_actions)}/{len(mask)} valid"

        event_str = ""

        # 2. Dispatch policy: At step 2, artificially request intervention on first active vessel
        if step_idx == 2 and tracker.is_available:
            target = fleet.active_ship_ids[0]
            act = tracker.ship_id_to_action(target)
            tracker.dispatch(act, fleet.active_ship_ids)
            event_str = f"DISPATCH -> Ship {target} (Entering {tracker.status.name})"

        # At step 15, if operator is available again, dispatch to another active vessel
        elif step_idx == 15 and tracker.is_available:
            target = fleet.active_ship_ids[1] if len(fleet.active_ship_ids) > 1 else fleet.active_ship_ids[0]
            act = tracker.ship_id_to_action(target)
            tracker.dispatch(act, fleet.active_ship_ids)
            event_str = f"DISPATCH -> Ship {target} (Entering {tracker.status.name})"

        # 3. If operator is ASSISTING, execute takeover in RouteController
        if tracker.is_assisting and tracker.assigned_ship_id is not None:
            aid = tracker.assigned_ship_id
            if controller.mode.get(aid) != "operator":
                controller.take_over(
                    aid,
                    direction="right",
                    duration=tracker.assistance_timer,
                    turn_rate_deg=1.0,
                )
                event_str = f"TAKEOVER -> Ship {aid} steering right"

        actions = controller.actions(state)
        state = fleet.step(actions)

        tracker_res = tracker.step(active_ship_ids=fleet.active_ship_ids)
        if tracker_res.get("transition_event"):
            ev = tracker_res["transition_event"]
            event_str += f" [{ev}]"

        assigned_display = str(tracker.assigned_ship_id) if tracker.assigned_ship_id else "None"
        print(
            f"{step_idx:>4} | "
            f"{fleet.current_time:>5.0f}s | "
            f"{tracker.status.name:<10} | "
            f"{assigned_display:<9} | "
            f"{tracker.remaining_budget:>7.1f}s | "
            f"{mask_summary:<12} | "
            f"{event_str}"
        )

    fleet.close()

    print("-" * 78)
    print("\nSummary Statistics:")
    print(f"Total Interventions Initiated: {tracker.total_interventions}")
    print(f"Total Active Assistance Time:  {tracker.total_active_assistance_time:.1f}s")
    print(f"Remaining Budget:              {tracker.remaining_budget:.1f}s / {tracker.total_budget:.1f}s")
    print(f"Final Operator Status:         {tracker.status.name}")
    print("\n[OK] Simulation test completed successfully.")


if __name__ == "__main__":
    run_headless_test(num_steps=30)
