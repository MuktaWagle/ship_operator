"""Headless integration check for BC navigation plus operator allocation.

This checks the merged runtime path without training an RL policy:

    prepared BC scenario -> current FleetSimulator -> BCController
    -> OperatorBudgetTracker -> FleetAllocationEnv

At step 2 it dispatches one fixed allocation action, then returns to idle.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.allocation_env import FleetAllocationEnv, RewardWeights
from bc_navigation import BCController, FrozenBCPolicy, load_bc_scenario
from src.fleet_simulator import FleetSimulator
from src.operator_tracker import OperatorBudgetTracker


REGION_OF_INTEREST = {
    "LON": (103.82, 103.88),
    "LAT": (1.15, 1.22),
}


def run(ships=3, split="train", seed=43, steps=30):
    policy = FrozenBCPolicy()
    trajectories, timestamps, vessel_ids = load_bc_scenario(
        policy,
        fleet_size=ships,
        seed=seed,
        min_start_distance=800.0,
        split=split,
    )

    fleet = FleetSimulator(
        trajectories=trajectories,
        timestamps=timestamps,
        vessel_ids=vessel_ids,
        region_of_interest=REGION_OF_INTEREST,
        max_steps=steps,
        enable_graphics=False,
        dt=10,
    )
    controller = BCController(fleet, policy=policy)
    tracker = OperatorBudgetTracker(
        vessel_ids=fleet.vessel_ids,
        total_budget=150.0,
        duration=40.0,
        delay=10.0,
        dt=fleet.dt,
        max_simulation_time=steps * fleet.dt,
    )
    environment = FleetAllocationEnv(
        fleet,
        controller,
        tracker,
        warning_distance_m=500.0,
        collision_distance_m=50.0,
        reward_weights=RewardWeights(
            completed_journey=1.0,
            collision=10.0,
            near_miss=1.0,
            intervention=1.0,
        ),
        assistance_direction="right",
        assistance_turn_rate_deg=1.0,
        speed_scale_mps=10.0,
    )

    try:
        _, info = environment.reset(seed=seed)
        total_reward = 0.0

        print(f"Merged BC allocation test: {len(vessel_ids)} ships")
        print(f"Vessel IDs: {vessel_ids}")

        for step in range(steps):
            if environment.fleet.finished:
                break

            # Action 0 is idle. Action 1 assigns the first vessel.
            action = 1 if step == 2 and info["action_mask"][1] else 0
            _, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            print(
                f"step={step:02d} action={action} reward={reward:6.2f} "
                f"status={info['operator_state']['status_name']} "
                f"budget={info['operator_state']['budget']:.1f}s"
            )

            if terminated or truncated:
                break

        print(f"Total reward: {total_reward:.2f}")
        print("[OK] BC navigation and operator allocation are integrated.")
    finally:
        environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ships", type=int, default=3)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    run(
        ships=args.ships,
        split=args.split,
        seed=args.seed,
        steps=args.steps,
    )
