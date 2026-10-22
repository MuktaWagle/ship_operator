"""Run a simple nearest-ship allocation policy and log every SAR step."""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bc_navigation import BCController, FrozenBCPolicy, load_bc_scenario
from src.allocation_env import FleetAllocationEnv, RewardWeights
from src.fleet_simulator import FleetSimulator
from src.operator_tracker import OperatorBudgetTracker


REGION = {
    "LON": (103.82, 103.88),
    "LAT": (1.15, 1.22),
}


def simple_allocation_policy(observation, action_mask):
    """Assign the operator to the closest active ship, otherwise stay idle."""
    candidates = [
        (float(observation["nearest_distance"][index]), index + 1)
        for index, ship in enumerate(observation["ships"])
        if ship[6] > 0.5 and action_mask[index + 1]
    ]
    return min(candidates)[1] if candidates else 0


def jsonable(value):
    """Convert NumPy values and nested containers into JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def run(ships=3, steps=30, seed=43):
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "simple_allocation_policy.txt"

    policy = FrozenBCPolicy()
    trajectories, timestamps, vessel_ids = load_bc_scenario(
        policy,
        fleet_size=ships,
        seed=seed,
        min_start_distance=800.0,
        split="train",
    )

    fleet = FleetSimulator(
        trajectories,
        timestamps,
        vessel_ids,
        region_of_interest=REGION,
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
        reward_weights=RewardWeights(),
        speed_scale_mps=10.0,
    )

    total_reward = 0.0
    records = []
    try:
        observation, info = environment.reset(seed=seed)

        for step in range(steps):
            if environment.fleet.finished:
                break

            action = simple_allocation_policy(
                observation,
                info["action_mask"],
            )
            next_observation, reward, terminated, truncated, next_info = (
                environment.step(action)
            )
            total_reward += reward
            records.append({
                "step": step,
                "action": action,
                "observation": jsonable(observation),
                "action_mask": jsonable(info["action_mask"]),
                "reward": reward,
                "reward_components": jsonable(
                    next_info["reward_components"]
                ),
                "operator_state": jsonable(
                    next_info["operator_state"]
                ),
                "event_pairs": jsonable(next_info["event_pairs"]),
                "next_observation": jsonable(next_observation),
                "terminated": terminated,
                "truncated": truncated,
            })
            observation, info = next_observation, next_info

            if terminated or truncated:
                break
    finally:
        environment.close()

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Simple nearest-ship allocation policy\n")
        log_file.write(f"ships={ships}, seed={seed}, steps={len(records)}\n")
        log_file.write(f"total_reward={total_reward:.3f}\n\n")
        for record in records:
            log_file.write(json.dumps(record, sort_keys=True))
            log_file.write("\n")

    print(f"Log written to {log_path}")
    print(f"Total reward: {total_reward:.3f}")
    return log_path


if __name__ == "__main__":
    run(5, 100, 43)
