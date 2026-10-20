"""Scenario loader for the prepared BC journeys."""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_bc_scenario(
    policy,
    data=None,
    fleet_size=10,
    seed=43,
    min_start_distance=800.0,
    split="train",
):
    """Load synchronized trajectories compatible with the current fleet API."""
    data = Path(data) if data is not None else Path(__file__).resolve().parent / "data_bc"

    manifest_path = data / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != policy.manifest_sha256:
        raise ValueError("BC dataset manifest differs from checkpoint.")

    manifest = json.loads(manifest_bytes)
    if split not in ("train", "val", "test"):
        raise ValueError("Choose train, val or test split.")

    candidates = []
    with np.load(data / "journeys.npz", allow_pickle=False) as journeys:
        for journey in manifest["journeys"]:
            if journey["split"] != split:
                continue

            positions = journeys[f"positions_{journey['journey_id']}"]
            delta = np.diff(positions, axis=0)
            speed = np.linalg.norm(delta, axis=1) / manifest["dt"]
            if not 0.01 <= speed[0] <= manifest["max_speed_ms"]:
                continue

            heading = np.arctan2(delta[:, 1], delta[:, 0])
            trajectory = np.column_stack((positions[1:], speed, heading))
            candidates.append((journey["ship_id"], trajectory))

    if not 1 <= fleet_size <= len(candidates):
        raise ValueError(
            f"Requested {fleet_size}; split {split} has "
            f"{len(candidates)} usable ships."
        )

    rng = np.random.default_rng(seed)
    best = []
    for _ in range(200):
        selected = []
        for index in rng.permutation(len(candidates)):
            index = int(index)
            position = candidates[index][1][0, :2]
            if all(
                np.linalg.norm(position - candidates[other][1][0, :2])
                >= min_start_distance
                for other in selected
            ):
                selected.append(index)
            if len(selected) == fleet_size:
                break

        if len(selected) > len(best):
            best = selected
        if len(best) == fleet_size:
            break

    if len(best) < fleet_size:
        raise ValueError(
            f"Found {len(best)} separated ships in {split}. "
            "Reduce fleet_size or min_start_distance."
        )

    vessel_ids = [int(candidates[index][0]) for index in best]
    trajectories = [candidates[index][1] for index in best]
    timestamps = [
        np.arange(len(trajectory), dtype=np.int64) * int(manifest["dt"])
        for trajectory in trajectories
    ]

    print(
        f"BC scenario: split={split}, original ship IDs={vessel_ids}; "
        "synchronized synthetic start times."
    )
    return trajectories, timestamps, vessel_ids
