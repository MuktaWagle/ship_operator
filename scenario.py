"""Prepare source routes and reproducible simultaneous fleet scenarios."""
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ship_env import read_position_data, chunk_to_traj
from split_chunks import seperate_chunks


def load_source_trajectories(region, dt):
    region_array = np.array([
        [region["LON"][0], region["LAT"][0]],
        [region["LON"][1], region["LAT"][1]],
    ])

    df = read_position_data(str(Path(__file__).resolve().parent / "raw_data" / "synthetic_ais_data.csv"))

    filters = [
        pl.col("LON") >= region["LON"][0],
        pl.col("LON") <= region["LON"][1],
        pl.col("LAT") >= region["LAT"][0],
        pl.col("LAT") <= region["LAT"][1],
        pl.col("HEADING").is_not_null(),
        pl.col("HEADING") != 511,
        pl.col("STATUS") == 0,
    ]

    df = df.filter(*filters)

    # Make preprocessing order explicit.
    df = df.sort(["SHIP_ID", "TIMESTAMP_UTC"])

    chunks = seperate_chunks(
        df,
        timedelta(minutes=30),
        5,
    )

    # Use a stable ordering when several journeys start together.
    chunks = chunks.sort(["start_time", "SHIP_ID", "chunk_id"])

    trajectories = []
    timestamps = []

    for index in range(len(chunks)):
        trajectory, times = chunk_to_traj(
            chunks[index],
            timedelta(seconds=dt),
            region_array,
        )

        trajectories.append(trajectory)
        timestamps.append(times)

    if not trajectories:
        raise RuntimeError("No usable trajectories were prepared.")

    print(f"Prepared {len(trajectories)} source trajectory segments.")

    return trajectories, timestamps


# ------------------------------------------------------------
# Construct simultaneous synthetic traffic
# ------------------------------------------------------------

def build_fleet_scenario(
    source_trajectories,
    source_times,
    fleet_size,
    seed=42,
    min_start_distance=100.0,
    dt=10,
):
    if not 1 <= fleet_size <= len(source_trajectories):
        raise ValueError(
            f"Requested {fleet_size} ships, but only "
            f"{len(source_trajectories)} source segments are available."
        )

    rng = np.random.default_rng(seed)
    best_selection = []

    for _ in range(200):
        selected = []

        for index in rng.permutation(len(source_trajectories)):
            index = int(index)
            trajectory = source_trajectories[index]

            if len(trajectory) < 12:
                continue

            position = trajectory[0, :2]

            separated = all(
                np.linalg.norm(
                    position - source_trajectories[other][0, :2]
                ) >= min_start_distance
                for other in selected
            )

            if separated:
                selected.append(index)

            if len(selected) == fleet_size:
                break

        if len(selected) > len(best_selection):
            best_selection = selected.copy()

        if len(selected) == fleet_size:
            break

    if len(best_selection) < fleet_size:
        raise ValueError(
            f"The selection search found only {len(best_selection)} "
            f"ships with {min_start_distance:g} m starting separation. "
            f"The requested {fleet_size}-ship scenario was not constructed. "
            "This does not prove that a larger selection is impossible."
        )

    common_start = int(min(times[0] for times in source_times))

    fleet_trajectories = []
    fleet_times = []
    vessel_ids = []

    for number, source_index in enumerate(best_selection):
        trajectory = source_trajectories[source_index].copy()
        original_times = np.asarray(source_times[source_index])

        shifted_times = (
            original_times - original_times[0] + common_start
        )

        if not np.all(np.diff(shifted_times) == dt):
            raise ValueError(
                f"Source trajectory {source_index} "
                f"is not sampled every {dt} seconds."
            )

        ship_id = 10000 + number

        fleet_trajectories.append(trajectory)
        fleet_times.append(shifted_times)
        vessel_ids.append(ship_id)

        print(
            f"Simulated ship {ship_id} uses "
            f"source trajectory index {source_index}"
        )

    return fleet_trajectories, fleet_times, vessel_ids


