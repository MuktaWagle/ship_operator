import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fleet_simulator import FleetSimulator


REGION = {
    "LON": (103.82, 103.88),
    "LAT": (1.15, 1.22),
}


def make_fleet(max_steps=20):
    trajectories = [
        np.array([
            [1000.0, 1000.0, 10.0, 0.0],
            [1100.0, 1000.0, 10.0, 0.0],
            [1200.0, 1000.0, 10.0, 0.0],
            [1300.0, 1000.0, 10.0, 0.0],
        ]),
        np.array([
            [1000.0, 2000.0, 10.0, 0.0],
            [1100.0, 2000.0, 10.0, 0.0],
            [1200.0, 2000.0, 10.0, 0.0],
            [1300.0, 2000.0, 10.0, 0.0],
        ]),
    ]

    timestamps = [
        np.array([0, 10, 20, 30]),
        np.array([0, 10, 20, 30]),
    ]

    return FleetSimulator(
        trajectories,
        timestamps,
        vessel_ids=[101, 102],
        region_of_interest=REGION,
        max_steps=max_steps,
    )


# 1. Arrival and retirement.
fleet = make_fleet()

try:
    fleet.reset()

    state = fleet.step({
        101: [150.0, 0.0, 0.0],
        102: [0.0, 0.0, 0.0],
    })

    # Ship 101 is now 150 m from its goal:
    # inside the original environment's 200 m arrival threshold.
    assert state["ships"][101]["status"] == "arrived"
    assert fleet.active_ship_ids == [102]
    assert not fleet.finished

    retired_state = state["ships"][101].copy()

    # Only the remaining active ship receives a command.
    state = fleet.step({
        102: [0.0, 0.0, 0.0],
    })

    assert state["ships"][101] == retired_state

    neighbors = (
        fleet.environments[102].current_obs["neighbors"][:, -1, :]
    )
    assert np.all(neighbors == 0)

    print("PASS: arrived ships stop moving and leave active traffic.")

    # Confirm distance records do not leak across resets.
    assert fleet.get_distance_metrics()["episode_minima"]
    fleet.reset()

    assert fleet.get_distance_metrics() == {
        "last_step": {},
        "episode_minima": {},
    }

    print("PASS: reset clears distance measurements.")

finally:
    fleet.close()


# 2. Episode time limit.
fleet = make_fleet(max_steps=2)

try:
    fleet.reset()

    for _ in range(2):
        state = fleet.step({
            101: [0.0, 0.0, 0.0],
            102: [0.0, 0.0, 0.0],
        })

    assert all(
        ship["status"] == "time limit"
        for ship in state["ships"].values()
    )
    assert fleet.finished
    assert fleet.active_ship_ids == []

    try:
        fleet.step({})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Stepping a finished fleet should fail.")

    print("PASS: time limits finish the episode and block further steps.")

finally:
    fleet.close()


# 3. Leaving the map using valid-sized movement commands.
fleet = make_fleet()

try:
    fleet.reset()

    for _ in range(11):
        state = fleet.step({
            101: [-100.0, 0.0, 0.0],
            102: [0.0, 0.0, 0.0],
        })

    assert state["ships"][101]["x"] == -100.0
    assert state["ships"][101]["status"] == "out of bounds"
    assert state["ships"][102]["status"] == "active"
    assert fleet.active_ship_ids == [102]

    print("PASS: a vessel leaving the map retires independently.")

finally:
    fleet.close()
