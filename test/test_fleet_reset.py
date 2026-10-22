import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fleet_simulator import FleetSimulator


region = {
    "LON": (103.82, 103.88),
    "LAT": (1.15, 1.22),
}

# Two small artificial reference journeys.
# Each row contains: x, y, speed, heading.
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

times = [
    np.array([0, 10, 20, 30]),
    np.array([0, 10, 20, 30]),
]

fleet = FleetSimulator(
    trajectories,
    times,
    vessel_ids=[101, 102],
    region_of_interest=region,
)

try:
    initial = fleet.reset()
    print("Created fleet:", list(initial["ships"]))

    # Check that both ships can see the other's initial position.
    for observer_id, target_id in [(101, 102), (102, 101)]:
        observer = fleet.environments[observer_id]
        target = initial["ships"][target_id]

        expected = [
            target["x"],
            target["y"],
            target["speed"],
            target["heading"],
        ]

        neighbors = observer.current_obs["neighbors"][:, -1, :]

        assert any(
            np.allclose(row, expected)
            for row in neighbors
        )

    print("PASS: initial observations contain the other vessel.")

    # Editing a returned snapshot must not change the simulator.
    snapshot = fleet.get_state()
    snapshot["ships"][101]["x"] = -999.0

    assert fleet.get_state()["ships"][101]["x"] == 1000.0
    print("PASS: returned states are independent copies.")

    # Deliberately alter internal data to test reset.
    fleet.environments[101].current_state["x"] = 1500.0
    fleet.status[101] = "arrived"
    fleet.live_history[0][999] = np.zeros(4)

    restored = fleet.reset()

    assert restored == initial
    assert 999 not in fleet.live_history[0]
    print("PASS: reset restores states, statuses and shared history.")
    # Move one ship and hold the other stationary.
    moved = fleet.step({
        101: [50.0, 0.0, 0.0],
        102: [0.0, 0.0, 0.0],
    })

    assert moved["step"] == 1
    assert moved["timestamp"] == 10

    assert moved["ships"][101]["x"] == 1050.0
    assert moved["ships"][101]["speed"] == 5.0

    assert moved["ships"][102]["x"] == 1000.0
    assert moved["ships"][102]["speed"] == 0.0

    print("PASS: ships execute different commands on one shared clock.")

    # Ship 102 must observe ship 101's modified live position.
    neighbors = (
        fleet.environments[102].current_obs["neighbors"][:, -1, :]
    )

    assert any(
        np.allclose(row, [1050.0, 1000.0, 5.0, 0.0])
        for row in neighbors
    )

    print("PASS: neighbours observe the commanded movement.")

    # A missing command must fail before any ship moves.
    before_invalid_action = fleet.get_state()

    try:
        fleet.step({
            101: [10.0, 0.0, 0.0],
        })
    except ValueError:
        pass
    else:
        raise AssertionError("Missing vessel action was not rejected.")

    assert fleet.get_state() == before_invalid_action

    print("PASS: invalid action sets do not move the fleet.")

    # Reset after actual movement.
    assert fleet.reset() == initial

    print("PASS: reset works after movement.")

    # Stay stationary beyond the recorded journey's final timestamp.
    for _ in range(5):
        held = fleet.step({
            101: [0.0, 0.0, 0.0],
            102: [0.0, 0.0, 0.0],
        })

    assert held["timestamp"] == 50
    assert all(
        ship["status"] == "active"
        for ship in held["ships"].values()
    )

    neighbors = (
        fleet.environments[101].current_obs["neighbors"][:, -1, :]
    )

    assert any(
        np.allclose(row, [1000.0, 2000.0, 0.0, 0.0])
        for row in neighbors
    )

    print("PASS: live vessels remain observable beyond recorded journeys.")
finally:
    fleet.close()
