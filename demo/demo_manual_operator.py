"""
Manual operator demo.

Controls:
    TAB          Select the next active ship
    Click row    Select a ship from the right-hand panel
    LEFT / A     Turn the selected ship left
    RIGHT / D    Turn the selected ship right
    S            Stop and hold the selected ship
    R            Release control and resume BC navigation
    UP / DOWN    Increase / decrease command duration
    SPACE        Pause / resume
    ESC          Exit

Only one ship can be under manual operator control at a time.
A ship running BC navigation does not occupy the operator.

This demo uses the frozen behavioural-cloning navigator and FleetSimulator.
It does not implement collision avoidance or intervention decisions.
"""

import numpy as np
import pygame
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fleet_simulator import FleetSimulator
from bc_navigation import BCController, FrozenBCPolicy, load_bc_scenario


FLEET_SIZE = 6
SCENARIO_SEED = 43
DT = 10
MAX_STEPS = 1000
WARNING_DISTANCE_M = 500.0
MIN_START_DISTANCE_M = 100.0

# One simulation step represents 10 simulated seconds.
# Display speed is independent of the simulation step duration.
STEPS_PER_REAL_SECOND = 3

TURN_RATE_DEG = 1.0
INITIAL_COMMAND_DURATION = 100
MAX_COMMAND_DURATION = 240

REGION_OF_INTEREST = {
    "LON": (103.82, 103.88),
    "LAT": (1.15, 1.22),
}

WINDOW_WIDTH = 1450
WINDOW_HEIGHT = 950
MAP_WIDTH = 900

ROW_TOP = 205
ROW_HEIGHT = 45


def ship_colour(index, count):
    colour = pygame.Color(0)
    colour.hsva = (
        index * 360.0 / max(1, count),
        75,
        75,
        100,
    )
    return colour.r, colour.g, colour.b


def current_operator_ship(fleet, controller):
    """Find the active vessel currently using the operator."""
    for ship_id in fleet.active_ship_ids:
        if controller.mode[ship_id] == "operator":
            return ship_id
    return None

def calculate_live_distances(state):
    """Calculate current distances between active ships only."""
    active_ids = [
        ship_id
        for ship_id, ship in state["ships"].items()
        if ship["status"] == "active"
    ]

    pairs = {}
    nearest = {}

    for index, ship_a_id in enumerate(active_ids):
        ship_a = state["ships"][ship_a_id]

        for ship_b_id in active_ids[index + 1:]:
            ship_b = state["ships"][ship_b_id]

            distance = float(np.hypot(
                ship_a["x"] - ship_b["x"],
                ship_a["y"] - ship_b["y"],
            ))

            pair = tuple(sorted((ship_a_id, ship_b_id)))
            pairs[pair] = distance

            for own_id, other_id in (
                (ship_a_id, ship_b_id),
                (ship_b_id, ship_a_id),
            ):
                if (
                    own_id not in nearest
                    or distance < nearest[own_id]["distance_m"]
                ):
                    nearest[own_id] = {
                        "ship_id": other_id,
                        "distance_m": distance,
                    }

    return pairs, nearest


def report_distance_alerts(
    fleet, state, previous_warning_pairs
):
    """Print warning transitions once per simulation step."""
    pairs, _ = calculate_live_distances(state)

    warning_pairs = {
        pair
        for pair, distance in pairs.items()
        if distance < WARNING_DISTANCE_M
    }

    elapsed = float(state["timestamp"] - fleet.start_time)

    for pair in sorted(warning_pairs - previous_warning_pairs):
        print(
            f"[{elapsed:.0f}s] DISTANCE WARNING: "
            f"ships {pair[0]} and {pair[1]} are "
            f"{pairs[pair]:.1f} m apart "
            f"(threshold {WARNING_DISTANCE_M:.0f} m)."
        )

    for pair in sorted(previous_warning_pairs - warning_pairs):
        if pair in pairs:
            print(
                f"[{elapsed:.0f}s] DISTANCE WARNING CLEARED: "
                f"ships {pair[0]} and {pair[1]} are now "
                f"{pairs[pair]:.1f} m apart."
            )
        else:
            print(
                f"[{elapsed:.0f}s] DISTANCE MONITOR ENDED: "
                f"pair {pair[0]}/{pair[1]} no longer has "
                "two active ships."
            )

    # Detect an encounter between displayed positions.
    # Avoid repeating a warning for pairs already below the threshold
    # at either endpoint.
    last_step = fleet.get_distance_metrics()["last_step"]

    for pair, result in sorted(last_step.items()):
        if (
            result["distance_m"] < WARNING_DISTANCE_M
            and pair not in warning_pairs
            and pair not in previous_warning_pairs
        ):
            print(
                f"[{elapsed:.0f}s] CLOSE ENCOUNTER DURING LAST STEP: "
                f"ships {pair[0]} and {pair[1]} reached "
                f"{result['distance_m']:.1f} m separation "
                f"at {result['elapsed_seconds']:.1f}s."
            )

    return warning_pairs
def draw_display(
    window,
    fleet,
    controller,
    state,
    trails,
    selected_ship,
    command_duration,
    paused,
    message,
    font,
    small_font,
):
    window.fill((232, 244, 250))
    live_pairs, nearest = calculate_live_distances(state)

    warning_pairs = {
        pair: distance
        for pair, distance in live_pairs.items()
        if distance < WARNING_DISTANCE_M
    }

    warning_ship_ids = {
        ship_id
        for pair in warning_pairs
        for ship_id in pair
    }

    # Blink every half-second of real time.
    # This does not affect the simulation clock.
    blink_on = (pygame.time.get_ticks() // 500) % 2 == 0

    last_step = fleet.get_distance_metrics()["last_step"]
    recent_encounters = {
        pair: result
        for pair, result in last_step.items()
        if result["distance_m"] < WARNING_DISTANCE_M
    }
    reference = next(iter(fleet.environments.values()))
    margin = 55

    scale = min(
        (MAP_WIDTH - 2 * margin) / reference.max_x,
        (WINDOW_HEIGHT - 150) / reference.max_y,
    )

    def screen_position(position):
        x, y = position
        return (
            int(margin + x * scale),
            int(WINDOW_HEIGHT - 65 - y * scale),
        )

    top_left = screen_position((0, reference.max_y))
    bottom_right = screen_position((reference.max_x, 0))

    pygame.draw.rect(
        window,
        (110, 140, 160),
        pygame.Rect(
            top_left[0],
            top_left[1],
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
        ),
        2,
    )

    # Prepared BC journeys are shown for visual comparison only. The
    # controller does not read these waypoints when choosing actions.
    for trajectory in fleet.trajectories:
        points = [
            screen_position(point)
            for point in trajectory[:, :2]
        ]
        if len(points) >= 2:
            pygame.draw.lines(
                window, (195, 205, 211), False, points, 1
            )

    for index, ship_id in enumerate(fleet.vessel_ids):
        ship = state["ships"][ship_id]
        colour = ship_colour(index, len(fleet.vessel_ids))
        position = screen_position((ship["x"], ship["y"]))

        trail_points = [
            screen_position(point)
            for point in trails[ship_id]
        ]
        if len(trail_points) >= 2:
            pygame.draw.lines(
                window, colour, False, trail_points, 2
            )

        goal = screen_position(ship["goal"])
        pygame.draw.circle(window, colour, goal, 7, 2)

        if ship["status"] == "active":
            heading = ship["heading"]

            hull = [
                (16, 0),
                (5, 6),
                (-11, 6),
                (-11, -6),
                (5, -6),
            ]

            polygon = []
            for forward, sideways in hull:
                dx = (
                    forward * np.cos(heading)
                    - sideways * np.sin(heading)
                )
                dy = (
                    forward * np.sin(heading)
                    + sideways * np.cos(heading)
                )
                polygon.append((
                    int(position[0] + dx),
                    int(position[1] - dy),
                ))

            pygame.draw.polygon(window, colour, polygon)
            pygame.draw.polygon(
                window, (30, 40, 50), polygon, 1
            )
        else:
            pygame.draw.circle(window, colour, position, 5, 2)

        if ship_id == selected_ship:
            pygame.draw.circle(
                window, (235, 140, 20), position, 23, 3
            )
        # An outer ring keeps selection and warning visible separately.
        if ship_id in warning_ship_ids and blink_on:
            pygame.draw.circle(
                window, (220, 35, 35), position, 30, 4
            )
        window.blit(
            small_font.render(str(ship_id), True, colour),
            (position[0] + 15, position[1] + 8),
        )

    elapsed = float(state["timestamp"] - fleet.start_time)
    title = (
        f"Step {state['step']} | Time {elapsed:.0f}s | "
        f"Active {len(fleet.active_ship_ids)}/{FLEET_SIZE}"
    )
    window.blit(
        font.render(title, True, (25, 45, 65)),
        (20, 15),
    )
    window.blit(
        small_font.render(
            "Grey: BC journey reference   Colour: travelled path",
            True,
            (65, 80, 95),
        ),
        (20, 43),
    )
    # Current closest pair across all active ships.
    if live_pairs:
        closest_pair, closest_distance = min(
            live_pairs.items(),
            key=lambda item: item[1],
        )

        closest_text = (
            f"Closest now: {closest_pair[0]} / {closest_pair[1]} "
            f"= {closest_distance:.1f} m"
        )

        closest_colour = (
            (210, 30, 30)
            if closest_distance < WARNING_DISTANCE_M
            else (25, 65, 85)
        )
    else:
        closest_text = "Closest now: fewer than two active ships."
        closest_colour = (65, 80, 95)

    window.blit(
        small_font.render(closest_text, True, closest_colour),
        (20, 66),
    )

    warning_text = (
        f"Warning threshold: {WARNING_DISTANCE_M:.0f} m | "
        f"Pairs below threshold now: {len(warning_pairs)}"
    )

    if warning_pairs and blink_on:
        pygame.draw.rect(
            window,
            (255, 215, 215),
            (15, 86, MAP_WIDTH - 30, 25),
        )

    window.blit(
        small_font.render(
            warning_text,
            True,
            (190, 30, 30) if warning_pairs else (65, 80, 95),
        ),
        (20, 90),
    )

    # This is a historical measurement from the most recent step,
    # not a prediction of a future encounter.
    if recent_encounters:
        recent_pair, recent_result = min(
            recent_encounters.items(),
            key=lambda item: item[1]["distance_m"],
        )

        recent_text = (
            f"Last-step encounter: {recent_pair[0]} / "
            f"{recent_pair[1]} = "
            f"{recent_result['distance_m']:.1f} m "
            f"at {recent_result['elapsed_seconds']:.1f}s"
        )

        window.blit(
            small_font.render(
                recent_text, True, (180, 90, 10)
            ),
            (20, WINDOW_HEIGHT - 30),
        )
    # Operator panel.
    pygame.draw.rect(
        window,
        (245, 247, 250),
        (MAP_WIDTH, 0, WINDOW_WIDTH - MAP_WIDTH, WINDOW_HEIGHT),
    )

    panel_x = MAP_WIDTH + 15

    def text(value, y, colour=(30, 45, 60), small=False):
        selected_font = small_font if small else font
        window.blit(
            selected_font.render(value, True, colour),
            (panel_x, y),
        )

    text("MANUAL OPERATOR", 15)
    text(
        f"Selected: {selected_ship} | Duration: {command_duration}s",
        44,
    )

    operator_ship = current_operator_ship(fleet, controller)
    operator_label = (
        "available"
        if operator_ship is None
        else f"controlling ship {operator_ship}"
    )
    text(f"Operator: {operator_label}", 73)

    text("TAB / click row: select ship", 106, small=True)
    text("LEFT/A: left  RIGHT/D: right  S: stop  R: release", 129, small=True)
    text("UP / DOWN: duration    SPACE: pause    ESC: exit", 152, small=True)

    run_label = (
        "FINISHED" if fleet.finished
        else "PAUSED" if paused
        else "RUNNING"
    )
    text(run_label, 177, (170, 85, 20))

    row_rectangles = {}

    for index, ship_id in enumerate(fleet.vessel_ids):
        y = ROW_TOP + index * ROW_HEIGHT
        ship = state["ships"][ship_id]

        rectangle = pygame.Rect(
            MAP_WIDTH + 8,
            y,
            WINDOW_WIDTH - MAP_WIDTH - 16,
            ROW_HEIGHT - 2,
        )
        row_rectangles[ship_id] = rectangle

        if ship_id == selected_ship:
            pygame.draw.rect(
                window, (255, 231, 189), rectangle
            )

        mode = (
            controller.mode[ship_id]
            if ship["status"] == "active"
            else ship["status"]
        )

        if mode == "operator":
            remaining = controller.remaining_seconds(ship_id)
            command_name = controller.operator[ship_id]["direction"]

            if command_name == "stop":
                mode = f"holding ({remaining:.0f}s left)"
            else:
                mode = (
                    f"operator {command_name} "
                    f"({remaining:.0f}s left)"
                )

        neighbour = nearest.get(ship_id)

        if neighbour is not None:
            neighbour_text = (
                f"near {neighbour['ship_id']}: "
                f"{neighbour['distance_m']:.1f}m"
            )
        else:
            neighbour_text = (
                "no active neighbour"
                if ship["status"] == "active"
                else "retired"
            )

        row_colour = (
            (210, 30, 30)
            if ship_id in warning_ship_ids
            else (30, 45, 60)
        )

        text(
            f"{ship_id}: {mode} | {neighbour_text}",
            y + 2,
            colour=row_colour,
            small=True,
        )
        text(
            f"x={ship['x']:.1f}  y={ship['y']:.1f}  "
            f"speed={ship['speed']:.2f} m/s",
            y + 22,
            small=True,
        )

    # Two lines keep messages inside the panel.
    words = message.split()
    lines = [""]
    for word in words:
        candidate = (lines[-1] + " " + word).strip()
        if small_font.size(candidate)[0] > WINDOW_WIDTH - MAP_WIDTH - 30:
            lines.append(word)
        else:
            lines[-1] = candidate

    for index, line in enumerate(lines[:2]):
        text(line, 891 + index * 21, (140, 65, 15), small=True)

    pygame.display.flip()
    return row_rectangles


def main():
    policy = FrozenBCPolicy()
    trajectories, timestamps, vessel_ids = load_bc_scenario(
        policy,
        fleet_size=FLEET_SIZE,
        seed=SCENARIO_SEED,
        min_start_distance=MIN_START_DISTANCE_M,
    )

    fleet = FleetSimulator(
        trajectories=trajectories,
        timestamps=timestamps,
        vessel_ids=vessel_ids,
        region_of_interest=REGION_OF_INTEREST,
        max_steps=MAX_STEPS,
        enable_graphics=False,
        dt=DT,
    )

    try:
        state = fleet.reset(seed=SCENARIO_SEED)
        controller = BCController(fleet, policy=policy)
        previous_warning_pairs = report_distance_alerts(
            fleet,
            state,
            set(),
        )
        if not fleet.active_ship_ids:
            raise RuntimeError("The scenario has no active ships.")

        selected_ship = fleet.active_ship_ids[0]

        trails = {
            ship_id: [(ship["x"], ship["y"])]
            for ship_id, ship in state["ships"].items()
        }

        pygame.init()
        window = pygame.display.set_mode((
            WINDOW_WIDTH, WINDOW_HEIGHT
        ))
        pygame.display.set_caption(
            "ShipNaviSim - Manual Operator and BC Navigation"
        )

        font = pygame.font.Font(None, 26)
        small_font = pygame.font.Font(None, 21)
        clock = pygame.time.Clock()

        # Start paused so the user can inspect and select a vessel.
        paused = True
        running = True
        command_duration = INITIAL_COMMAND_DURATION
        message = "Select a ship, then press SPACE to begin."
        row_rectangles = {}
        accumulator = 0.0
        step_interval = 1.0 / STEPS_PER_REAL_SECOND
        finish_reported = False

        print(f"\nManual operator demo: {len(vessel_ids)} ships.")
        print("The simulation starts PAUSED.")
        print("TAB or click a panel row: select a ship.")
        print("LEFT/A or RIGHT/D: issue a timed turn. S: stop and hold.")
        print("R: release the selected ship early.")
        print("UP/DOWN: change command duration.")
        print("SPACE: pause/resume. ESC: exit.")
        print("There is no scheduled takeover in this demo.")

        while running:
            real_seconds = min(clock.tick(30) / 1000.0, 0.25)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                ):
                    for ship_id, rectangle in row_rectangles.items():
                        if rectangle.collidepoint(event.pos):
                            selected_ship = ship_id
                            message = f"Selected ship {ship_id}."
                            break

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        if not fleet.finished:
                            paused = not paused
                            accumulator = 0.0
                            message = (
                                "Paused. Commands use simulated time."
                                if paused
                                else "Simulation running."
                            )

                    elif event.key == pygame.K_TAB:
                        active = fleet.active_ship_ids
                        if active:
                            if selected_ship in active:
                                index = (
                                    active.index(selected_ship) + 1
                                ) % len(active)
                            else:
                                index = 0
                            selected_ship = active[index]
                            message = f"Selected ship {selected_ship}."

                    elif event.key == pygame.K_UP:
                        command_duration = min(
                            MAX_COMMAND_DURATION,
                            command_duration + DT,
                        )
                        message = (
                            f"Next command duration: {command_duration}s."
                        )

                    elif event.key == pygame.K_DOWN:
                        command_duration = max(
                            DT, command_duration - DT
                        )
                        message = (
                            f"Next command duration: {command_duration}s."
                        )

                    elif event.key in (
                        pygame.K_LEFT,
                        pygame.K_a,
                        pygame.K_RIGHT,
                        pygame.K_d,
                        pygame.K_s,
                    ):
                        if selected_ship not in fleet.active_ship_ids:
                            message = "Select an active ship first."
                            continue

                        operator_ship = current_operator_ship(
                            fleet, controller
                        )

                        if (
                            operator_ship is not None
                            and operator_ship != selected_ship
                        ):
                            message = (
                                f"Operator busy with {operator_ship}. "
                                "Select that ship and press R to release."
                            )
                            continue

                        if event.key == pygame.K_s:
                            direction = "stop"
                        elif event.key in (
                            pygame.K_LEFT, pygame.K_a
                        ):
                            direction = "left"
                        else:
                            direction = "right"

                        controller.take_over(
                            selected_ship,
                            direction=direction,
                            duration=command_duration,
                            turn_rate_deg=TURN_RATE_DEG,
                        )
                        message = (
                            f"Ship {selected_ship}: {direction} "
                            f"for {command_duration} simulated seconds."
                        )
                        print(message)

                    elif event.key == pygame.K_r:
                        if selected_ship not in fleet.active_ship_ids:
                            message = "Select an active ship first."
                        elif controller.mode[selected_ship] != "operator":
                            message = (
                                "The selected ship is not under "
                                "operator control."
                            )
                        else:
                            controller.release(selected_ship)
                            message = (
                                f"Released ship {selected_ship}; "
                                "BC navigation resumed."
                            )

            if not running:
                break

            if not paused and not fleet.finished:
                accumulator += real_seconds

                if accumulator >= step_interval:
                    # Advance at most once per display frame.
                    accumulator -= step_interval

                    active_before = fleet.active_ship_ids.copy()
                    modes_before = controller.mode.copy()
                    statuses_before = {
                        ship_id: ship["status"]
                        for ship_id, ship in state["ships"].items()
                    }

                    actions = controller.actions(state)
                    state = fleet.step(actions)
                    previous_warning_pairs = report_distance_alerts(
                        fleet,
                        state,
                        previous_warning_pairs,
                    )
                    for ship_id in active_before:
                        ship = state["ships"][ship_id]
                        trails[ship_id].append((
                            ship["x"], ship["y"]
                        ))

                        if ship["status"] != statuses_before[ship_id]:
                            print(
                                f"Ship {ship_id}: {ship['status']} "
                                f"at step {state['step']}."
                            )
                            if ship_id == selected_ship:
                                message = (
                                    f"Ship {ship_id}: {ship['status']}."
                                )

                        elif (
                            modes_before[ship_id] == "operator"
                            and controller.mode[ship_id] == "autonomous"
                        ):
                            message = (
                                f"Ship {ship_id}: command finished; "
                                "BC navigation resumed."
                            )
            else:
                accumulator = 0.0

            if fleet.finished and not finish_reported:
                finish_reported = True
                message = "Run finished. Close the window or press ESC."
                print("\nNo active vessels remain.")

            row_rectangles = draw_display(
                window,
                fleet,
                controller,
                state,
                trails,
                selected_ship,
                command_duration,
                paused,
                message,
                font,
                small_font,
            )

        print("\nFinal vessel status:")
        for ship_id, ship in state["ships"].items():
            print(f"  Ship {ship_id}: {ship['status']}")

        pair_minima = fleet.get_distance_metrics()["episode_minima"]
        print(f"\nMeasured {len(pair_minima)} distinct vessel pairs.")

        for pair, result in sorted(
            pair_minima.items(),
            key=lambda item: item[1]["distance_m"],
        )[:10]:
            print(
                f"Ships {pair[0]} and {pair[1]}: "
                f"{result['distance_m']:.2f} m "
                f"at {result['elapsed_seconds']:.1f}s"
            )

    finally:
        fleet.close()
        pygame.quit()


if __name__ == "__main__":
    main()
