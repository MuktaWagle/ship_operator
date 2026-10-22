"""Operator State and Budget Tracker demo.

Visual demonstration of Task 2 integrated with the full ShipNaviSim fleet.
Keeps ship visuals, polygon hulls, colors, and vessel speeds identical to demo_manual_operator.py.

Controls:
    TAB / Click Row : Select an active ship
    LEFT / A        : Turn selected ship left (via operator dispatch)
    RIGHT / D       : Turn selected ship right (via operator dispatch)
    S               : Stop and hold selected ship (via operator dispatch)
    R               : Early release from operator control
    G               : Toggle Greedy Auto-Intervention Dispatcher
    UP / DOWN       : Change intervention duration (tau) in 10-second steps
    SPACE           : Pause / resume simulation
    ESC             : Exit
"""

import sys
from pathlib import Path
import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fleet_simulator import FleetSimulator
from src.operator_tracker import OperatorBudgetTracker, OperatorStatus
from bc_navigation import BCController, FrozenBCPolicy, load_bc_scenario


FLEET_SIZE = 15
SCENARIO_SEED = 43
DT = 10
MAX_STEPS = 1000
WARNING_DISTANCE_M = 500.0
MIN_START_DISTANCE_M = 100.0

STEPS_PER_REAL_SECOND = 3

TURN_RATE_DEG = 1.0
INITIAL_COMMAND_DURATION = 60
MAX_COMMAND_DURATION = 240

# Operator & Budget settings
TOTAL_BUDGET = 300.0         # Total assistance budget B in simulated seconds
COMMUNICATION_DELAY = 10.0   # Buffer delay delta in simulated seconds

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


def report_distance_alerts(fleet, state, previous_warning_pairs):
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
    tracker,
    state,
    trails,
    selected_ship,
    command_duration,
    auto_greedy,
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

        if tracker.is_preparing and tracker.assigned_ship_id == ship_id:
            pygame.draw.circle(
                window, (240, 160, 20), position, 26, 2
            )

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

    # ------------------OPERATOR & BUDGET PANEL ------------------
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

    text("REMOTE OPERATOR & BUDGET", 12)

    # Operator status
    if tracker.status == OperatorStatus.AVAILABLE:
        status_label = "Available"
        status_col = (20, 135, 60)
    elif tracker.status == OperatorStatus.PREPARING:
        status_label = f"Preparing ship {tracker.assigned_ship_id} (Buffer δ: {tracker.preparation_timer:.0f}s)"
        status_col = (200, 110, 10)
    else:
        status_label = f"Assisting ship {tracker.assigned_ship_id} ({tracker.assistance_timer:.0f}s left)"
        status_col = (20, 90, 200)

    text(f"Operator: {status_label}", 36, colour=status_col)

    # Budget bar & info
    ratio = tracker.remaining_budget / tracker.total_budget if tracker.total_budget > 0 else 0
    bar_width = WINDOW_WIDTH - MAP_WIDTH - 30
    bar_rect = pygame.Rect(panel_x, 62, bar_width, 10)
    pygame.draw.rect(window, (215, 220, 225), bar_rect, border_radius=3)
    fill_w = int(bar_width * ratio)
    fill_col = (30, 150, 70) if ratio > 0.4 else (210, 140, 20) if ratio > 0.15 else (210, 40, 40)
    if fill_w > 0:
        pygame.draw.rect(window, fill_col, pygame.Rect(panel_x, 62, fill_w, 10), border_radius=3)
    pygame.draw.rect(window, (160, 170, 180), bar_rect, 1, border_radius=3)

    text(
        f"Budget (b_t): {tracker.remaining_budget:.0f}s / {tracker.total_budget:.0f}s ({ratio*100:.0f}%) | "
        f"Active: {tracker.total_active_assistance_time:.0f}s | Interventions: {tracker.total_interventions}",
        76,
        small=True,
    )

    greedy_str = "ON" if auto_greedy else "OFF"
    text(
        f"Selected: {selected_ship} | Duration (τ): {command_duration}s | Greedy Policy (G): {greedy_str}",
        96,
        small=True,
    )

    text("TAB / click row: select ship    UP/DOWN: duration", 118, small=True)
    text("LEFT/A: left  RIGHT/D: right  S: stop  R: release", 137, small=True)
    text("SPACE: pause    G: toggle auto-greedy dispatcher    ESC: exit", 156, small=True)

    run_label = (
        "FINISHED" if fleet.finished
        else "PAUSED" if paused
        else "RUNNING"
    )
    text(run_label, 178, (170, 85, 20))

    row_rectangles = {}
    action_mask = tracker.get_action_mask(fleet.active_ship_ids)

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
                mode = f"operator {command_name} ({remaining:.0f}s left)"
        elif tracker.is_preparing and tracker.assigned_ship_id == ship_id:
            mode = f"buffer delay ({tracker.preparation_timer:.0f}s left)"

        neighbour = nearest.get(ship_id)
        if neighbour is not None:
            neighbour_text = (
                f"near {neighbour['ship_id']}: {neighbour['distance_m']:.1f}m"
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

        act_idx = tracker.ship_id_to_action(ship_id)
        can_assign = action_mask[act_idx]
        mask_tag = "OK" if can_assign else "--"

        text(
            f"{ship_id}: {mode} | {neighbour_text} | [{mask_tag}]",
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

    # Status / log message
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
    print(f"\nOperator State & Budget Tracker demo: {FLEET_SIZE} ships.")
    print("Loading BC navigation scenario...")

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

    controller = BCController(fleet, policy=policy)
    tracker = OperatorBudgetTracker(
        vessel_ids=fleet.vessel_ids,
        total_budget=TOTAL_BUDGET,
        duration=INITIAL_COMMAND_DURATION,
        delay=COMMUNICATION_DELAY,
        dt=fleet.dt,
        max_simulation_time=MAX_STEPS * DT,
    )

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("ShipNaviSim — Operator State & Budget Tracker Demo")

    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Helvetica, Arial", 16)
    small_font = pygame.font.SysFont("Helvetica, Arial", 13)

    state = fleet.reset(seed=SCENARIO_SEED)
    controller.reset()
    tracker.reset()

    trails = {ship_id: [] for ship_id in fleet.vessel_ids}
    selected_ship = fleet.vessel_ids[0]
    command_duration = INITIAL_COMMAND_DURATION
    command_direction = "stop"
    auto_greedy = False
    paused = True
    running = True

    step_interval = 1.0 / STEPS_PER_REAL_SECOND
    accumulator = 0.0
    previous_warning_pairs = set()

    message = (
        "Paused. Select a ship and command LEFT/RIGHT/STOP, "
        "or press G for Greedy Auto Dispatcher."
    )

    try:
        while running:
            real_seconds = clock.tick(30) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked_pos = event.pos
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        message = "Paused." if paused else "Resumed simulation."
                    elif event.key == pygame.K_g:
                        auto_greedy = not auto_greedy
                        message = f"Greedy Policy: {'ENABLED' if auto_greedy else 'DISABLED'}."
                    elif event.key == pygame.K_TAB:
                        active = fleet.active_ship_ids
                        if active:
                            if selected_ship in active:
                                next_index = (
                                    active.index(selected_ship) + 1
                                ) % len(active)
                                selected_ship = active[next_index]
                            else:
                                selected_ship = active[0]
                            message = f"Selected ship {selected_ship}."
                    elif event.key == pygame.K_UP:
                        command_duration = min(
                            MAX_COMMAND_DURATION, command_duration + 10
                        )
                        tracker.duration = command_duration
                        message = f"Intervention duration: {command_duration}s."
                    elif event.key == pygame.K_DOWN:
                        command_duration = max(10, command_duration - 10)
                        tracker.duration = command_duration
                        message = f"Intervention duration: {command_duration}s."
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

                        direction = (
                            "stop" if event.key == pygame.K_s
                            else "left" if event.key in (pygame.K_LEFT, pygame.K_a)
                            else "right"
                        )
                        command_direction = direction

                        act = tracker.ship_id_to_action(selected_ship)
                        if not tracker.is_action_valid(act, fleet.active_ship_ids):
                            if not tracker.is_available:
                                message = (
                                    f"Operator busy with ship {tracker.assigned_ship_id} "
                                    f"({tracker.status.name})."
                                )
                            elif not tracker.has_sufficient_budget:
                                message = (
                                    f"Insufficient budget ({tracker.remaining_budget:.0f}s left). "
                                    f"Requires {tracker.duration:.0f}s."
                                )
                            else:
                                message = f"Cannot assign ship {selected_ship}."
                            continue

                        # Dispatch via OperatorBudgetTracker
                        tracker.dispatch(act, fleet.active_ship_ids)
                        message = (
                            f"Assigned ship {selected_ship} ({direction}). "
                            f"Entering {tracker.status.name} (buffer {tracker.delay:.0f}s)..."
                        )
                        print(f"DISPATCH ship {selected_ship}: {direction} (Status: {tracker.status.name})")

                    elif event.key == pygame.K_r:
                        if selected_ship not in fleet.active_ship_ids:
                            message = "Select an active ship first."
                        elif tracker.assigned_ship_id != selected_ship:
                            message = "The selected ship is not assigned to the operator."
                        else:
                            if controller.mode.get(selected_ship) == "operator":
                                controller.release(selected_ship)
                            tracker._release_operator()
                            message = f"Released ship {selected_ship}; operator is now Available."
                            print(f"EARLY RELEASE ship {selected_ship}.")

            if not running:
                break

            # Greedy auto-intervention policy: assigns assistance to vessel with closest separation
            if (
                auto_greedy
                and tracker.is_available
                and tracker.has_sufficient_budget
                and not paused
                and not fleet.finished
            ):
                pairs, nearest = calculate_live_distances(state)
                candidates = [
                    (nb["distance_m"], sid)
                    for sid, nb in nearest.items()
                    if nb["distance_m"] < WARNING_DISTANCE_M and sid in fleet.active_ship_ids
                ]
                if candidates:
                    candidates.sort()
                    closest_dist, target_ship = candidates[0]
                    act = tracker.ship_id_to_action(target_ship)
                    command_direction = "right"  # turn right to evade
                    tracker.dispatch(act, fleet.active_ship_ids)
                    message = (
                        f"[Greedy] Dispatched assistance to ship {target_ship} "
                        f"(separation: {closest_dist:.1f}m)."
                    )
                    print(message)

            if not paused and not fleet.finished:
                accumulator += real_seconds

                if accumulator >= step_interval:
                    accumulator -= step_interval

                    # When tracker reaches ASSISTING state, engage BC controller takeover
                    if tracker.is_assisting and tracker.assigned_ship_id is not None:
                        aid = tracker.assigned_ship_id
                        if controller.mode.get(aid) != "operator":
                            controller.take_over(
                                aid,
                                direction=command_direction,
                                duration=tracker.assistance_timer,
                                turn_rate_deg=TURN_RATE_DEG,
                            )
                            message = f"Operator controlling ship {aid}: turning {command_direction}."

                    actions = controller.actions(state)
                    state = fleet.step(actions)

                    tracker_state = tracker.step(active_ship_ids=fleet.active_ship_ids)
                    if tracker_state.get("transition_event"):
                        ev = tracker_state["transition_event"]
                        if "EXPIRED" in ev or "BUDGET" in ev:
                            message = f"Assistance finished: {ev}."

                    for ship_id in fleet.vessel_ids:
                        ship = state["ships"][ship_id]
                        trails[ship_id].append((ship["x"], ship["y"]))
                        if len(trails[ship_id]) > 100:
                            trails[ship_id].pop(0)

                    previous_warning_pairs = report_distance_alerts(
                        fleet, state, previous_warning_pairs
                    )

            row_rectangles = draw_display(
                window,
                fleet,
                controller,
                tracker,
                state,
                trails,
                selected_ship,
                command_duration,
                auto_greedy,
                paused,
                message,
                font,
                small_font,
            )

            if pygame.mouse.get_pressed()[0]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for ship_id, rectangle in row_rectangles.items():
                    if rectangle.collidepoint(mouse_x, mouse_y):
                        if ship_id != selected_ship:
                            selected_ship = ship_id
                            message = f"Selected ship {selected_ship}."

    finally:
        fleet.close()
        pygame.quit()
        print("Demo ended.")


if __name__ == "__main__":
    main()
