"""Visual allocation demo using the same BC traffic as the operator demo."""

import sys
from pathlib import Path

import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bc_navigation import BCController, FrozenBCPolicy, load_bc_scenario
from src.allocation_env import FleetAllocationEnv, RewardWeights
from src.fleet_simulator import FleetSimulator
from src.operator_tracker import OperatorBudgetTracker


# Match demo/demo_operator_budget.py.
FLEET_SIZE = 15
SCENARIO_SEED = 43
DT = 10
MAX_STEPS = 1000
WARNING_DISTANCE_M = 500.0
MIN_START_DISTANCE_M = 100.0
STEPS_PER_REAL_SECOND = 3

TURN_RATE_DEG = 1.0
INITIAL_COMMAND_DURATION = 60
TOTAL_BUDGET = 300.0
COMMUNICATION_DELAY = 10.0
COLLISION_DISTANCE_M = 50.0

REGION = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
WINDOW_SIZE = (1200, 760)
MAP_WIDTH = 850


def build_environment():
    policy = FrozenBCPolicy()
    trajectories, timestamps, vessel_ids = load_bc_scenario(
        policy,
        fleet_size=FLEET_SIZE,
        seed=SCENARIO_SEED,
        min_start_distance=MIN_START_DISTANCE_M,
        split="train",
    )

    fleet = FleetSimulator(
        trajectories,
        timestamps,
        vessel_ids,
        region_of_interest=REGION,
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
        dt=DT,
        max_simulation_time=MAX_STEPS * DT,
    )
    return FleetAllocationEnv(
        fleet,
        controller,
        tracker,
        warning_distance_m=WARNING_DISTANCE_M,
        collision_distance_m=COLLISION_DISTANCE_M,
        reward_weights=RewardWeights(
            completed_journey=1.0,
            collision=10.0,
            near_miss=1.0,
            intervention=1.0,
        ),
        assistance_direction="right",
        assistance_turn_rate_deg=TURN_RATE_DEG,
        speed_scale_mps=10.0,
    )


def encounter_policy(observation, action_mask, map_diagonal):
    """Assign the operator to the closest ship inside the warning distance."""
    warning_ratio = WARNING_DISTANCE_M / map_diagonal
    candidates = [
        (float(observation["nearest_distance"][index]), index + 1)
        for index, ship in enumerate(observation["ships"])
        if ship[6] > 0.5
        and observation["nearest_distance"][index] < warning_ratio
        and action_mask[index + 1]
    ]
    return min(candidates)[1] if candidates else 0


def ship_colour(index, count):
    colour = pygame.Color(0)
    colour.hsva = (index * 360.0 / max(1, count), 75, 75, 100)
    return colour.r, colour.g, colour.b


def draw(
    window,
    environment,
    state,
    trails,
    action,
    reward,
    total_reward,
    collisions,
    near_misses,
    paused,
    font,
    small_font,
):
    window.fill((232, 244, 250))
    reference = next(iter(environment.fleet.environments.values()))
    margin = 45
    scale = min(
        (MAP_WIDTH - 2 * margin) / reference.max_x,
        (WINDOW_SIZE[1] - 100) / reference.max_y,
    )

    def screen(position):
        return (
            int(margin + position[0] * scale),
            int(WINDOW_SIZE[1] - 45 - position[1] * scale),
        )

    pygame.draw.rect(
        window,
        (110, 140, 160),
        (
            margin,
            45,
            int(reference.max_x * scale),
            int(reference.max_y * scale),
        ),
        2,
    )

    # Reference journeys are visual context only; BCController does not use
    # these waypoints when choosing its next action.
    for trajectory in environment.fleet.trajectories:
        pygame.draw.lines(
            window,
            (195, 205, 211),
            False,
            [screen(point) for point in trajectory[:, :2]],
            1,
        )

    for index, ship_id in enumerate(environment.fleet.vessel_ids):
        ship = state["ships"][ship_id]
        colour = ship_colour(index, len(environment.fleet.vessel_ids))
        position = screen((ship["x"], ship["y"]))

        trail = [screen(point) for point in trails[ship_id]]
        if len(trail) >= 2:
            pygame.draw.lines(window, colour, False, trail, 2)

        pygame.draw.circle(window, colour, screen(ship["goal"]), 6, 2)
        if ship["status"] == "active":
            pygame.draw.circle(window, colour, position, 7)
        else:
            pygame.draw.circle(window, colour, position, 7, 2)

        if environment.controller.mode[ship_id] == "operator":
            pygame.draw.circle(window, (240, 155, 20), position, 16, 3)

        window.blit(
            small_font.render(str(ship_id), True, colour),
            (position[0] + 10, position[1] - 10),
        )

    panel_x = MAP_WIDTH + 25
    operator = environment.tracker
    lines = [
        "ALLOCATION DEMO",
        f"Step: {state['step']} | Time: {state['timestamp']:.0f}s",
        f"Action: {action} | Reward: {reward:+.1f}",
        f"Total reward: {total_reward:+.1f}",
        f"Collisions: {collisions}",
        f"Near misses: {near_misses}",
        f"Budget: {operator.remaining_budget:.0f}s / {operator.total_budget:.0f}s",
        f"Operator: {operator.status.name}",
        f"Assigned: {operator.assigned_ship_id}",
        f"State: {'PAUSED' if paused else 'RUNNING'}",
        "",
        "Allocation: closest ship under 500m",
        "SPACE: pause/resume    ESC: exit",
    ]
    for index, line in enumerate(lines):
        selected_font = font if index == 0 else small_font
        window.blit(
            selected_font.render(line, True, (30, 50, 70)),
            (panel_x, 35 + index * 32),
        )
    pygame.display.flip()


def main():
    environment = build_environment()
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Ship Operator - BC Allocation Demo")
    font = pygame.font.Font(None, 30)
    small_font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    observation, info = environment.reset(seed=SCENARIO_SEED)
    state = environment.state
    trails = {
        ship_id: [(ship["x"], ship["y"])]
        for ship_id, ship in state["ships"].items()
    }
    total_reward = 0.0
    collisions = 0
    near_misses = 0
    action = 0
    reward = 0.0
    paused = False
    running = True
    accumulator = 0.0
    interval = 1.0 / STEPS_PER_REAL_SECOND

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    try:
        with open(log_dir / "allocation_demo.txt", "w", encoding="utf-8") as log_file:
            while running:
                elapsed = clock.tick(30) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            paused = not paused

                if not paused and not environment.fleet.finished:
                    accumulator += elapsed
                    if accumulator >= interval:
                        accumulator -= interval
                        action = encounter_policy(
                            observation,
                            info["action_mask"],
                            environment.map_diagonal,
                        )
                        observation, reward, terminated, truncated, info = (
                            environment.step(action)
                        )
                        state = environment.state
                        total_reward += reward
                        components = info["reward_components"]
                        collisions += components["collision_events"]
                        near_misses += components["near_miss_events"]

                        for ship_id, ship in state["ships"].items():
                            trails[ship_id].append((ship["x"], ship["y"]))

                        log_line = (
                             f"step={state['step']:03d} action={action} "
                             f"reward={reward:+.1f} "
                             f"collisions={collisions} near_misses={near_misses}"
                             )
                        
                        print(log_line)
                        log_file.write(log_line + "\n")
                        
                        if terminated or truncated:
                            paused = True

                draw(
                    window,
                    environment,
                    state,
                    trails,
                    action,
                    reward,
                    total_reward,
                    collisions,
                    near_misses,
                    paused,
                    font,
                    small_font,
                )
    finally:
        environment.close()
        pygame.quit()


if __name__ == "__main__":
    main()
