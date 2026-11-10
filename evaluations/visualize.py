"""
Controls
    1-9            switch to that policy and restart the current scenario
    LEFT / RIGHT   previous / next policy (restarts the scenario)
    UP / DOWN      faster / slower simulation
    SPACE          pause / resume
    ESC            quit

On screen
    red blinking rings   ships closer than the warning distance (500 m)
    orange ring          ship currently under operator assistance
    yellow ring          operator is preparing to assist this ship (delay phase)

Commands to run:
    python -m evaluations.visualize
    python -m evaluations.visualize --dqn runs/dqn_s0/best.pt
    python -m evaluations.visualize --ppo runs/ppo_s0/best.pt
    python -m evaluations.visualize --dqn runs/dqn_s0/best.pt --ppo runs/ppo_s0/best.pt
    python -m evaluations.visualize --dqn runs/dqn_s0/best.pt --ppo runs/ppo_s0/best.pt --mcts --autoplay
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.mcts import ModelMCTSConfig, ModelMCTSPolicy 
from evaluations.policies import HighestRiskGreedyPolicy  
from evaluations.ppo import PPOPolicy  
from evaluations.run_benchmark import base_env_config, load_checkpoints  
from training.baselines import make_baselines 
from training.evaluation import DQNPolicy  
from training.features import ObservationEncoder  
from training.scenarios import EnvConfig, build_unique_envs  

WINDOW_SIZE = (1320, 800)
MAP_WIDTH = 880
SPEEDS = [0.5, 1, 2, 3, 5, 10, 20, 50]
BACKGROUND = (232, 244, 250)
TEXT = (30, 50, 70)

POLICY_DISPLAY_NAMES = {
    "idle": "Idle",
    "greedy_highest_risk": "Greedy Highest Risk",
    "mcts": "MCTS",
}

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dqn", nargs="*", default=[], help="DQN checkpoints (.pt)")
    p.add_argument("--ppo", nargs="*", default=[], help="PPO checkpoints (.pt)")
    p.add_argument("--mcts", action="store_true", help="include the MCTS planner (each decision can take a while)")
    # argument used for RiskThresholdPolicy calibration
    # p.add_argument("--gate", nargs=2, type=float, metavar=("THRESHOLD_M", "HORIZON_S"), default=None, help="add a risk gate with these values, e.g. the best row of calibration.csv")
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--seed-offset", type=int, default=20_000)
    p.add_argument("--split", choices=("train", "val", "test"), default=None)
    p.add_argument("--speed", type=float, default=3.0, help="simulation steps per real second")
    p.add_argument("--start-policy", default=None, help="policy name to start with (default: first)")
    p.add_argument("--autoplay", action="store_true", help="start with auto-cycle on")
    p.add_argument("--seed", type=int, default=0, help="seed for the random policy")
    # used only when no checkpoint supplies an environment config
    p.add_argument("--fleet-size", type=int, default=10)
    p.add_argument("--min-start-distance", type=float, default=800.0)
    p.add_argument("--maneuver", choices=("right", "left", "stop"), default="right")
    p.add_argument("--scenario-source", choices=("real", "synthetic", "mixed"), default=None,
                   help="where scenarios come from; default: the checkpoint's own setting (real if none). "
                        "Use real on a synthetic-trained checkpoint for the realistic test set.")
    p.add_argument("--synthetic-spec", default=None, metavar="S:O[,S:O...]",
                   help="synthetic difficulty as SPREAD_S:OFFSET_M pairs; default: the checkpoint's own")
    p.add_argument("--controller", choices=("bc", "rule", "vo"), default=None,
                   help="what an assisted vessel does: bc = fixed --maneuver, rule / vo = Task 5 planning "
                        "operator. Default: the checkpoint's own setting (bc if there is none). "
                        "Given explicitly, it overrides the checkpoint's setting for every policy.")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--budget", type=float, default=300.0)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--delay", type=float, default=0.0)
    # MCTS
    p.add_argument("--mcts-rollouts", type=int, default=60)
    p.add_argument("--mcts-horizon", type=int, default=40)
    p.add_argument("--mcts-depth", type=int, default=12)
    p.add_argument("--mcts-top-k", type=int, default=3)
    p.add_argument("--mcts-activation", type=float, default=1000.0)
    return p.parse_args(argv)


def build_policies(args, loaded, eval_envs, eval_config):
    baselines = make_baselines(seed=args.seed)
    policies = {
        "idle": baselines["idle"],
        # "random_valid": baselines["random_valid"],
        # "greedy_nearest": baselines["greedy_nearest"],
        "greedy_highest_risk": HighestRiskGreedyPolicy(),
        # "risk_threshold": baselines["risk_threshold"],
    }
    # if args.gate is not None:
        # policies["risk_gate_tuned"] = RiskThresholdPolicy(threshold_m=args.gate[0], horizon_s=args.gate[1])
    for label, group, agent, extra in loaded:
        encoder = ObservationEncoder.for_env(eval_envs[0], **extra["encoder"])
        wrapper = DQNPolicy if group == "dqn" else PPOPolicy
        policies[label] = wrapper(agent, encoder)
    if args.mcts:
        config = ModelMCTSConfig(
            rollouts=args.mcts_rollouts, horizon=args.mcts_horizon, tree_depth=args.mcts_depth,
            top_k=args.mcts_top_k, activation_m=args.mcts_activation,
        )
        policies["mcts"] = ModelMCTSPolicy(config)
    return policies

def display_policy_name(name):
    if name in POLICY_DISPLAY_NAMES:
        return POLICY_DISPLAY_NAMES[name]

    # Checkpoint policies, e.g. dqn:dqn_s0/best
    if name.startswith("dqn:"):
        return "DQN"

    if name.startswith("ppo:"):
        return "PPO"

    return name

def ship_colour(index, count):
    colour = pygame.Color(0)
    colour.hsva = (index * 360.0 / max(1, count), 75, 75, 100)
    return colour.r, colour.g, colour.b


class Viewer:
    def __init__(self, envs, policies, start_policy=0, speed=3.0, autoplay=False):
        self.envs = envs
        self.policy_names = list(policies)
        self.policies = policies
        self.scenario_index = 0
        self.policy_index = start_policy
        self.speed_index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))
        self.auto = autoplay
        self.paused = False
        self.results = {}
        self.finished_at = None
        self.busy = False

        pygame.init()
        self.window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Ship Operator - policy viewer")
        self.font = pygame.font.Font(None, 30)
        self.small = pygame.font.Font(None, 22)
        self.clock = pygame.time.Clock()
        self.restart()

    # ------------------------------------------------------------ properties
    @property
    def env(self):
        return self.envs[self.scenario_index]

    @property
    def policy_name(self):
        return self.policy_names[self.policy_index]

    @property
    def steps_per_second(self):
        return SPEEDS[self.speed_index]

    # --------------------------------------------------------------- control
    def restart(self):
        env = self.env
        self.observation, self.info = env.reset(seed=env.scenario_seed)
        self.state = env.state
        self.trails = {
            ship_id: [(ship["x"], ship["y"])] for ship_id, ship in self.state["ships"].items()
        }
        self.action, self.reward = 0, 0.0
        self.total_reward = 0.0
        self.collisions = self.near_misses = self.completed = self.interventions = 0
        self.done = False
        self.finished_at = None
        self.accumulator = 0.0

    def set_policy(self, index):
        self.policy_index = index % len(self.policy_names)
        self.restart()

    def set_scenario(self, index):
        self.scenario_index = index % len(self.envs)
        self.restart()

    def step(self):
        env = self.env
        policy = self.policies[self.policy_name]
        if self.policy_name == "mcts":
            self.busy = True
            self.draw()
            pygame.event.pump()
        self.action = int(policy(self.observation, self.info, env))
        self.busy = False
        self.observation, self.reward, terminated, truncated, self.info = env.step(self.action)
        self.state = env.state
        self.total_reward += self.reward
        components = self.info["reward_components"]
        self.collisions += components["collision_events"]
        self.near_misses += components["near_miss_events"]
        self.completed += components["completed_journeys"]
        self.interventions += components["interventions"]
        for ship_id, ship in self.state["ships"].items():
            self.trails[ship_id].append((ship["x"], ship["y"]))
        if terminated or truncated:
            self.done = True
            self.finished_at = time.time()
            self.results[(self.scenario_index, self.policy_name)] = {
                "reward": self.total_reward,
                "collisions": self.collisions,
                "near_misses": self.near_misses,
                "completed": self.completed,
                "interventions": self.interventions,
            }
            print(
                f"[scenario {self.scenario_index + 1}/{len(self.envs)} seed {env.scenario_seed}] "
                f"{display_policy_name(self.policy_name):<30} reward {self.total_reward:+7.2f} | collisions {self.collisions} | "
                f"near-misses {self.near_misses} | completed {self.completed} | interventions {self.interventions}"
            )

    # ------------------------------------------------------------------ main
    def run(self):
        running = True
        try:
            while running:
                elapsed = self.clock.tick(30) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        running = self.on_key(event.key)

                if self.auto and self.done and self.finished_at is not None:
                    if time.time() - self.finished_at > 1.5:
                        if self.policy_index < len(self.policy_names) - 1:
                            self.set_policy(self.policy_index + 1)
                        else:
                            self.auto = False

                if not self.paused and not self.done:
                    self.accumulator += elapsed
                    interval = 1.0 / self.steps_per_second
                    # Draw between steps so slow policies (MCTS) never block the window.
                    if self.accumulator >= interval:
                        self.accumulator -= interval
                        self.accumulator = min(self.accumulator, interval)
                        self.step()
                self.draw()
        finally:
            pygame.quit()
            for env in self.envs:
                env.close()

    def on_key(self, key):
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_UP:
            self.speed_index = min(len(SPEEDS) - 1, self.speed_index + 1)
        elif key == pygame.K_DOWN:
            self.speed_index = max(0, self.speed_index - 1)
        elif key == pygame.K_RIGHT:
            self.set_policy(self.policy_index + 1)
        elif key == pygame.K_LEFT:
            self.set_policy(self.policy_index - 1)
        elif key == pygame.K_n:
            self.set_scenario(self.scenario_index + 1)
        elif key == pygame.K_p:
            self.set_scenario(self.scenario_index - 1)
        elif key == pygame.K_r:
            self.restart()
        elif key == pygame.K_a:
            self.auto = not self.auto
        elif pygame.K_1 <= key <= pygame.K_9:
            index = key - pygame.K_1
            if index < len(self.policy_names):
                self.set_policy(index)
        return True

    # --------------------------------------------------------------- drawing
    def draw(self):
        window, env, state = self.window, self.env, self.state
        window.fill(BACKGROUND)
        reference = next(iter(env.fleet.environments.values()))
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
            window, (110, 140, 160),
            (margin, 45, int(reference.max_x * scale), int(reference.max_y * scale)), 2,
        )
        for trajectory in env.fleet.trajectories:
            pygame.draw.lines(window, (195, 205, 211), False, [screen(pt) for pt in trajectory[:, :2]], 1)

        warning_m = getattr(env, "warning_distance_m", 500.0)
        collision_m = getattr(env, "collision_distance_m", 50.0)
        active = [(sid, ship) for sid, ship in state["ships"].items() if ship["status"] == "active"]
        warned, collided = set(), set()
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                distance = math.hypot(a[1]["x"] - b[1]["x"], a[1]["y"] - b[1]["y"])
                if distance < collision_m:
                    collided.update((a[0], b[0]))
                if distance < warning_m:
                    warned.update((a[0], b[0]))
        blink_on = (pygame.time.get_ticks() // 500) % 2 == 0

        tracker = env.tracker
        preparing = tracker.status.name == "PREPARING"
        count = len(env.fleet.vessel_ids)
        for index, ship_id in enumerate(env.fleet.vessel_ids):
            ship = state["ships"][ship_id]
            colour = ship_colour(index, count)
            position = screen((ship["x"], ship["y"]))

            trail = [screen(pt) for pt in self.trails[ship_id]]
            if len(trail) >= 2:
                pygame.draw.lines(window, colour, False, trail, 2)
            pygame.draw.circle(window, colour, screen(ship["goal"]), 6, 2)
            pygame.draw.circle(window, colour, position, 7, 0 if ship["status"] == "active" else 2)

            if ship_id in warned and blink_on:
                pygame.draw.circle(window, (210, 40, 40), position, 20, 3)
            if ship_id in collided:
                pygame.draw.circle(window, (120, 0, 0), position, 24, 4)
            if env.controller.mode[ship_id] == "operator":
                pygame.draw.circle(window, (240, 155, 20), position, 15, 3)
            elif preparing and tracker.assigned_ship_id == ship_id:
                pygame.draw.circle(window, (235, 210, 40), position, 15, 2)

            window.blit(self.small.render(str(ship_id), True, colour), (position[0] + 10, position[1] - 10))

        self.draw_panel()
        pygame.display.flip()

    def draw_panel(self):
        window, env, state, tracker = self.window, self.env, self.state, self.env.tracker
        x = MAP_WIDTH + 25
        y = 25

        def line(text, font=None, colour=TEXT, gap=26):
            nonlocal y
            window.blit((font or self.small).render(text, True, colour), (x, y))
            y += gap

        line("POLICY VIEWER", self.font, gap=34)
        line(f"Policy: {display_policy_name(self.policy_name)}", self.font, (20, 90, 160), gap=34)
        line(f"Scenario {self.scenario_index + 1}/{len(self.envs)} (seed {env.scenario_seed})")
        line(f"Step {state['step']} | Time {state['timestamp']:.0f}s | Speed {self.steps_per_second:g} steps/s")
        if self.busy:
            status, colour = "THINKING...", (200, 100, 0)
        elif self.done:
            status, colour = "FINISHED", (30, 120, 50)
        elif self.paused:
            status, colour = "PAUSED", (150, 90, 0)
        else:
            status, colour = "RUNNING", TEXT
        line(f"State: {status}   Auto-cycle: {'ON' if self.auto else 'off'}", colour=colour, gap=30)

        line(f"Last action: {self.action}    Reward: {self.reward:+.1f}")
        line(f"Total reward: {self.total_reward:+.1f}", self.font, gap=32)
        line(f"Collisions: {self.collisions}    Near-misses: {self.near_misses}")
        line(f"Completed: {self.completed}    Interventions: {self.interventions}")
        line(f"Budget: {tracker.remaining_budget:.0f}s / {tracker.total_budget:.0f}s")
        line(f"Operator: {tracker.status.name}    Assigned: {tracker.assigned_ship_id}", gap=32)

        line("Policies (press number)", self.font, gap=30)
        for index, name in enumerate(self.policy_names):
            result = self.results.get((self.scenario_index, name))
            marker = ">" if index == self.policy_index else " "
            shown = display_policy_name(name)
            shown = shown if len(shown) <= 26 else shown[:25] + "~"
            text = f"{marker}{index + 1 if index < 9 else '-'} {shown}"
            if result:
                text += f"  R{result['reward']:+.1f} C{result['collisions']} N{result['near_misses']}"
            colour = (20, 90, 160) if index == self.policy_index else TEXT
            line(text, colour=colour, gap=22)

        y = WINDOW_SIZE[1] - 150
        for text in (
            "R restart   N/P scenario   </> policy",
            "UP/DOWN speed   A auto-cycle   SPACE pause",
            "Red ring: < warning distance",
            "Orange ring: assisted   Yellow: preparing",
            "R = reward, C = collisions, N = near-misses",
            "ESC quit",
        ):
            line(text, colour=(80, 100, 120), gap=22)


def main(argv=None):
    args = parse_args(argv)
    loaded = load_checkpoints(args)
    config = base_env_config(args, loaded)
    split = args.split or ("test" if config.fleet_size <= 4 else "train")
    eval_config = EnvConfig(**{**config.to_dict(), "split": split})

    exclude = []
    if split == "train":
        for _, _, _, extra in loaded:
            exclude += extra.get("train_ship_sets", []) + extra.get("eval_ship_sets", [])

    print(f"Building {args.scenarios} scenarios (fleet={config.fleet_size}, split={split})...")
    envs = build_unique_envs(eval_config, args.scenarios, args.seed_offset, exclude_sets=exclude)
    policies = build_policies(args, loaded, envs, eval_config)

    start = 0
    if args.start_policy is not None:
        names = list(policies)
        if args.start_policy not in names:
            raise SystemExit(f"Unknown policy '{args.start_policy}'. Available: {', '.join(names)}")
        start = names.index(args.start_policy)

    print(
    "Policies:",
    ", ".join(
        f"{i + 1}={display_policy_name(n)}"
        for i, n in enumerate(policies)
    ),
   )
    Viewer(envs, policies, start_policy=start, speed=args.speed, autoplay=args.autoplay).run()


if __name__ == "__main__":
    main()
