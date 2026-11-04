"""Operator planners: what the remote operator does while it controls one vessel.

A planner sees the live state of the assisted vessel and of every other active
vessel and returns a `(turn, speed)` pair for the next simulation step. It does
not know about budgets, delays or timers; the tracker and `BCController` own
those. Conventions match the teammates' simulator:

- angles in radians, 0 = east, counter-clockwise positive (so a right / starboard
  turn is NEGATIVE), positions in metres;
- `turn` is the heading change applied this step; the vessel then moves along its
  NEW heading for `dt` seconds at `speed` (see `BCController._movement`);
- ship dicts use the simulator's keys: x, y, heading, speed, goal.

Other vessels are predicted at constant velocity. That is a good assumption here
because the autonomous BC vessels cannot see each other and just head for their
goals, so nobody else is going to make way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """Physical limits the planner must respect (taken from the BC manifest)."""

    dt: float = 10.0  # seconds per simulation step
    max_turn: float = math.radians(30.0)  # largest allowed heading change per step
    max_speed: float = 10.0  # m/s


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def closest_in_step(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1) -> float:
    """Minimum separation during one step, assuming straight motion for both.

    Same geometry as `closest_distance_during_step` in src/fleet_simulator.py, which
    is what the reward uses to count near misses and collisions.
    """
    rx, ry = ax0 - bx0, ay0 - by0
    cx, cy = (ax1 - ax0) - (bx1 - bx0), (ay1 - ay0) - (by1 - by0)
    denominator = cx * cx + cy * cy
    if denominator <= 1e-12:
        fraction = 0.0
    else:
        fraction = clamp(-(rx * cx + ry * cy) / denominator, 0.0, 1.0)
    return math.hypot(rx + fraction * cx, ry + fraction * cy)


def goal_bearing(ship: dict) -> float:
    goal = ship["goal"]
    return math.atan2(goal[1] - ship["y"], goal[0] - ship["x"])


class Planner:
    """Interface: `plan` returns (turn, speed) for the assisted ship."""

    name = "planner"

    def reset(self) -> None:
        """Forget per-episode memory (called when the environment resets)."""

    def plan(self, ship_id, ship: dict, others: list[dict], limits: Limits, steps_left=None):
        """`steps_left` is how many steps (including this one) the operator still
        controls the vessel before it is handed back to BC navigation."""
        raise NotImplementedError


# --------------------------------------------------------------------------
class RuleBasedPlanner(Planner):
    """Fallback operator: if a neighbour is predicted to come within `safe_dist`
    while closing, turn to starboard and slow down for `hold_steps` steps, then
    head for the goal again. Otherwise steer straight for the goal.
    """

    name = "rule_based"

    def __init__(
        self,
        safe_dist: float = 700.0,
        horizon_steps: int = 36,
        starboard_deg: float = 40.0,
        slow_factor: float = 0.7,
        hold_steps: int = 6,
    ):
        self.safe_dist = safe_dist
        self.horizon_steps = horizon_steps
        self.starboard = math.radians(starboard_deg)
        self.slow_factor = slow_factor
        self.hold_steps = hold_steps
        self._hold: dict = {}  # ship_id -> (steps left, target heading)
        self._cruise: dict = {}  # ship_id -> speed when the operator took over

    def reset(self) -> None:
        self._hold.clear()
        self._cruise.clear()

    def _threatened(self, ship, others, limits) -> bool:
        horizon = self.horizon_steps * limits.dt
        vx = ship["speed"] * math.cos(ship["heading"])
        vy = ship["speed"] * math.sin(ship["heading"])
        for other in others:
            ovx = other["speed"] * math.cos(other["heading"])
            ovy = other["speed"] * math.sin(other["heading"])
            px, py = other["x"] - ship["x"], other["y"] - ship["y"]
            rvx, rvy = ovx - vx, ovy - vy
            speed_sq = rvx * rvx + rvy * rvy
            if px * rvx + py * rvy >= 0.0:  # not closing
                continue
            t = 0.0 if speed_sq < 1e-9 else clamp(-(px * rvx + py * rvy) / speed_sq, 0.0, horizon)
            if math.hypot(px + rvx * t, py + rvy * t) < self.safe_dist:
                return True
        return False

    def plan(self, ship_id, ship, others, limits, steps_left=None):
        cruise = self._cruise.setdefault(ship_id, max(ship["speed"], 1.0))
        hold_left, target = self._hold.get(ship_id, (0, None))
        if hold_left <= 0 and self._threatened(ship, others, limits):
            hold_left, target = self.hold_steps, wrap(ship["heading"] - self.starboard)
        if hold_left > 0:
            self._hold[ship_id] = (hold_left - 1, target)
            speed = cruise * self.slow_factor
        else:
            self._hold.pop(ship_id, None)
            target, speed = goal_bearing(ship), cruise
        turn = clamp(wrap(target - ship["heading"]), -limits.max_turn, limits.max_turn)
        return turn, clamp(speed, 0.0, limits.max_speed)


# --------------------------------------------------------------------------
class VelocityObstaclePlanner(Planner):
    """Sampling planner with a turn-rate-aware look-ahead.

    Each step it builds candidates (target heading, speed). A candidate is rolled
    forward `horizon_steps` steps: the vessel turns toward the target heading at
    most `turn_limit` per step, then holds it, while every other vessel keeps its
    current velocity. The smallest separation along the rollout is the candidate's
    clearance. Candidates with clearance >= `safe_dist` are admissible; the cheapest
    one wins. Cost = deviation from the goal bearing + speed loss + change of mind
    + a small penalty for passing to port (starboard is preferred, COLREG-style).
    If nothing is admissible, the candidate with the largest clearance is used.

    Hand-back awareness: the operator only controls the vessel for `steps_left` more
    steps. After that, BC navigation steers the vessel straight back to its goal, which
    can lead into the same conflict again. So after `steps_left` the rollout switches to
    "turn toward the goal at `handback_turn_deg` per step, at cruise speed" (a proxy for
    BC), and clearance is measured over the whole rollout including that part.

    With no threat the goal-bearing candidate costs zero and is admissible, so the
    operator does nothing unusual and the intervention is harmless.
    """

    name = "velocity_obstacle"

    def __init__(
        self,
        safe_dist: float = 700.0,
        horizon_steps: int = 36,
        turn_limit_deg: float = 20.0,
        max_deviation_deg: float = 110.0,
        n_headings: int = 23,
        speed_factors: tuple = (1.0, 0.85, 0.7),
        min_speed_fraction: float = 0.4,
        w_goal: float = 1.0,
        w_speed: float = 0.4,
        w_switch: float = 0.3,
        w_port: float = 0.4,
        handback_turn_deg: float = 12.0,
    ):
        self.safe_dist = safe_dist
        self.horizon_steps = horizon_steps
        self.handback_turn = math.radians(handback_turn_deg)
        self.turn_limit = math.radians(turn_limit_deg)
        self.offsets = [
            math.radians(max_deviation_deg) * (2.0 * k / (n_headings - 1) - 1.0)
            for k in range(n_headings)
        ]
        self.speed_factors = speed_factors
        self.min_speed_fraction = min_speed_fraction
        self.w_goal, self.w_speed, self.w_switch, self.w_port = w_goal, w_speed, w_switch, w_port
        self._cruise: dict = {}
        self._previous_target: dict = {}

    def reset(self) -> None:
        self._cruise.clear()
        self._previous_target.clear()

    # -- look-ahead ---------------------------------------------------------
    def _clearance(self, ship, target, speed, others, limits, steps_left, cruise) -> float:
        dt = limits.dt
        operator_turn = min(self.turn_limit, limits.max_turn)
        bc_turn = min(self.handback_turn, limits.max_turn)
        goal_x, goal_y = ship["goal"]
        x, y, heading = ship["x"], ship["y"], ship["heading"]
        best = math.inf
        for k in range(self.horizon_steps):
            if k < steps_left:  # operator in control
                aim, limit, step_speed = target, operator_turn, speed
            else:  # handed back: BC heads for the goal at cruise speed
                aim, limit, step_speed = math.atan2(goal_y - y, goal_x - x), bc_turn, cruise
            heading += clamp(wrap(aim - heading), -limit, limit)
            nx = x + step_speed * dt * math.cos(heading)
            ny = y + step_speed * dt * math.sin(heading)
            for ox, oy, ovx, ovy in others:
                best = min(
                    best,
                    closest_in_step(
                        x, y, nx, ny,
                        ox + ovx * dt * k, oy + ovy * dt * k,
                        ox + ovx * dt * (k + 1), oy + ovy * dt * (k + 1),
                    ),
                )
            x, y = nx, ny
        return best

    def plan(self, ship_id, ship, others, limits, steps_left=None):
        steps_left = self.horizon_steps if steps_left is None else steps_left
        cruise = self._cruise.setdefault(ship_id, clamp(max(ship["speed"], 1.0), 1.0, limits.max_speed))
        bearing = goal_bearing(ship)
        previous = self._previous_target.get(ship_id, bearing)

        # Only vessels that can possibly matter inside the horizon.
        reach = (cruise + limits.max_speed) * self.horizon_steps * limits.dt + self.safe_dist
        nearby = []
        for other in others:
            if math.hypot(other["x"] - ship["x"], other["y"] - ship["y"]) <= reach:
                nearby.append((
                    other["x"], other["y"],
                    other["speed"] * math.cos(other["heading"]),
                    other["speed"] * math.sin(other["heading"]),
                ))

        speeds = sorted(
            {
                clamp(ship["speed"] * factor, self.min_speed_fraction * cruise, cruise)
                for factor in self.speed_factors
            },
            reverse=True,
        )

        best_safe = best_unsafe = None  # (cost or -clearance, target, speed)
        for offset in self.offsets:
            target = wrap(bearing + offset)
            for speed in speeds:
                clearance = self._clearance(ship, target, speed, nearby, limits, steps_left, cruise)
                cost = (
                    self.w_goal * abs(offset) / math.pi
                    + self.w_speed * (1.0 - speed / cruise)
                    + self.w_switch * abs(wrap(target - previous)) / math.pi
                    + (self.w_port * offset / math.pi if offset > 0 else 0.0)
                )
                if clearance >= self.safe_dist:
                    if best_safe is None or cost < best_safe[0]:
                        best_safe = (cost, target, speed)
                else:
                    key = (-clearance, cost)
                    if best_unsafe is None or key < best_unsafe[0]:
                        best_unsafe = (key, target, speed)

        _, target, speed = best_safe if best_safe is not None else best_unsafe
        self._previous_target[ship_id] = target
        turn = clamp(wrap(target - ship["heading"]), -min(self.turn_limit, limits.max_turn), min(self.turn_limit, limits.max_turn))
        return turn, clamp(speed, 0.0, limits.max_speed)
