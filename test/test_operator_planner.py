"""Unit checks for the Task 5 operator planners.

These use plain dictionaries for ships, so they need no BC checkpoint, no journey
data and no simulator. Run with either:

    python -m pytest test/test_operator_planner.py
    python test/test_operator_planner.py
"""

import math
import sys
from pathlib import Path

import importlib.util

# Load planners.py directly instead of `import operator_planner`: the package
# __init__ also imports the controller, which needs torch. The planners do not.
_path = Path(__file__).resolve().parents[1] / "operator_planner" / "planners.py"
_spec = importlib.util.spec_from_file_location("planners_under_test", _path)
_planners = importlib.util.module_from_spec(_spec)
sys.modules["planners_under_test"] = _planners  # dataclasses needs the module registered
_spec.loader.exec_module(_planners)

Limits = _planners.Limits
RuleBasedPlanner = _planners.RuleBasedPlanner
VelocityObstaclePlanner = _planners.VelocityObstaclePlanner
closest_in_step = _planners.closest_in_step
wrap = _planners.wrap

LIMITS = Limits(dt=10.0, max_turn=math.radians(30.0), max_speed=10.0)


def ship(x, y, heading, speed, goal):
    return {"x": x, "y": y, "heading": heading, "speed": speed, "goal": goal}


def head_on():
    """Assisted ship heads east, the other heads west on the same line."""
    mine = ship(0.0, 0.0, 0.0, 8.0, (6000.0, 0.0))
    other = ship(3000.0, 0.0, math.pi, 8.0, (-3000.0, 0.0))
    return mine, other


def advance(s, turn, speed, dt):
    """Same motion rule as BCController._movement: move along the NEW heading."""
    heading = wrap(s["heading"] + turn)
    s["x"] += speed * dt * math.cos(heading)
    s["y"] += speed * dt * math.sin(heading)
    s["heading"], s["speed"] = heading, speed


def straight(s, dt):
    s["x"] += s["speed"] * dt * math.cos(s["heading"])
    s["y"] += s["speed"] * dt * math.sin(s["heading"])


def simulate(planner, mine, other, steps=40, control_steps=None):
    """Planner steers `mine` for `control_steps`, then it heads for its goal.
    Returns the smallest separation seen, using the reward's closest-in-step geometry."""
    planner.reset()
    closest = math.inf
    for k in range(steps):
        x0, y0, ox0, oy0 = mine["x"], mine["y"], other["x"], other["y"]
        if control_steps is None or k < control_steps:
            left = None if control_steps is None else control_steps - k
            turn, speed = planner.plan("A", dict(mine), [dict(other)], LIMITS, left)
        else:
            bearing = math.atan2(mine["goal"][1] - mine["y"], mine["goal"][0] - mine["x"])
            turn = max(-LIMITS.max_turn, min(LIMITS.max_turn, wrap(bearing - mine["heading"])))
            speed = 8.0
        advance(mine, turn, speed, LIMITS.dt)
        straight(other, LIMITS.dt)
        closest = min(closest, closest_in_step(x0, y0, mine["x"], mine["y"], ox0, oy0, other["x"], other["y"]))
    return closest


def test_unassisted_head_on_is_a_collision():
    mine, other = head_on()
    assert simulate(_Straight(), mine, other) < 50.0


class _Straight:
    """Reference: keep heading and speed (what an idle vessel would do)."""

    def reset(self):
        pass

    def plan(self, ship_id, s, others, limits, steps_left=None):
        return 0.0, s["speed"]


def test_no_threat_steers_for_goal():
    mine = ship(0.0, 0.0, 0.0, 8.0, (6000.0, 0.0))
    far = ship(0.0, 5000.0, 0.0, 8.0, (6000.0, 5000.0))  # parallel, 5 km away
    for planner in (VelocityObstaclePlanner(), RuleBasedPlanner()):
        planner.reset()
        turn, speed = planner.plan("A", mine, [far], LIMITS)
        assert abs(turn) < math.radians(1.0), planner.name
        assert speed >= 7.9, planner.name


def test_head_on_turns_to_starboard_first():
    # Starboard (right) is a NEGATIVE turn in this simulator.
    mine, other = head_on()
    for planner in (VelocityObstaclePlanner(), RuleBasedPlanner()):
        planner.reset()
        turn, _ = planner.plan("A", mine, [other], LIMITS)
        assert turn < 0.0, planner.name


def test_velocity_obstacle_clears_head_on_with_full_control():
    mine, other = head_on()
    assert simulate(VelocityObstaclePlanner(), mine, other) >= 500.0


def test_velocity_obstacle_avoids_collision_when_handed_back():
    # Operator controls only 6 steps (60 s), then BC-like goal seeking resumes.
    mine, other = head_on()
    assert simulate(VelocityObstaclePlanner(), mine, other, control_steps=6) >= 50.0


def test_rule_based_avoids_collision():
    mine, other = head_on()
    assert simulate(RuleBasedPlanner(), mine, other) >= 50.0


def test_commands_respect_limits():
    mine, other = head_on()
    for planner in (VelocityObstaclePlanner(), RuleBasedPlanner()):
        planner.reset()
        for _ in range(20):
            turn, speed = planner.plan("A", mine, [other], LIMITS)
            assert abs(turn) <= LIMITS.max_turn + 1e-9, planner.name
            assert 0.0 <= speed <= LIMITS.max_speed + 1e-9, planner.name
            advance(mine, turn, speed, LIMITS.dt)
            straight(other, LIMITS.dt)


def test_reset_clears_memory():
    planner = VelocityObstaclePlanner()
    mine, other = head_on()
    planner.plan("A", mine, [other], LIMITS)
    assert planner._cruise and planner._previous_target
    planner.reset()
    assert not planner._cruise and not planner._previous_target


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name} {error!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
