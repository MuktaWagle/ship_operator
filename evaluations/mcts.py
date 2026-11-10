"""Monte-Carlo tree search allocator"""

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AVAILABLE, PREPARING, ASSISTING = 0, 1, 2
_STATUS = {"AVAILABLE": AVAILABLE, "PREPARING": PREPARING, "ASSISTING": ASSISTING}


@dataclass
class ModelMCTSConfig:
    rollouts: int = 150
    tree_depth: int = 12           # decision-tree depth in steps
    horizon: int = 40              # total steps simulated per rollout
    gamma: float = 0.99
    c_explore: float = 2.0         # in reward units
    top_k: int = 3                 # riskiest ships considered besides idle
    activation_m: float = 1000.0   # search only if an available ship's predicted CPA < this
    default_dispatch_prob: float = 0.15   # rollout policy: dispatch the riskiest ship with this prob.
    goal_threshold_m: float = 200.0       # arrival radius used by the simulator's ships
    risk_horizon_s: float = 300.0
    seed: int = 0


@dataclass
class ModelParams:
    dt: float
    delay: float
    duration: float
    warning_m: float
    collision_m: float
    goal_threshold_m: float
    max_x: float
    max_y: float
    direction: str
    turn_rad: float
    w_collision: float
    w_near_miss: float
    w_intervention: float
    risk_horizon_s: float
    # Task 5 planning operator (direction == "avoid"): the model calls a copy of the planner
    # for the assisted vessel instead of applying a fixed turn. None for left/right/stop.
    planner: Any = None
    limits: Any = None
    plan_cache: Any = None


class ModelState:
    __slots__ = (
        "pos", "heading", "speed", "goal", "active", "status", "prep", "assist",
        "assigned", "budget", "takeover", "op_left", "saved_speed", "near_pairs",
        "coll_pairs", "t", "plan_mem",
    )

    def copy(self) -> "ModelState":
        new = ModelState.__new__(ModelState)
        new.pos = self.pos.copy()
        new.heading = self.heading.copy()
        new.speed = self.speed.copy()
        new.goal = self.goal  # never modified
        new.active = self.active.copy()
        new.status = self.status
        new.prep = self.prep
        new.assist = self.assist
        new.assigned = self.assigned
        new.budget = self.budget
        new.takeover = self.takeover
        new.op_left = self.op_left
        new.saved_speed = self.saved_speed
        new.near_pairs = self.near_pairs.copy()
        new.coll_pairs = self.coll_pairs.copy()
        new.t = self.t
        new.plan_mem = self.plan_mem
        return new


def _model_planner(env, shared: dict):
    """A private copy of the env's planner plus a plan cache, reused across searches.

    The copy keeps the live operator's memory untouched. The cache is dropped if the live
    planner's configuration changes (e.g. the policy is reused with a different planner).
    """
    live = env.controller.planner
    signature = (type(live).__name__, tuple(sorted(
        (k, repr(v)) for k, v in vars(live).items() if not k.startswith("_")
    )))
    if shared.get("signature") != signature:
        shared["signature"] = signature
        shared["planner"] = copy.deepcopy(live)
        shared["planner"].reset()
        shared["cache"] = {}
    return shared["planner"], env.controller.limits, shared["cache"]


def params_from_env(env, config: ModelMCTSConfig, shared: Optional[dict] = None) -> ModelParams:
    tracker = env.tracker
    weights = env.reward_weights
    direction = env.assistance_direction
    dt = float(env.fleet.dt)
    planner = limits = cache = None
    if direction == "avoid":
        turn = 0.0
        planner, limits, cache = _model_planner(env, shared if shared is not None else {})
    elif direction == "stop":
        turn = 0.0
    else:
        turn = math.radians(env.assistance_turn_rate_deg) * dt * (1 if direction == "left" else -1)
    return ModelParams(
        dt=dt, delay=float(tracker.delay), duration=float(tracker.duration),
        warning_m=float(env.warning_distance_m), collision_m=float(env.collision_distance_m),
        goal_threshold_m=config.goal_threshold_m, max_x=float(env.max_x), max_y=float(env.max_y),
        direction=direction, turn_rad=turn,
        w_collision=float(weights.collision), w_near_miss=float(weights.near_miss),
        w_intervention=float(weights.intervention), risk_horizon_s=config.risk_horizon_s,
        planner=planner, limits=limits, plan_cache=cache,
    )


def _pair_sets(pos, active, p: ModelParams):
    n = len(pos)
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    valid = np.triu(np.ones((n, n), dtype=bool), k=1) & active[:, None] & active[None, :]
    coll = valid & (dist < p.collision_m)
    near = valid & (dist >= p.collision_m) & (dist < p.warning_m)
    return coll, near


def build_state(env, info) -> ModelState:
    """Build the model state from observable quantities only."""
    ships = info["fleet_state"]["ships"]
    ids = env.vessel_ids
    s = ModelState.__new__(ModelState)
    s.pos = np.array([[ships[i]["x"], ships[i]["y"]] for i in ids], dtype=float)
    s.heading = np.array([ships[i]["heading"] for i in ids], dtype=float)
    s.speed = np.array([ships[i]["speed"] for i in ids], dtype=float)
    s.goal = np.array([ships[i]["goal"] for i in ids], dtype=float)
    s.active = np.array([ships[i]["status"] == "active" for i in ids], dtype=bool)

    tracker = env.tracker
    s.status = _STATUS[tracker.status.name]
    s.prep = float(tracker.preparation_timer)
    s.assist = float(tracker.assistance_timer)
    s.budget = float(tracker.remaining_budget)
    s.assigned = ids.index(tracker.assigned_ship_id) if tracker.assigned_ship_id in ids else -1
    s.takeover, s.op_left, s.saved_speed = False, 0.0, 0.0
    if s.assigned >= 0 and env.controller.mode.get(ids[s.assigned]) == "operator":
        s.takeover = True
        s.op_left = float(env.controller.remaining_seconds(ids[s.assigned]))
        s.saved_speed = float(s.speed[s.assigned])

    p = env._mcts_params
    s.plan_mem = None
    if s.takeover and p.planner is not None:
        s.plan_mem = env.controller.planner.snapshot(ids[s.assigned])  # live memory of the running takeover
    s.coll_pairs, s.near_pairs = _pair_sets(s.pos, s.active, p)
    s.t = 0
    return s


def _release(s: ModelState):
    if s.takeover and s.assigned >= 0 and s.active[s.assigned]:
        a = s.assigned
        delta = s.goal[a] - s.pos[a]
        s.heading[a] = math.atan2(delta[1], delta[0])
        s.speed[a] = s.saved_speed
    s.takeover = False
    s.op_left = 0.0


def _planned_move(p: ModelParams, s: ModelState, a: int):
    """Let the planner choose this step's turn and speed for the assisted vessel `a`.

    Other vessels are treated as constant-velocity (the planner's own assumption). Identical
    situations are answered from a cache, because many rollouts share the same prefix.
    """
    steps_left = max(1, int(math.ceil(s.op_left / p.dt - 1e-9)))
    idx = np.flatnonzero(s.active)
    table = np.round(np.column_stack((s.pos, s.heading, s.speed))[idx], 2)
    key = (a, steps_left, s.plan_mem, idx.tobytes(), table.tobytes())
    hit = p.plan_cache.get(key)
    if hit is None:
        ships = {
            int(i): {
                "x": float(s.pos[i, 0]), "y": float(s.pos[i, 1]),
                "heading": float(s.heading[i]), "speed": float(s.speed[i]),
                "goal": (float(s.goal[i, 0]), float(s.goal[i, 1])),
            }
            for i in idx
        }
        others = [ships[int(i)] for i in idx if i != a]
        p.planner.restore(a, s.plan_mem)
        turn, speed = p.planner.plan(a, ships[a], others, p.limits, steps_left)
        hit = (float(turn), float(speed), p.planner.snapshot(a))
        if len(p.plan_cache) > 200_000:
            p.plan_cache.clear()
        p.plan_cache[key] = hit
    turn, speed, s.plan_mem = hit
    s.heading[a] = (s.heading[a] + turn + math.pi) % (2.0 * math.pi) - math.pi
    s.speed[a] = speed


def model_step(p: ModelParams, s: ModelState, action: int):
    """Advance the internal model one step (mirrors FleetAllocationEnv.step ordering)."""
    dispatched = False
    if action > 0 and s.status == AVAILABLE and s.budget >= p.duration and s.active[action - 1]:
        s.assigned = action - 1
        dispatched = True
        s.assist = min(p.duration, s.budget)
        if p.delay > 0:
            s.status, s.prep = PREPARING, p.delay
        else:
            s.status, s.prep = ASSISTING, 0.0

    if s.status == ASSISTING and s.assigned >= 0 and s.active[s.assigned] and not s.takeover:
        a = s.assigned
        s.takeover = True
        s.op_left = s.assist
        s.saved_speed = float(s.speed[a]) if s.speed[a] > 0 else 3.0
        s.speed[a] = 0.0 if p.direction == "stop" else s.saved_speed
        s.plan_mem = None  # a new takeover starts with a clean planner memory

    if s.takeover and s.assigned >= 0 and s.active[s.assigned]:
        a = s.assigned
        if p.planner is not None:
            _planned_move(p, s, a)
        else:
            s.heading[a] += p.turn_rad
            if p.direction == "stop":
                s.speed[a] = 0.0

    was_active = s.active.copy()
    velocity = s.speed[:, None] * np.stack([np.cos(s.heading), np.sin(s.heading)], axis=1)
    s.pos = s.pos + velocity * p.dt * was_active[:, None]

    if s.takeover:
        s.op_left -= p.dt
        if s.op_left <= 1e-9:
            _release(s)

    reached = np.linalg.norm(s.goal - s.pos, axis=1) < p.goal_threshold_m
    outside = (s.pos[:, 0] < 0) | (s.pos[:, 0] > p.max_x) | (s.pos[:, 1] < 0) | (s.pos[:, 1] > p.max_y)

    coll, near = _pair_sets(s.pos, was_active, p)
    reward = -(
        p.w_collision * float((coll & ~s.coll_pairs).sum())
        + p.w_near_miss * float((near & ~s.near_pairs).sum())
        + p.w_intervention * float(dispatched)
    )
    s.coll_pairs, s.near_pairs = coll, near

    s.active = was_active & ~(reached | outside)

    if s.status == PREPARING:
        s.prep -= p.dt
        if s.prep <= 1e-9:
            s.status, s.prep = ASSISTING, 0.0
    elif s.status == ASSISTING:
        used = min(p.dt, s.assist, s.budget)
        s.budget = max(0.0, s.budget - used)
        s.assist = max(0.0, s.assist - p.dt)
        if s.assist <= 1e-9 or s.budget <= 1e-9 or (s.assigned >= 0 and not s.active[s.assigned]):
            _release(s)
            s.status, s.assigned, s.assist, s.prep = AVAILABLE, -1, 0.0, 0.0

    s.t += 1
    return reward, not bool(s.active.any())


def predicted_cpa(p: ModelParams, s: ModelState) -> np.ndarray:
    """Per-ship predicted closest-approach distance (m) under constant velocity."""
    n = len(s.pos)
    vel = s.speed[:, None] * np.stack([np.cos(s.heading), np.sin(s.heading)], axis=1)
    rel_p = s.pos[None, :, :] - s.pos[:, None, :]
    rel_v = vel[None, :, :] - vel[:, None, :]
    speed_sq = (rel_v ** 2).sum(-1)
    safe = np.where(speed_sq < 1e-9, 1.0, speed_sq)
    t = np.clip(-(rel_p * rel_v).sum(-1) / safe, 0.0, p.risk_horizon_s)
    t = np.where(speed_sq < 1e-9, 0.0, t)
    dist = np.linalg.norm(rel_p + rel_v * t[..., None], axis=-1)
    pair = s.active[:, None] & s.active[None, :] & ~np.eye(n, dtype=bool)
    dist = np.where(pair, dist, np.inf)
    return dist.min(axis=1)


class _Node:
    __slots__ = ("children", "untried", "visits", "total")

    def __init__(self):
        self.children = {}
        self.untried = None
        self.visits = 0
        self.total = 0.0


class ModelMCTSPolicy:
    """Policy wrapper with the baseline call signature ``(obs, info, env) -> int``."""

    def __init__(self, config: Optional[ModelMCTSConfig] = None):
        self.cfg = config or ModelMCTSConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.searches = 0
        self._shared = {}  # planner copy + plan cache, used when the operator is a planner

    def __call__(self, observation, info, env):
        mask = np.asarray(info["action_mask"], dtype=bool)
        if not mask[1:].any():
            return 0
        env._mcts_params = params_from_env(env, self.cfg, self._shared)
        params = env._mcts_params
        root_state = build_state(env, info)

        cpa = predicted_cpa(params, root_state)
        valid = [a for a in range(1, env.num_vessels + 1) if mask[a]]
        if min(cpa[a - 1] for a in valid) >= self.cfg.activation_m:
            return 0

        root = _Node()
        for _ in range(self.cfg.rollouts):
            self._rollout(params, root, root_state)
        self.searches += 1
        if not root.children:
            return 0
        best = max(root.children, key=lambda a: (root.children[a].visits,
                                                  root.children[a].total / max(1, root.children[a].visits)))
        return int(best) if mask[best] else 0

    # ---------------------------------------------------------------- search
    def _free(self, p, s):
        return s.status == AVAILABLE and s.budget >= p.duration

    def _candidates(self, p, s) -> List[int]:
        if not self._free(p, s):
            return [0]
        cpa = predicted_cpa(p, s)
        ships = [i + 1 for i in range(len(cpa)) if s.active[i]]
        ships.sort(key=lambda a: cpa[a - 1])
        candidates = [0] + ships[: self.cfg.top_k]
        self.rng.shuffle(candidates)
        return candidates

    def _default_action(self, p, s) -> int:
        if not self._free(p, s) or self.rng.random() >= self.cfg.default_dispatch_prob:
            return 0
        cpa = predicted_cpa(p, s)
        if not np.isfinite(cpa).any():
            return 0
        target = int(np.argmin(cpa))
        return target + 1 if cpa[target] < p.warning_m and s.active[target] else 0

    def _select(self, node):
        log_n = math.log(max(1, node.visits))
        best_score, best_action = -math.inf, None
        for action, child in node.children.items():
            if child.visits == 0:
                return action
            score = child.total / child.visits + self.cfg.c_explore * math.sqrt(log_n / child.visits)
            if score > best_score:
                best_score, best_action = score, action
        return best_action

    def _rollout(self, p, root, root_state):
        cfg = self.cfg
        s = root_state.copy()
        node, path, rewards = root, [], []
        done, expanded = False, False

        while len(path) < cfg.tree_depth and not done and not expanded:
            if node.untried is None:
                node.untried = self._candidates(p, s)
            if node.untried:
                action = node.untried.pop()
                expanded = True
            elif node.children:
                action = self._select(node)
            else:
                break
            child = node.children.get(action)
            if child is None:
                child = _Node()
                node.children[action] = child
            reward, done = model_step(p, s, action)
            path.append(child)
            rewards.append(reward)
            node = child

        while not done and len(rewards) < cfg.horizon:
            reward, done = model_step(p, s, self._default_action(p, s))
            rewards.append(reward)

        returns, running = [0.0] * len(rewards), 0.0
        for k in reversed(range(len(rewards))):
            running = rewards[k] + cfg.gamma * running
            returns[k] = running
        root.visits += 1
        for k, child in enumerate(path):
            child.visits += 1
            child.total += returns[k]


__all__ = ["ModelMCTSConfig", "ModelMCTSPolicy"]
