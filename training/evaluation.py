"""Run allocation policies on FleetAllocationEnv and collect episode metrics."""

from typing import Callable, Dict, Iterable, List

import numpy as np

from .dqn import MaskedDQNAgent
from .features import ObservationEncoder


class DQNPolicy:
    """Wrap a trained agent with the same call signature as the baselines."""

    def __init__(self, agent: MaskedDQNAgent, encoder: ObservationEncoder):
        self.agent = agent
        self.encoder = encoder

    def __call__(self, observation, info, env):
        x = self.encoder.encode(observation, info["fleet_state"], env.vessel_ids)
        return self.agent.act(x, info["action_mask"], explore=False)


def run_episode(env, policy: Callable, seed: int) -> Dict[str, float]:
    """Play one episode; return totals the report tables need."""
    observation, info = env.reset(seed=seed)
    totals = {
        "reward": 0.0,
        "completed_journeys": 0,
        "collision_events": 0,
        "near_miss_events": 0,
        "interventions": 0,
        "invalid_actions": 0,
        "steps": 0,
    }
    while True:
        action = int(policy(observation, info, env))
        if not info["action_mask"][action]:
            totals["invalid_actions"] += 1
        observation, reward, terminated, truncated, info = env.step(action)
        totals["reward"] += reward
        totals["steps"] += 1
        for key, value in info["reward_components"].items():
            totals[key] += value
        if terminated or truncated:
            break

    statuses = [ship["status"] for ship in info["fleet_state"]["ships"].values()]
    minima = env.fleet.get_distance_metrics()["episode_minima"]
    totals.update({
        "arrived": statuses.count("arrived"),
        "out_of_bounds": statuses.count("out of bounds"),
        "time_limit": statuses.count("time limit"),
        "budget_used_s": env.tracker.total_budget - env.tracker.remaining_budget,
        "min_pair_distance_m": float(min((m["distance_m"] for m in minima.values()), default=np.nan)),
        "scenario_seed": getattr(env, "scenario_seed", -1),
        "truncated": bool(truncated),
    })
    return totals


def evaluate(envs: Iterable, policy: Callable, episodes_per_env: int = 1) -> List[Dict[str, float]]:
    results = []
    for env in envs:
        for k in range(episodes_per_env):
            results.append(run_episode(env, policy, seed=getattr(env, "scenario_seed", 0) + k))
    return results


SUMMARY_KEYS = (
    "reward",
    "collision_events",
    "near_miss_events",
    "completed_journeys",
    "interventions",
    "budget_used_s",
    "min_pair_distance_m",
    "invalid_actions",
)


def summarise(results: List[Dict[str, float]]) -> Dict[str, float]:
    summary = {"episodes": len(results)}
    for key in SUMMARY_KEYS:
        values = np.array([r[key] for r in results], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values)) if len(values) else float("nan")
        summary[f"{key}_std"] = float(np.nanstd(values)) if len(values) else float("nan")
    return summary


def format_table(summaries: Dict[str, Dict[str, float]]) -> str:
    cols = [
        ("reward", "reward"),
        ("collision_events", "collisions"),
        ("near_miss_events", "near-miss"),
        ("completed_journeys", "completed"),
        ("interventions", "interv."),
        ("budget_used_s", "budget(s)"),
        ("min_pair_distance_m", "min dist(m)"),
        ("invalid_actions", "invalid"),
    ]
    header = f"{'policy':<16}" + "".join(f"{name:>13}" for _, name in cols)
    lines = [header, "-" * len(header)]
    for policy, s in summaries.items():
        row = f"{policy:<16}"
        for key, _ in cols:
            row += f"{s[key + '_mean']:>8.2f}±{s[key + '_std']:<4.1f}"
        lines.append(row)
    return "\n".join(lines)


__all__ = ["DQNPolicy", "evaluate", "format_table", "run_episode", "summarise"]
