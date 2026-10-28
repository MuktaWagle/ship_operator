"""Build FleetAllocationEnv instances and a pool of training scenarios.

``FleetSimulator`` is built around one fixed set of journeys, and the frozen
BC navigator is deterministic, so resetting the same fleet replays the same
traffic.  A DQN trained on one scenario would just memorise it.  ``ScenarioPool``
therefore holds several environments, each built from a different random
selection (``scenario_seed``) of BC journeys, and picks one at every reset.
Every scenario has the same fleet size N, so the action space {0..N} and the
observation shape stay fixed; only *which* real ship sits in each slot changes.

Data limits to keep in mind.  After load_bc_scenario's validity filter the
prepared BC data has only:
    train split: 15 usable journeys    val split: 4    test split: 4
Consequences:
  * fleet_size=15 always uses the same 15 ships; seeds only reorder slots,
    and BC is deterministic, so every episode is the same traffic.
  * fleet_size >= 5 cannot be built from val or test at all.
  * For 5..14 ships, held-out evaluation uses train-journey *combinations*
    that never appeared in the training pool (``build_heldout_envs``).
  * fleet_size=10 with 800 m start separation gives ~26 distinct ship sets
    in 50 seeds and near-misses in most of them: the best default.
"""

from dataclasses import asdict, dataclass, field
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bc_navigation import BCController, FrozenBCPolicy, load_bc_scenario  # noqa: E402
from src.allocation_env import FleetAllocationEnv, RewardWeights  # noqa: E402
from src.fleet_simulator import FleetSimulator  # noqa: E402
from src.operator_tracker import OperatorBudgetTracker  # noqa: E402


REGION_OF_INTEREST = {
    "LON": (103.82, 103.88),
    "LAT": (1.15, 1.22),
}


@dataclass
class EnvConfig:
    """Everything needed to rebuild the same environment (saved with checkpoints)."""

    fleet_size: int = 10
    split: str = "train"
    max_steps: int = 300
    dt: int = 10
    min_start_distance_m: float = 800.0
    total_budget_s: float = 300.0
    duration_s: float = 60.0
    delay_s: float = 10.0
    warning_distance_m: float = 500.0
    collision_distance_m: float = 50.0
    assistance_direction: str = "right"
    assistance_turn_rate_deg: float = 1.0
    speed_scale_mps: float = 10.0
    invalid_action_penalty: float = 0.0
    reward_completed: float = 1.0
    reward_collision: float = 10.0
    reward_near_miss: float = 1.0
    reward_intervention: float = 1.0

    def reward_weights(self) -> RewardWeights:
        return RewardWeights(
            completed_journey=self.reward_completed,
            collision=self.reward_collision,
            near_miss=self.reward_near_miss,
            intervention=self.reward_intervention,
        )

    def to_dict(self) -> Dict:
        return asdict(self)


_POLICY_CACHE: Dict[str, FrozenBCPolicy] = {}


def shared_bc_policy() -> FrozenBCPolicy:
    """Load the frozen BC checkpoint once and share it across all fleets."""
    if "policy" not in _POLICY_CACHE:
        _POLICY_CACHE["policy"] = FrozenBCPolicy()
    return _POLICY_CACHE["policy"]


def build_env(config: EnvConfig, scenario_seed: int, policy: Optional[FrozenBCPolicy] = None) -> FleetAllocationEnv:
    """One FleetAllocationEnv for one journey selection."""
    policy = policy or shared_bc_policy()
    trajectories, timestamps, vessel_ids = load_bc_scenario(
        policy,
        fleet_size=config.fleet_size,
        seed=scenario_seed,
        min_start_distance=config.min_start_distance_m,
        split=config.split,
    )
    fleet = FleetSimulator(
        trajectories=trajectories,
        timestamps=timestamps,
        vessel_ids=vessel_ids,
        region_of_interest=REGION_OF_INTEREST,
        max_steps=config.max_steps,
        enable_graphics=False,
        dt=config.dt,
    )
    controller = BCController(fleet, policy=policy)
    tracker = OperatorBudgetTracker(
        vessel_ids=fleet.vessel_ids,
        total_budget=config.total_budget_s,
        duration=config.duration_s,
        delay=config.delay_s,
        dt=fleet.dt,
        max_simulation_time=config.max_steps * fleet.dt,
    )
    env = FleetAllocationEnv(
        fleet,
        controller,
        tracker,
        warning_distance_m=config.warning_distance_m,
        collision_distance_m=config.collision_distance_m,
        reward_weights=config.reward_weights(),
        assistance_direction=config.assistance_direction,
        assistance_turn_rate_deg=config.assistance_turn_rate_deg,
        speed_scale_mps=config.speed_scale_mps,
        invalid_action_penalty=config.invalid_action_penalty,
    )
    env.scenario_seed = int(scenario_seed)
    env.ship_set = frozenset(int(v) for v in vessel_ids)
    return env


def build_unique_envs(
    config: EnvConfig,
    count: int,
    start_seed: int,
    exclude_sets=(),
    max_tries: int = 500,
) -> List[FleetAllocationEnv]:
    """Build ``count`` envs with distinct ship sets, skipping ``exclude_sets``.

    Seeds that only reshuffle an already-used ship set are skipped, so the
    returned scenarios are genuinely different traffic.  If fewer distinct
    sets exist (e.g. fleet_size=15), returns as many as it finds.
    """
    policy = shared_bc_policy()
    seen = {frozenset(s) for s in exclude_sets}
    envs: List[FleetAllocationEnv] = []
    seed = start_seed
    while len(envs) < count and seed < start_seed + max_tries:
        env = build_env(config, seed, policy)
        seed += 1
        if env.ship_set in seen:
            env.close()
            continue
        seen.add(env.ship_set)
        envs.append(env)
    if not envs:
        raise ValueError("No scenario found that is not excluded.")
    if len(envs) < count:
        print(f"[scenarios] only {len(envs)} distinct ship sets available (asked {count}).")
    return envs


def build_heldout_envs(config: EnvConfig, train_sets, count: int, start_seed: int = 10_000):
    """Evaluation scenarios: the val split if it can hold the fleet, else unseen train combinations."""
    if config.fleet_size <= 4:
        val = EnvConfig(**{**config.to_dict(), "split": "val"})
        return build_unique_envs(val, count, start_seed), "val"
    return build_unique_envs(config, count, start_seed, exclude_sets=train_sets), "train-unseen"


class ScenarioPool:
    """A set of FleetAllocationEnvs with identical spaces; reset() samples one."""

    def __init__(self, config: EnvConfig, count: int, start_seed: int = 0, rng_seed: int = 0):
        self.config = config
        self.envs: List[FleetAllocationEnv] = build_unique_envs(config, count, start_seed)
        self.ship_sets = [sorted(env.ship_set) for env in self.envs]
        self.rng = np.random.default_rng(rng_seed)
        self.current: FleetAllocationEnv = self.envs[0]

        n = self.envs[0].num_vessels
        if any(env.num_vessels != n for env in self.envs):
            raise ValueError("All scenarios must have the same fleet size.")
        self.num_vessels = n
        self.action_space = self.envs[0].action_space

    def reset(self, index: Optional[int] = None):
        if index is None:
            index = int(self.rng.integers(len(self.envs)))
        self.current = self.envs[index]
        return self.current.reset(seed=self.current.scenario_seed)

    def step(self, action: int):
        return self.current.step(action)

    def close(self):
        for env in self.envs:
            env.close()


__all__ = [
    "EnvConfig",
    "REGION_OF_INTEREST",
    "ScenarioPool",
    "build_env",
    "build_heldout_envs",
    "build_unique_envs",
    "shared_bc_policy",
]
