"""How hard is a scenario setting? Run each one with NO intervention and count conflicts.

For every setting this plays full episodes of an idle fleet (the operator is never used) on
the real frozen BC policy and reports, over the scenarios:

    arrived     share of vessels that reach their goal. Low means BC cannot sail these
                routes, so the scenario is not usable.
    collide     share of scenarios with at least one pair closer than the collision distance
    near        share of scenarios with at least one pair closer than the warning distance
    pairs_c/n   average number of distinct pairs that collided / had a near miss
    closest     average closest approach in the scenario, metres
    usable      "yes" when arrival >= 0.9, collisions in at least 20% of scenarios, near misses in
                at least half, and on average no more than 4 collided pairs (one operator
                cannot rescue more than a handful). These cut-offs are only a starting point.

Two kinds of setting:

    --real N:D        N vessels picked from the real journeys with start separation D metres
                      (what training/scenarios.py does today, e.g. 10:800; the demos use D=100)
    --synthetic N:S[:O]
                      N vessels on straight crossing routes, arrival-time spread S seconds and
                      optional sideways route offset O metres (default 300)
                      (training/synthetic_scenarios.py; smaller S and O = denser conflicts)

Usage:
    python training/scan_difficulty.py
    python training/scan_difficulty.py --real 10:800 10:100 15:100 --synthetic 6:30 10:100:1500 --seeds 20
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bc_navigation import BCController, FrozenBCPolicy, load_bc_scenario  # noqa: E402
from src.allocation_env import FleetAllocationEnv, RewardWeights  # noqa: E402
from src.fleet_simulator import FleetSimulator  # noqa: E402
from src.operator_tracker import OperatorBudgetTracker  # noqa: E402
from training.synthetic_scenarios import make_crossing_scenario  # noqa: E402

REGION = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}


def run_idle(scenario, policy, seed, max_steps, warning, collision):
    trajectories, timestamps, vessel_ids = scenario
    fleet = FleetSimulator(
        trajectories=trajectories, timestamps=timestamps, vessel_ids=vessel_ids,
        region_of_interest=REGION, max_steps=max_steps, enable_graphics=False, dt=10,
    )
    controller = BCController(fleet, policy=policy)
    tracker = OperatorBudgetTracker(
        vessel_ids=fleet.vessel_ids, total_budget=300.0, duration=60.0, delay=10.0,
        dt=fleet.dt, max_simulation_time=max_steps * fleet.dt,
    )
    env = FleetAllocationEnv(
        fleet, controller, tracker, warning_distance_m=warning, collision_distance_m=collision,
        reward_weights=RewardWeights(), speed_scale_mps=10.0,
    )
    try:
        _, info = env.reset(seed=seed)
        while True:
            _, _, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        minima = env.fleet.get_distance_metrics()["episode_minima"]
        distances = [m["distance_m"] for m in minima.values()]
        statuses = [s["status"] for s in info["fleet_state"]["ships"].values()]
        return {
            "vessels": len(vessel_ids),
            "arrived": statuses.count("arrived") / len(vessel_ids),
            "out_of_bounds": statuses.count("out of bounds") / len(vessel_ids),
            "time_limit": statuses.count("time limit") / len(vessel_ids),
            "pairs_collided": sum(d < collision for d in distances),
            "pairs_near": sum(collision <= d < warning for d in distances),
            "closest_m": min(distances) if distances else float("nan"),
        }
    finally:
        env.close()


def parse_setting(text, name):
    """'N:V' or 'N:V:O' -> (N, V, O); O (route offset, metres) only matters for synthetic."""
    try:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise ValueError
        return int(parts[0]), float(parts[1]), float(parts[2]) if len(parts) == 3 else 300.0
    except ValueError:
        raise SystemExit(f"--{name} expects N:NUMBER (or N:NUMBER:OFFSET), for example 10:800 (got {text!r})")


def summarise(rows):
    arrived = np.mean([r["arrived"] for r in rows])
    collide = np.mean([r["pairs_collided"] > 0 for r in rows])
    near = np.mean([(r["pairs_near"] + r["pairs_collided"]) > 0 for r in rows])
    pairs_c = np.mean([r["pairs_collided"] for r in rows])
    usable = arrived >= 0.9 and collide >= 0.2 and near >= 0.5 and pairs_c <= 4
    return {
        "episodes": len(rows),
        "arrived": arrived,
        "out_of_bounds": np.mean([r["out_of_bounds"] for r in rows]),
        "time_limit": np.mean([r["time_limit"] for r in rows]),
        "collide": collide,
        "near": near,
        "pairs_c": np.mean([r["pairs_collided"] for r in rows]),
        "pairs_n": np.mean([r["pairs_near"] for r in rows]),
        "closest": np.nanmean([r["closest_m"] for r in rows]),
        "usable": "yes" if usable else "no",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", nargs="*", default=["10:800", "10:300", "10:100", "15:100"], metavar="N:D")
    p.add_argument("--synthetic", nargs="*", default=["6:100:300", "6:250:1500", "10:100:300", "10:250:1000", "10:400:1500"],
                   metavar="N:S[:O]")
    p.add_argument("--seeds", type=int, default=15, help="scenarios per setting")
    p.add_argument("--first-seed", type=int, default=30_000, help="kept away from training seeds")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--warning", type=float, default=500.0)
    p.add_argument("--collision", type=float, default=50.0)
    p.add_argument("--out", default="difficulty_scan.csv")
    args = p.parse_args(argv)

    policy = FrozenBCPolicy()
    settings = [("real", *parse_setting(t, "real")) for t in args.real]
    settings += [("synthetic", *parse_setting(t, "synthetic")) for t in args.synthetic]

    header = (f"{'setting':<22}{'ep':>4}{'arrived':>9}{'oob':>6}{'tlim':>6}{'collide':>9}{'near':>7}"
              f"{'pairs_c':>9}{'pairs_n':>9}{'closest(m)':>12}{'usable':>8}")
    print(header)
    print("-" * len(header))
    all_rows, csv_rows = [], []
    for kind, n, value, offset in settings:
        label = f"{kind} {n}:{value:g}" + (f":{offset:g}" if kind == "synthetic" else "")
        rows = []
        for k in range(args.seeds):
            seed = args.first_seed + k
            try:
                if kind == "real":
                    scenario = load_bc_scenario(
                        policy, fleet_size=n, seed=seed, min_start_distance=value, split="train",
                    )
                else:
                    scenario = make_crossing_scenario(n, seed, spread_s=value, offset_m=offset)
                rows.append(run_idle(scenario, policy, seed, args.max_steps, args.warning, args.collision))
            except ValueError as error:  # e.g. cannot pick N separated real ships
                print(f"{label:<22} seed {seed}: skipped ({error})")
        if not rows:
            continue
        s = summarise(rows)
        print(f"{label:<22}{s['episodes']:>4}{s['arrived']:>9.2f}{s['out_of_bounds']:>6.2f}{s['time_limit']:>6.2f}"
              f"{s['collide']:>9.2f}{s['near']:>7.2f}{s['pairs_c']:>9.2f}{s['pairs_n']:>9.2f}"
              f"{s['closest']:>12.0f}{s['usable']:>8}")
        csv_rows.append({"setting": label, **s})

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved {args.out}")
    print("arrived = share of vessels reaching their goal; oob/tlim = out of bounds / time limit;")
    print("collide / near = share of scenarios with at least one collision / near miss;")
    print("pairs_c / pairs_n = average number of vessel pairs that collided / had a near miss.")


if __name__ == "__main__":
    main()
