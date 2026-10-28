"""Train the masked DQN operator allocator


The val/test splits only have 4 usable journeys each, so for fleets of 5+
ships the script evaluates on combinations of train journeys that never
appear in the training pool
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dqn import DQNConfig, MaskedDQNAgent  
from training.evaluation import DQNPolicy, evaluate, summarise  
from training.features import ObservationEncoder 
from training.scenarios import EnvConfig, ScenarioPool, build_heldout_envs 


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=None)
    p.add_argument("--fleet-size", type=int, default=10)
    p.add_argument("--min-start-distance", type=float, default=800.0)
    p.add_argument("--maneuver", choices=("right", "left", "stop"), default="right", help="fixed assistance manoeuvre (until Task 5 controller exists)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--train-scenarios", type=int, default=12, help="number of journey selections in the training pool")
    p.add_argument("--eval-scenarios", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=60_000, help="environment steps (one step = dt simulated seconds)")
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-risk-features", action="store_true", help="train on the raw Task 3 observation only")
    # operator / reward
    p.add_argument("--budget", type=float, default=300.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--delay", type=float, default=10.0)
    p.add_argument("--w-collision", type=float, default=10.0)
    p.add_argument("--w-near-miss", type=float, default=1.0)
    p.add_argument("--w-intervention", type=float, default=1.0)
    p.add_argument("--w-completed", type=float, default=1.0)
    # DQN
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=2_000)
    p.add_argument("--eps-decay-steps", type=int, default=None, help="default: half of total steps")
    p.add_argument("--explore-idle-prob", type=float, default=0.9,
                   help="chance an exploratory action is idle; <0 means uniform over valid actions")
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--no-dueling", action="store_true")
    p.add_argument("--no-double", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_name = args.run_name or f"dqn_n{args.fleet_size}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = PROJECT_ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_config = EnvConfig(
        fleet_size=args.fleet_size,
        split="train",
        max_steps=args.max_steps,
        min_start_distance_m=args.min_start_distance,
        assistance_direction=args.maneuver,
        total_budget_s=args.budget,
        duration_s=args.duration,
        delay_s=args.delay,
        reward_completed=args.w_completed,
        reward_collision=args.w_collision,
        reward_near_miss=args.w_near_miss,
        reward_intervention=args.w_intervention,
    )
    print(f"Building up to {args.train_scenarios} distinct training scenarios (fleet={args.fleet_size})...")
    pool = ScenarioPool(env_config, args.train_scenarios, start_seed=args.seed * 1000, rng_seed=args.seed)
    print(f"Building up to {args.eval_scenarios} held-out evaluation scenarios...")
    eval_envs, eval_split = build_heldout_envs(env_config, pool.ship_sets, args.eval_scenarios)
    if args.fleet_size >= 5 and len(pool.envs) == 1:
        print("WARNING: only one distinct scenario exists for this fleet size; the agent can only memorise it.")

    encoder = ObservationEncoder.for_env(pool.envs[0], risk_features=not args.no_risk_features)
    dqn_config = DQNConfig(
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        n_step=args.n_step,
        hidden=tuple(args.hidden),
        dueling=not args.no_dueling,
        double=not args.no_double,
        eps_decay_steps=args.eps_decay_steps or max(1, args.total_steps // 2),
        explore_idle_prob=None if args.explore_idle_prob < 0 else args.explore_idle_prob,
        seed=args.seed,
    )
    agent = MaskedDQNAgent(encoder.dim, pool.num_vessels + 1, dqn_config)

    checkpoint_extra = {
        "env_config": env_config.to_dict(),
        "encoder": {"risk_features": encoder.risk_features, "risk_horizon_s": encoder.risk_horizon_s},
        "train_seeds": [env.scenario_seed for env in pool.envs],
        "train_ship_sets": pool.ship_sets,
        "eval_split": eval_split,
        "eval_seeds": [env.scenario_seed for env in eval_envs],
        "eval_ship_sets": [sorted(env.ship_set) for env in eval_envs],
    }
    (run_dir / "config.json").write_text(json.dumps({
        "args": vars(args), "dqn": asdict(dqn_config), **checkpoint_extra
    }, indent=2))

    train_log = open(run_dir / "train_episodes.csv", "w", newline="")
    train_writer = csv.writer(train_log)
    train_writer.writerow([
        "episode", "env_steps", "scenario_seed", "reward", "collisions", "near_misses",
        "completed", "interventions", "budget_used_s", "epsilon", "loss", "q_mean", "seconds",
    ])
    eval_log = open(run_dir / "eval_history.csv", "w", newline="")
    eval_writer = None

    best_score = -np.inf
    episode = 0
    start = time.time()
    try:
        next_eval = args.eval_every
        while agent.env_steps < args.total_steps:
            observation, info = pool.reset()
            env = pool.current
            x = encoder.encode(observation, info["fleet_state"], env.vessel_ids)
            mask = info["action_mask"]
            ep = dict(reward=0.0, collisions=0, near_misses=0, completed=0, interventions=0)
            losses, qs = [], []

            while True:
                action = agent.act(x, mask, explore=True)
                observation, reward, terminated, truncated, info = env.step(action)
                x_next = encoder.encode(observation, info["fleet_state"], env.vessel_ids)
                mask_next = info["action_mask"]
                stats = agent.observe(x, mask, action, reward, x_next, mask_next, terminated, truncated)
                if stats:
                    losses.append(stats["loss"])
                    qs.append(stats["q_mean"])

                c = info["reward_components"]
                ep["reward"] += reward
                ep["collisions"] += c["collision_events"]
                ep["near_misses"] += c["near_miss_events"]
                ep["completed"] += c["completed_journeys"]
                ep["interventions"] += c["interventions"]
                x, mask = x_next, mask_next
                if terminated or truncated:
                    break

            episode += 1
            train_writer.writerow([
                episode, agent.env_steps, env.scenario_seed, f"{ep['reward']:.3f}", ep["collisions"],
                ep["near_misses"], ep["completed"], ep["interventions"],
                env.tracker.total_budget - env.tracker.remaining_budget, f"{agent.epsilon:.3f}",
                f"{np.mean(losses):.4f}" if losses else "", f"{np.mean(qs):.3f}" if qs else "",
                f"{time.time() - start:.1f}",
            ])
            train_log.flush()

            if episode % 10 == 0:
                print(
                    f"ep {episode:5d} | steps {agent.env_steps:7d} | eps {agent.epsilon:.2f} | "
                    f"R {ep['reward']:7.2f} | coll {ep['collisions']} | nm {ep['near_misses']:3d} | "
                    f"interv {ep['interventions']} | loss {np.mean(losses) if losses else float('nan'):.4f} | "
                    f"{time.time() - start:6.0f}s"
                )

            if agent.env_steps >= next_eval or agent.env_steps >= args.total_steps:
                next_eval += args.eval_every
                summary = summarise(evaluate(eval_envs, DQNPolicy(agent, encoder)))
                summary.update(env_steps=agent.env_steps, episode=episode)
                if eval_writer is None:
                    eval_writer = csv.DictWriter(eval_log, fieldnames=list(summary))
                    eval_writer.writeheader()
                eval_writer.writerow(summary)
                eval_log.flush()
                print(
                    f"  [eval @ {agent.env_steps}] reward {summary['reward_mean']:.2f} | "
                    f"collisions {summary['collision_events_mean']:.2f} | "
                    f"near-miss {summary['near_miss_events_mean']:.2f} | "
                    f"interventions {summary['interventions_mean']:.2f}"
                )
                if summary["reward_mean"] > best_score:
                    best_score = summary["reward_mean"]
                    agent.save(run_dir / "best.pt", extra={**checkpoint_extra, "eval": summary})
                    print(f"  saved best.pt (eval reward {best_score:.2f})")
    finally:
        agent.save(run_dir / "final.pt", extra=checkpoint_extra)
        train_log.close()
        eval_log.close()
        pool.close()
        for env in eval_envs:
            env.close()

    print(f"Done: {episode} episodes, {agent.env_steps} steps in {time.time() - start:.0f}s. Outputs in {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
