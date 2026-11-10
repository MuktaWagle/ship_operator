"""Train the action-masked PPO operator allocator

Mirrors `training/train_allocation.py` (same scenarios, observation encoder,
reward weights and held-out evaluation) so PPO and DQN are directly
comparable.  Outputs go to `runs/<run-name>/` (config.json,
train_episodes.csv, eval_history.csv, best.pt, final.pt).

Command:
    python -m evaluations.train_ppo --run-name ppo_s0 --seed 0
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

from evaluations.ppo import PPOAgent, PPOConfig, PPOPolicy, compute_gae  
from training.evaluation import evaluate, summarise  
from training.features import ObservationEncoder  
from training.scenarios import EnvConfig, ScenarioPool, build_heldout_envs  
from training.synthetic_scenarios import DEFAULT_SPEC  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default=None)
    p.add_argument("--fleet-size", type=int, default=10)
    p.add_argument("--min-start-distance", type=float, default=800.0)
    p.add_argument("--scenario-source", choices=("real", "synthetic", "mixed"), default="real",
                   help="real = BC journeys from the data; synthetic = generated crossing routes; "
                        "mixed = alternate by seed (see training/synthetic_scenarios.py)")
    p.add_argument("--synthetic-spec", default=DEFAULT_SPEC, metavar="S:O[,S:O...]",
                   help="synthetic difficulty as SPREAD_S:OFFSET_M pairs that take turns; smaller = denser conflicts")
    p.add_argument("--maneuver", choices=("right", "left", "stop"), default="right")
    p.add_argument("--controller", choices=("bc", "rule", "vo"), default="bc",
                   help="bc = fixed --maneuver; rule / vo = Task 5 planning operator")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--train-scenarios", type=int, default=12)
    p.add_argument("--eval-scenarios", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=60_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-risk-features", action="store_true")
    
    p.add_argument("--budget", type=float, default=300.0)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--w-collision", type=float, default=10.0)
    p.add_argument("--w-near-miss", type=float, default=5.0)
    p.add_argument("--w-intervention", type=float, default=1.0)
    p.add_argument("--w-completed", type=float, default=1.0)
    
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_name = args.run_name or f"ppo_n{args.fleet_size}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
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
        controller=args.controller,
        scenario_source=args.scenario_source,
        synthetic_spec=args.synthetic_spec,
    )
    print(f"Building up to {args.train_scenarios} training scenarios (fleet={args.fleet_size})...")
    pool = ScenarioPool(env_config, args.train_scenarios, start_seed=args.seed * 1000, rng_seed=args.seed)
    print(f"Building up to {args.eval_scenarios} held-out evaluation scenarios...")
    eval_envs, eval_split = build_heldout_envs(env_config, pool.ship_sets, args.eval_scenarios)

    encoder = ObservationEncoder.for_env(pool.envs[0], risk_features=not args.no_risk_features)
    config = PPOConfig(
        gamma=args.gamma, lam=args.lam, lr=args.lr, clip=args.clip, epochs=args.epochs,
        minibatch_size=args.minibatch_size, rollout_steps=args.rollout_steps,
        ent_coef=args.ent_coef, vf_coef=args.vf_coef, hidden=tuple(args.hidden), seed=args.seed,
    )
    agent = PPOAgent(encoder.dim, pool.num_vessels + 1, config)

    extra = {
        "algorithm": "ppo",
        "env_config": env_config.to_dict(),
        "encoder": {"risk_features": encoder.risk_features, "risk_horizon_s": encoder.risk_horizon_s},
        "train_seeds": [env.scenario_seed for env in pool.envs],
        "train_ship_sets": pool.ship_sets,
        "eval_split": eval_split,
        "eval_seeds": [env.scenario_seed for env in eval_envs],
        "eval_ship_sets": [sorted(env.ship_set) for env in eval_envs],
    }
    (run_dir / "config.json").write_text(json.dumps({"args": vars(args), "ppo": asdict(config), **extra}, indent=2))

    train_log = open(run_dir / "train_episodes.csv", "w", newline="")
    train_writer = csv.writer(train_log)
    train_writer.writerow([
        "episode", "env_steps", "scenario_seed", "reward", "collisions", "near_misses",
        "completed", "interventions", "budget_used_s", "seconds",
    ])
    eval_log = open(run_dir / "eval_history.csv", "w", newline="")
    eval_writer = None

    def encode(observation, info, env):
        return encoder.encode(observation, info["fleet_state"], env.vessel_ids)

    best_score, episode, start = -np.inf, 0, time.time()
    next_eval = args.eval_every
    env, x, mask, ep = None, None, None, None
    try:
        while agent.env_steps < args.total_steps:
            obs_buf, mask_buf, act_buf, logp_buf, val_buf, rew_buf, end_buf = [], [], [], [], [], [], []

            for _ in range(config.rollout_steps):
                if env is None:
                    observation, info = pool.reset()
                    env = pool.current
                    x, mask = encode(observation, info, env), info["action_mask"]
                    ep = dict(reward=0.0, collisions=0, near_misses=0, completed=0, interventions=0)

                action, logp, value = agent.step(x, mask)
                observation, reward, terminated, truncated, info = env.step(action)
                x_next, mask_next = encode(observation, info, env), info["action_mask"]

                shaped = reward
                if truncated and not terminated:
                    shaped += config.gamma * agent.value(x_next)
                end = bool(terminated or truncated)

                obs_buf.append(x); mask_buf.append(mask); act_buf.append(action)
                logp_buf.append(logp); val_buf.append(value); rew_buf.append(shaped); end_buf.append(float(end))

                c = info["reward_components"]
                ep["reward"] += reward
                ep["collisions"] += c["collision_events"]
                ep["near_misses"] += c["near_miss_events"]
                ep["completed"] += c["completed_journeys"]
                ep["interventions"] += c["interventions"]
                agent.env_steps += 1
                x, mask = x_next, mask_next

                if end:
                    episode += 1
                    train_writer.writerow([
                        episode, agent.env_steps, env.scenario_seed, f"{ep['reward']:.3f}",
                        ep["collisions"], ep["near_misses"], ep["completed"], ep["interventions"],
                        env.tracker.total_budget - env.tracker.remaining_budget,
                        f"{time.time() - start:.1f}",
                    ])
                    train_log.flush()
                    env = None

            last_value = 0.0 if env is None else agent.value(x)
            advantages, returns = compute_gae(rew_buf, val_buf, end_buf, last_value, config.gamma, config.lam)
            stats = agent.update({
                "obs": np.asarray(obs_buf, dtype=np.float32),
                "masks": np.asarray(mask_buf, dtype=bool),
                "actions": np.asarray(act_buf, dtype=np.int64),
                "logp": np.asarray(logp_buf, dtype=np.float32),
                "advantages": advantages,
                "returns": returns,
            })
            print(
                f"update {agent.updates:4d} | steps {agent.env_steps:7d} | episodes {episode:4d} | "
                f"pi {stats['policy_loss']:.4f} | v {stats['value_loss']:.4f} | "
                f"ent {stats['entropy']:.3f} | kl {stats['approx_kl']:.4f} | {time.time() - start:6.0f}s"
            )

            if agent.env_steps >= next_eval or agent.env_steps >= args.total_steps:
                next_eval += args.eval_every
                summary = summarise(evaluate(eval_envs, PPOPolicy(agent, encoder)))
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
                    agent.save(run_dir / "best.pt", extra={**extra, "eval": summary})
                    print(f"  saved best.pt (eval reward {best_score:.2f})")
    finally:
        agent.save(run_dir / "final.pt", extra=extra)
        train_log.close()
        eval_log.close()
        pool.close()
        for e in eval_envs:
            e.close()

    print(f"Done: {episode} episodes, {agent.env_steps} steps in {time.time() - start:.0f}s. Outputs in {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
