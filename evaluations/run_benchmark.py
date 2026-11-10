"""Compare allocation policies on identical, unseen scenarios and compile results

Policies:
    idle                                      (from training/baselines.py)
    greedy_highest_risk                       (dispatch to highest predicted risk)
    dqn, ppo                                  (one checkpoint each)
    mcts                                      (simulator-based planning, optional)

Outputs go to `evaluations/results/<name>/`:
    summary.json, summary.md, summary_by_checkpoint.md, episodes.csv,
    calibration.csv, paired_vs_idle.json and comparison.png.

Command to run:
    python -m evaluations.run_benchmark \
        --dqn runs/dqn_s0/best.pt \
        --ppo runs/ppo_s0/best.pt \
        --mcts \
        --scenarios 8
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluations.policies import HighestRiskGreedyPolicy
from evaluations.ppo import PPOAgent, PPOPolicy  
from training.baselines import make_baselines  
from training.dqn import MaskedDQNAgent  
from training.evaluation import DQNPolicy, evaluate, format_table, summarise  
from training.features import ObservationEncoder  
from training.scenarios import EnvConfig, build_unique_envs  
from evaluations.mcts import ModelMCTSConfig, ModelMCTSPolicy  

TABLE_COLUMNS = [
    ("reward", "Reward"),
    ("collision_events", "Collisions"),
    ("near_miss_events", "Near-misses"),
    ("completed_journeys", "Completed"),
    ("interventions", "Interventions"),
    ("budget_used_s", "Budget used (s)"),
    ("min_pair_distance_m", "Min dist (m)"),
]
GROUP_ORDER = [
    "idle", "greedy_highest_risk", "dqn", "ppo", "mcts",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dqn", nargs="*", default=[], help="DQN checkpoints (.pt)")
    p.add_argument("--ppo", nargs="*", default=[], help="PPO checkpoints (.pt)")
    p.add_argument("--mcts", action="store_true", help="include the model-based MCTS planner")
    p.add_argument("--baselines", nargs="*", choices=("idle", "greedy_highest_risk"),
                   default=["idle", "greedy_highest_risk"],
                   help="choose baseline policies (default: idle greedy_highest_risk)")
    p.add_argument("--scenarios", type=int, default=6, help="number of unseen evaluation scenarios")
    p.add_argument("--scenario-source", choices=("real", "synthetic", "mixed"), default=None,
                   help="where scenarios come from; default: the checkpoint's own setting (real if none). "
                        "Use real on a synthetic-trained checkpoint for the realistic test set.")
    p.add_argument("--synthetic-spec", default=None, metavar="S:O[,S:O...]",
                   help="synthetic difficulty as SPREAD_S:OFFSET_M pairs; default: the checkpoint's own")
    p.add_argument("--seed-offset", type=int, default=20_000)
    p.add_argument("--exclude-scope", choices=("all", "train"), default="all",
                   help="which ship sets of the loaded checkpoints are kept out of the evaluation scenarios: "
                        "all = their training and held-out sets (default), train = training sets only")
    p.add_argument("--exclude-from", nargs="*", default=None, metavar="WORD",
                   help="only keep out the sets of checkpoints whose label contains one of these words "
                        "(default: every loaded checkpoint)")
    p.add_argument("--calibration-scenarios", type=int, default=4)
    p.add_argument("--calibration-seed-offset", type=int, default=40_000)
    p.add_argument("--split", choices=("train", "val", "test"), default=None,
                   help="default: test for fleets of <= 4 ships, else unseen train combinations")
    p.add_argument("--name", default=None, help="results folder name")
    # used only when no checkpoint supplies an environment config
    p.add_argument("--fleet-size", type=int, default=10)
    p.add_argument("--min-start-distance", type=float, default=800.0)
    p.add_argument("--maneuver", choices=("right", "left", "stop"), default="right")
    p.add_argument("--controller", choices=("bc", "rule", "vo"), default=None,
                   help="what an assisted vessel does: bc = fixed --maneuver, rule / vo = Task 5 planning "
                        "operator. Default: the checkpoint's own setting (bc if there is none). "
                        "Given explicitly, it overrides the checkpoint's setting for every policy.")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--budget", type=float, default=300.0)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--delay", type=float, default=0.0)
    # MCTS
    p.add_argument("--mcts-rollouts", type=int, default=150)
    p.add_argument("--mcts-horizon", type=int, default=40)
    p.add_argument("--mcts-depth", type=int, default=12)
    p.add_argument("--mcts-top-k", type=int, default=3)
    p.add_argument("--mcts-activation", type=float, default=1000.0)
    return p.parse_args(argv)


def load_checkpoints(args):
    """Return list of (label, group, agent, extra) for every checkpoint."""
    loaded = []
    for path in args.dqn:
        agent = MaskedDQNAgent.load(path)
        loaded.append((f"dqn:{Path(path).parent.name}/{Path(path).stem}", "dqn", agent, agent.extra))
    for path in args.ppo:
        agent = PPOAgent.load(path)
        loaded.append((f"ppo:{Path(path).parent.name}/{Path(path).stem}", "ppo", agent, agent.extra))
    return loaded


def base_env_config(args, loaded):
    controller = getattr(args, "controller", None)  # None = keep the checkpoint's own setting
    if loaded:
        config = EnvConfig(**loaded[0][3]["env_config"])
        for label, _, _, extra in loaded[1:]:
            other = EnvConfig(**extra["env_config"])
            if other.fleet_size != config.fleet_size:
                raise SystemExit(f"{label} was trained with fleet size {other.fleet_size}, expected {config.fleet_size}.")
        overrides = {
            key: value for key, value in (
                ("controller", controller),
                ("scenario_source", getattr(args, "scenario_source", None)),
                ("synthetic_spec", getattr(args, "synthetic_spec", None)),
            ) if value is not None
        }
        if overrides:
            config = EnvConfig(**{**config.to_dict(), **overrides})
        return config
    return EnvConfig(
        fleet_size=args.fleet_size, min_start_distance_m=args.min_start_distance,
        assistance_direction=args.maneuver, max_steps=args.max_steps,
        total_budget_s=args.budget, duration_s=args.duration, delay_s=args.delay,
        controller=controller or "bc",
        **{
            key: value for key, value in (
                ("scenario_source", getattr(args, "scenario_source", None)),
                ("synthetic_spec", getattr(args, "synthetic_spec", None)),
            ) if value is not None
        },
    )


def markdown_table(summaries):
    header = "| Policy | " + " | ".join(name for _, name in TABLE_COLUMNS) + " |"
    divider = "|---|" + "---|" * len(TABLE_COLUMNS)
    lines = [header, divider]
    for name, s in summaries.items():
        cells = [f"{s[key + '_mean']:.2f} ± {s[key + '_std']:.2f}" for key, _ in TABLE_COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_seed_reward(rows):
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["scenario_seed"], []).append(r["reward"])
    return {seed: float(np.mean(values)) for seed, values in by_seed.items()}


def paired_vs_idle(rows_by_group):
    if "idle" not in rows_by_group:
        return {}
    idle = per_seed_reward(rows_by_group["idle"])
    out = {}
    for group, rows in rows_by_group.items():
        if group == "idle":
            continue
        rewards = per_seed_reward(rows)
        diffs = np.array([rewards[s] - idle[s] for s in rewards if s in idle], dtype=float)
        if len(diffs) == 0:
            continue
        out[group] = {
            "scenarios": int(len(diffs)),
            "gain_mean": float(diffs.mean()),
            "gain_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "gain_se": float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan"),
            "win_rate": float((diffs > 0).mean()),
        }
    return out


def save_plots(rows_by_group, summaries, paired, out_dir):
    """Create complementary benchmark plots from episode-level results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping benchmark plots (pip install matplotlib)")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(summaries)
    labels = [n.replace("dqn:", "DQN\n").replace("ppo:", "PPO\n") for n in names]

    # 1. Six key metrics, mean +/- episode standard deviation.
    metrics = [
        ("reward", "Reward", True),
        ("collision_events", "Collision events", False),
        ("near_miss_events", "Near-miss events", False),
        ("completed_journeys", "Completed journeys", True),
        ("interventions", "Interventions", False),
        ("invalid_actions", "Invalid actions", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, higher_better) in zip(axes.flat, metrics):
        means = [summaries[n][key + "_mean"] for n in names]
        stds = [summaries[n][key + "_std"] for n in names]
        ax.bar(range(len(names)), means, yerr=stds, capsize=3)
        ax.set_title(title + (" (higher is better)" if higher_better else " (lower is better)"))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Policy performance overview — mean +/- episode SD")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=160)
    plt.close(fig)

    # 2. Episode distributions expose spread and outliers.
    dist_metrics = [
        ("reward", "Reward"), ("collision_events", "Collisions"),
        ("near_miss_events", "Near-misses"),
        ("completed_journeys", "Completed journeys"),
        ("interventions", "Interventions"),
        ("min_pair_distance_m", "Minimum pair distance (m)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.flat, dist_metrics):
        data = [[float(r[key]) for r in rows_by_group[n]
                 if np.isfinite(float(r[key]))] for n in names]
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Episode-level metric distributions")
    fig.tight_layout()
    fig.savefig(out_dir / "episode_distributions.png", dpi=160)
    plt.close(fig)

    # 3. Reward per scenario, useful for identifying scenario-specific failures.
    seeds = sorted({r["scenario_seed"] for rs in rows_by_group.values() for r in rs})
    matrix = np.full((len(names), len(seeds)), np.nan)
    for i, name in enumerate(names):
        lookup = {r["scenario_seed"]: r["reward"] for r in rows_by_group[name]}
        for j, seed in enumerate(seeds):
            if seed in lookup:
                matrix[i, j] = lookup[seed]
    fig, ax = plt.subplots(figsize=(max(8, .65 * len(seeds)), max(4, .55 * len(names))))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(seed) for seed in seeds], rotation=45, ha="right")
    ax.set_xlabel("Scenario seed")
    ax.set_title("Reward by policy and evaluation scenario")
    fig.colorbar(im, ax=ax, label="Episode reward")
    fig.tight_layout()
    fig.savefig(out_dir / "scenario_reward_heatmap.png", dpi=160)
    plt.close(fig)

    # 4. Paired gain versus idle, calculated on matching scenario seeds.
    if paired:
        paired_names = list(paired)
        means = [paired[n]["gain_mean"] for n in paired_names]
        errors = [paired[n]["gain_std"] for n in paired_names]
        fig, ax = plt.subplots(figsize=(max(8, .7 * len(paired_names)), 5))
        ax.bar(range(len(paired_names)), means, yerr=errors, capsize=4)
        ax.axhline(0, linewidth=1)
        ax.set_xticks(range(len(paired_names)))
        ax.set_xticklabels(paired_names, rotation=45, ha="right")
        ax.set_ylabel("Reward difference vs idle")
        ax.set_title("Paired reward gain over idle (mean +/- SD)")
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out_dir / "paired_reward_gain.png", dpi=160)
        plt.close(fig)

    # 5. Safety/productivity tradeoff.
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in names:
        summary = summaries[name]
        risk_events = summary["collision_events_mean"] + summary["near_miss_events_mean"]
        completions = summary["completed_journeys_mean"]
        ax.scatter(risk_events, completions, s=70)
        ax.annotate(name, (risk_events, completions), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("Collision + near-miss events per episode (lower is safer)")
    ax.set_ylabel("Completed journeys per episode (higher is better)")
    ax.set_title("Safety versus mission completion")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "safety_vs_completion.png", dpi=160)
    plt.close(fig)

    # 6. Productivity per intervention; NaN where no intervention occurred.
    efficiency = []
    for name in names:
        summary = summaries[name]
        interventions = summary["interventions_mean"]
        efficiency.append(summary["completed_journeys_mean"] / interventions
                         if interventions > 0 else np.nan)
    fig, ax = plt.subplots(figsize=(max(8, .7 * len(names)), 5))
    ax.bar(range(len(names)), efficiency)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Completed journeys per intervention")
    ax.set_title("Intervention efficiency (higher is better)")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "intervention_efficiency.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    loaded = load_checkpoints(args)
    config = base_env_config(args, loaded)
    split = args.split or ("test" if config.fleet_size <= 4 else "train")
    eval_config = EnvConfig(**{**config.to_dict(), "split": split})

    exclude = []
    if split == "train":
        for label, _, _, extra in loaded:
            if args.exclude_from and not any(word in label for word in args.exclude_from):
                continue
            exclude += extra.get("train_ship_sets", [])
            if args.exclude_scope == "all":
                exclude += extra.get("eval_ship_sets", [])

    print(f"Building {args.scenarios} evaluation scenarios (fleet={config.fleet_size}, split={split})...")
    eval_envs = build_unique_envs(eval_config, args.scenarios, args.seed_offset, exclude_sets=exclude)
    eval_sets = [sorted(e.ship_set) for e in eval_envs]
    eval_seeds = [e.scenario_seed for e in eval_envs]

    policies = {}  # label -> (group, policy)
    baselines = make_baselines()
    chosen = args.baselines
    for name in chosen:
        if name == "greedy_highest_risk":
            policies[name] = (name, HighestRiskGreedyPolicy())
        elif name in baselines:
            policies[name] = (name, baselines[name])
        else:
            raise SystemExit(f"Unknown baseline: {name}")

    for label, group, agent, extra in loaded:
        encoder = ObservationEncoder.for_env(eval_envs[0], **extra["encoder"])
        wrapper = DQNPolicy if group == "dqn" else PPOPolicy
        policies[label] = (group, wrapper(agent, encoder))

    if args.mcts:
        mcts_config = ModelMCTSConfig(
            rollouts=args.mcts_rollouts, horizon=args.mcts_horizon, tree_depth=args.mcts_depth,
            top_k=args.mcts_top_k, activation_m=args.mcts_activation,
        )
        policies["mcts"] = ("mcts", ModelMCTSPolicy(mcts_config))

    rows, rows_by_group, rows_by_label = [], {}, {}
    try:
        for label, (group, policy) in policies.items():
            t0 = time.time()
            results = evaluate(eval_envs, policy)
            print(f"  {label:<32} {len(results)} episodes in {time.time() - t0:6.1f}s")
            for r in results:
                r = {"policy": label, "group": group, **r}
                rows.append(r)
                rows_by_group.setdefault(group, []).append(r)
                rows_by_label.setdefault(label, []).append(r)
    finally:
        for env in eval_envs:
            env.close()

    ordered = [g for g in GROUP_ORDER if g in rows_by_group] + [g for g in rows_by_group if g not in GROUP_ORDER]
    by_group = {g: summarise(rows_by_group[g]) for g in ordered}
    by_label = {label: summarise(rs) for label, rs in rows_by_label.items()}
    paired = paired_vs_idle(rows_by_group)

    print(f"\nFleet {config.fleet_size} ships | split={split} | scenarios={eval_seeds}")
    print(format_table(by_group))
    if any(len(v) > 1 for v in (
        [r for r in rows_by_label if rows_by_label[r][0]["group"] == g] for g in ("dqn", "ppo")
    )):
        print("\nPer checkpoint:")
        print(format_table(by_label))
    if paired:
        print("\nPaired reward gain over idle (same scenarios):")
        for group, p in paired.items():
            print(f"  {group:<22} {p['gain_mean']:+7.2f} ± {p['gain_std']:.2f} (SD) | wins {p['win_rate']:.0%} of {p['scenarios']}")

    name = args.name or f"benchmark_n{config.fleet_size}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = PROJECT_ROOT / "evaluations" / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "fleet_size": config.fleet_size, "split": split, "scenario_seeds": eval_seeds,
        "env_config": eval_config.to_dict(), "by_group": by_group, "by_checkpoint": by_label,
        "paired_vs_idle": paired,
    }, indent=2))
    (out_dir / "summary.md").write_text(markdown_table(by_group) + "\n")
    (out_dir / "summary_by_checkpoint.md").write_text(markdown_table(by_label) + "\n")
    (out_dir / "paired_vs_idle.json").write_text(json.dumps(paired, indent=2))
    with open(out_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plots(rows_by_group, by_group, paired, out_dir)
    print(f"\nSaved results in {out_dir}")
    return by_group


if __name__ == "__main__":
    main()
