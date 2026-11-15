"""Compare allocation policies on identical, unseen scenarios and compile results

Policies:
    idle                                      (from training/baselines.py)
    greedy_highest_risk                       (dispatch to highest predicted risk)
    dqn, ppo                                  (one or more checkpoints each)
    mcts                                      (simulator-based planning, optional)

Outputs go to `evaluations/results/<name>/`:
    summary.json, summary.md, summary_by_checkpoint.md, episodes.csv,
    paired_vs_idle.json, latency.csv, latency.json and the plots
    comparison.png, scenario_reward_heatmap.png, paired_reward_gain.png,
    safety_vs_completion.png, latency.png.

Command to run:
    python -m evaluations.run_benchmark \
        --dqn runs/dqn_gate/best.pt runs/dqn_no_gate/best.pt \
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

# ----------------------------------------------------------------------------- plot theme
NAVY = "#0B2545"
NAVY_SOFT = "#3A4F6E"
MUTED = "#6B7A90"
GRID = "#E3E8EF"
PANEL = "#F7F9FC"
ORANGE = "#EE7A12"

POLICY_DISPLAY_NAMES = {
    "idle": "Idle",
    "greedy_highest_risk": "Greedy Highest Risk",
    "mcts": "MCTS",
}

POLICY_COLOURS = {
    "idle": "#9AA6B6",
    "greedy_highest_risk": "#F2C14E",
    "dqn_no_gate": "#1B3B6F",
    "dqn_gate": "#EE7A12",
    "ppo": "#4C7CC4",
    "mcts": "#A8380A",
}

EXTRA_COLOURS = ["#2E5EAA", "#F5A04A", "#0F2A55", "#C95A10", "#7FA3D1", "#FFB870"]

LATENCY_EXCLUDE = {"idle"}


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


# ----------------------------------------------------------------------------- naming
def checkpoint_kind(group, extra):
    
    if group == "dqn":
        return "dqn_gate" if (extra or {}).get("benefit") else "dqn_no_gate"
    return group


CHECKPOINT_LABELS = {"dqn_gate": "DQN_gate", "dqn_no_gate": "DQN_no_gate", "ppo": "PPO"}


def checkpoint_display_names(loaded):
    
    names, seen = {}, {}
    for label, group, _, extra in loaded:
        kind = checkpoint_kind(group, extra)
        base = CHECKPOINT_LABELS.get(kind, group.upper())
        seen[base] = seen.get(base, 0) + 1
        names[label] = base if seen[base] == 1 else f"{base} {seen[base]}"
    return names


def build_series(policies, loaded):
    
    ckpt_names = checkpoint_display_names(loaded)
    ckpt_kinds = {label: checkpoint_kind(group, extra) for label, group, _, extra in loaded}
    used, extra_colours, series = set(), iter(EXTRA_COLOURS), {}
    for label, (group, _) in policies.items():
        kind = ckpt_kinds.get(label, group)
        colour = POLICY_COLOURS.get(kind)
        if colour is None or kind in used:
            colour = next(extra_colours, NAVY_SOFT)
        used.add(kind)
        series[label] = {
            "name": ckpt_names.get(label, POLICY_DISPLAY_NAMES.get(label, label)),
            "colour": colour, "group": group, "kind": kind,
        }
    return series


# ----------------------------------------------------------------------------- timing
class TimedPolicy:
    

    def __init__(self, policy):
        self.policy = policy
        self.latencies = []

    def __call__(self, observation, info, env):
        t0 = time.perf_counter()
        action = self.policy(observation, info, env)
        self.latencies.append(time.perf_counter() - t0)
        return action

    def __getattr__(self, name):  # behave like the wrapped policy for anything else
        if name == "policy":
            raise AttributeError(name)
        return getattr(self.policy, name)


def latency_summary(lat):
    wall = np.asarray(lat["episode_wall_s"], dtype=float)
    pol = np.asarray(lat["episode_policy_s"], dtype=float)
    ms = np.asarray(lat["decision_ms"], dtype=float)
    sd = lambda a: float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {
        "episodes": int(len(wall)),
        "episode_wall_s_mean": float(wall.mean()) if len(wall) else float("nan"),
        "episode_wall_s_std": sd(wall),
        "episode_policy_s_mean": float(pol.mean()) if len(pol) else float("nan"),
        "episode_policy_s_std": sd(pol),
        "decisions": int(len(ms)),
        "decision_ms_mean": float(ms.mean()) if len(ms) else float("nan"),
        "decision_ms_median": float(np.median(ms)) if len(ms) else float("nan"),
        "decision_ms_p95": float(np.percentile(ms, 95)) if len(ms) else float("nan"),
        "decision_ms_max": float(ms.max()) if len(ms) else float("nan"),
        "total_s": float(lat["total_s"]),
    }


# ----------------------------------------------------------------------------- tables
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


# ----------------------------------------------------------------------------- plot helpers
def _rc():
    return {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.labelcolor": NAVY_SOFT,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "xtick.color": NAVY_SOFT,
        "ytick.color": NAVY_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    }


def _header(fig, title, subtitle=None):
    
    from matplotlib.lines import Line2D
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.20 / h, title, ha="left", va="top", fontsize=16, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.012, 1 - 0.55 / h, subtitle, ha="left", va="top", fontsize=9.5, color=MUTED)
    y = 1 - 0.86 / h
    fig.add_artist(Line2D([0.012, 0.07], [y, y], transform=fig.transFigure,
                          color=ORANGE, lw=3.2, solid_capstyle="round"))
    return 1 - 0.98 / h


def _legend(fig, names, colours, top):
    
    from matplotlib.patches import Patch
    h = fig.get_figheight()
    handles = [Patch(facecolor=c, edgecolor="white", label=n) for n, c in zip(names, colours)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, top),
               ncol=min(len(names), 6), fontsize=9.5, handlelength=1.3, handleheight=1.0,
               columnspacing=1.6, labelcolor=NAVY_SOFT)
    rows = int(np.ceil(len(names) / 6))
    return top - (0.32 * rows + 0.12) / h


def _wrap(name):
    if " (" in name:
        return name.replace(" (", "\n(", 1)
    words = name.split()
    if len(name) > 12 and len(words) > 1:
        cut = len(words) // 2
        return " ".join(words[:cut]) + "\n" + " ".join(words[cut:])
    return name


def _fmt(v):
    if not np.isfinite(v):
        return "n/a"
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"


def _mean_by_seed(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r["scenario_seed"], []).append(float(r[key]))
    return {s: float(np.mean(v)) for s, v in out.items()}


def save_plots(rows_by_label, summaries, paired, latency, series, out_dir):
    """Create the benchmark plots; one series per policy / checkpoint."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter
        from matplotlib.transforms import blended_transform_factory
    except ImportError:
        print("matplotlib not installed; skipping benchmark plots (pip install matplotlib)")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [l for l in series if l in summaries]
    names = [series[l]["name"] for l in labels]
    colours = [series[l]["colour"] for l in labels]
    n_eps = max((len(rows_by_label[l]) for l in labels), default=0)
    rng = np.random.default_rng(0)

    def stat(label, key, part):
        return float(summaries[label].get(f"{key}_{part}", np.nan))

    with plt.rc_context(_rc()):
        
        metrics = [
            ("reward", "Reward", True),
            ("collision_events", "Collision events", False),
            ("near_miss_events", "Near-miss events", False),
            ("completed_journeys", "Completed journeys", True),
            ("interventions", "Interventions", False),
            ("invalid_actions", "Invalid actions", False),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.4))
        top = _header(fig, "Policy performance overview",
                      f"Mean per episode ± episode standard deviation · {n_eps} episodes per policy on identical unseen scenarios")
        top = _legend(fig, names, colours, top)
        x = np.arange(len(labels))
        for ax, (key, title, higher_better) in zip(axes.flat, metrics):
            means = np.array([stat(l, key, "mean") for l in labels])
            stds = np.nan_to_num(np.array([stat(l, key, "std") for l in labels]))
            ax.bar(x, np.nan_to_num(means), width=0.7, color=colours, edgecolor="white", linewidth=2,
                   yerr=stds, error_kw=dict(ecolor=NAVY_SOFT, elinewidth=1.1, capsize=4, capthick=1.1),
                   zorder=3)
            ax.axhline(0, color=NAVY, lw=1, zorder=4)
            finite = np.where(np.isfinite(means))[0]
            best = None
            if len(finite):
                best = finite[np.argmax(means[finite])] if higher_better else finite[np.argmin(means[finite])]
            span = np.nanmax(np.abs(means) + stds) if len(finite) else 1.0
            span = span if span > 0 else 1.0
            for i, (m, s) in enumerate(zip(means, stds)):
                if not np.isfinite(m):
                    continue
                y = m + s + 0.04 * span if m >= 0 else m - s - 0.04 * span
                ax.text(i, y, _fmt(m), ha="center", va="bottom" if m >= 0 else "top", fontsize=8.5,
                        color=NAVY if i == best else NAVY_SOFT, fontweight="bold" if i == best else "normal")
            ax.set_title(title)
            ax.text(1.0, 1.035, "↑ higher is better" if higher_better else "↓ lower is better",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=ORANGE)
            ax.set_xticks([])
            ax.grid(axis="x", visible=False)
            ax.margins(y=0.18)
            if len(finite) and np.allclose(means[finite], 0) and np.allclose(stds, 0):
                ax.set_ylim(-0.05, 1)  
        fig.tight_layout(rect=(0, 0, 1, top), h_pad=2.2, w_pad=2.0)
        fig.savefig(out_dir / "comparison.png", dpi=170)
        plt.close(fig)

        # Reward per scenario ----------------------------------------------------------
        seeds = sorted({r["scenario_seed"] for l in labels for r in rows_by_label[l]})
        matrix = np.full((len(labels), len(seeds)), np.nan)
        for i, label in enumerate(labels):
            lookup = _mean_by_seed(rows_by_label[label], "reward")
            for j, seed in enumerate(seeds):
                matrix[i, j] = lookup.get(seed, np.nan)
        cmap = LinearSegmentedColormap.from_list("orange_navy", [ORANGE, "#FBE3CC", "#EEF1F5", "#C9D6EA", NAVY])
        cmap.set_bad("#FFFFFF")
        lo, hi = np.nanmin(matrix), np.nanmax(matrix)
        mid = np.nanmedian(matrix)
        norm = TwoSlopeNorm(vmin=lo, vcenter=mid, vmax=hi) if lo < mid < hi else Normalize(lo, hi if hi > lo else lo + 1)
        fig, ax = plt.subplots(figsize=(max(9, 0.85 * len(seeds) + 4.5), max(4.5, 0.62 * len(labels) + 2.2)))
        top = _header(fig, "Reward by policy and evaluation scenario",
                      "Navy = higher reward, orange = lower reward · spot scenario-specific failures")
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        if len(seeds) <= 24:
            for i in range(len(labels)):
                for j in range(len(seeds)):
                    v = matrix[i, j]
                    if np.isfinite(v):
                        dark = abs(norm(v) - 0.5) > 0.32
                        ax.text(j, i, _fmt(v), ha="center", va="center", fontsize=8,
                                color="white" if dark else NAVY)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(names)
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(s) for s in seeds], rotation=45, ha="right")
        ax.set_xticks(np.arange(-0.5, len(seeds)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Scenario seed")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.set_label("Episode reward", color=NAVY_SOFT)
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.savefig(out_dir / "scenario_reward_heatmap.png", dpi=170)
        plt.close(fig)

        # Paired gain versus idle ------------------------------------------------------
        paired_labels = [l for l in labels if l in paired]
        if paired_labels:
            means = np.array([paired[l]["gain_mean"] for l in paired_labels])
            errors = np.array([paired[l]["gain_std"] for l in paired_labels])
            ypos = np.arange(len(paired_labels))[::-1]
            fig, ax = plt.subplots(figsize=(12, max(4.2, 0.75 * len(paired_labels) + 2.2)))
            top = _header(fig, "Paired reward gain over idle",
                          "Reward difference vs. idle on the same scenario seeds · mean ± SD · win rate = share of scenarios beating idle")
            ax.barh(ypos, means, height=0.6, color=[series[l]["colour"] for l in paired_labels],
                    edgecolor="white", linewidth=2, xerr=errors,
                    error_kw=dict(ecolor=NAVY_SOFT, elinewidth=1.1, capsize=4, capthick=1.1), zorder=3)
            ax.axvline(0, color=NAVY, lw=1.2, zorder=4)
            span = float(np.max(np.abs(means) + errors)) or 1.0
            for y, l, m, e in zip(ypos, paired_labels, means, errors):
                p = paired[l]
                text = f"{m:+.2f}   ·   wins {p['win_rate']:.0%} of {p['scenarios']}"
                ax.text(m + np.sign(m or 1) * (e + 0.03 * span), y, text, va="center",
                        ha="left" if m >= 0 else "right", fontsize=9, color=NAVY)
            ax.set_yticks(ypos)
            ax.set_yticklabels([series[l]["name"] for l in paired_labels], fontsize=10)
            ax.set_xlim(-1.45 * span, 1.45 * span)
            ax.set_xlabel("Reward difference vs idle (higher is better)")
            ax.grid(axis="y", visible=False)
            fig.tight_layout(rect=(0, 0, 1, top))
            fig.savefig(out_dir / "paired_reward_gain.png", dpi=170)
            plt.close(fig)

        # Safety-productivity trade-off ---------------------------------------------
        fig, ax = plt.subplots(figsize=(10.5, 7.2))
        top = _header(fig, "Safety versus mission completion",
                      "Mean per episode · whiskers show ± standard error of the mean · top-left is best")
        pts = []
        for label in labels:
            risk = stat(label, "collision_events", "mean") + stat(label, "near_miss_events", "mean")
            risk_sd = np.hypot(np.nan_to_num(stat(label, "collision_events", "std")),
                               np.nan_to_num(stat(label, "near_miss_events", "std")))
            comp = stat(label, "completed_journeys", "mean")
            comp_sd = np.nan_to_num(stat(label, "completed_journeys", "std"))
            n = max(1, len(rows_by_label[label]))
            risk_sd, comp_sd = risk_sd / np.sqrt(n), comp_sd / np.sqrt(n)  # standard error of the mean
            c = series[label]["colour"]
            ax.errorbar(risk, comp, xerr=risk_sd, yerr=comp_sd, fmt="none", ecolor=c, alpha=0.6,
                        elinewidth=1.6, capsize=4, zorder=2)
            ax.scatter(risk, comp, s=230, color=c, edgecolor="white", linewidth=2.2, zorder=4)
            pts.append((risk, comp, series[label]["name"], c))
        
        xs = np.array([q[0] for q in pts], dtype=float)
        ys = np.array([q[1] for q in pts], dtype=float)
        xr = (np.nanmax(xs) - np.nanmin(xs)) or 1.0
        yr = (np.nanmax(ys) - np.nanmin(ys)) or 1.0
        corners = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        placed = []
        for k, (rx, cy, name, c) in enumerate(pts):
            others = [((xs[j] - rx) / xr, (ys[j] - cy) / yr) for j in range(len(pts)) if j != k]
            others += [((px - rx) / xr, (py - cy) / yr) for px, py in placed]
            def clearance(corner):
                cx, cyy = corner[0] * 0.12, corner[1] * 0.08
                return min((np.hypot(cx - ox, cyy - oy) for ox, oy in others), default=1.0)
            sx, sy = max(corners, key=clearance)
            placed.append((rx + sx * 0.12 * xr, cy + sy * 0.08 * yr))
            ax.annotate(name, (rx, cy), xytext=(14 * sx, 16 * sy), textcoords="offset points",
                        fontsize=9, color=NAVY, va="center", ha="left" if sx > 0 else "right",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, lw=1.2),
                        arrowprops=dict(arrowstyle="-", color=c, lw=1), zorder=5)
        ax.annotate("better", xy=(0.03, 0.97), xytext=(0.15, 0.85), xycoords="axes fraction",
                    textcoords="axes fraction", fontsize=10, color=ORANGE, fontweight="bold",
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2, shrinkA=2, shrinkB=0))
        ax.set_xlabel("Collision + near-miss events per episode (lower is safer)")
        ax.set_ylabel("Completed journeys per episode (higher is better)")
        ax.margins(x=0.18, y=0.18)
        fig.tight_layout(rect=(0, 0, 1, top))
        fig.savefig(out_dir / "safety_vs_completion.png", dpi=170)
        plt.close(fig)

        # Latency to complete the simulation ------------------------------------------
        lat_labels = [l for l in labels if l in latency and series[l]["group"] not in LATENCY_EXCLUDE]
        if lat_labels:
            summ = {l: latency_summary(latency[l]) for l in lat_labels}
            lat_names = [series[l]["name"] for l in lat_labels]
            lat_cols = [series[l]["colour"] for l in lat_labels]
            ypos = np.arange(len(lat_labels))[::-1]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, max(5.2, 0.8 * len(lat_labels) + 2.6)),
                                           gridspec_kw=dict(width_ratios=[1.15, 1]))
            top = _header(fig, "Latency: time to complete a simulation",
                          f"Wall-clock time on this machine · {summ[lat_labels[0]]['episodes']} episodes per policy · same scenarios for every policy")

            wall = np.array([summ[l]["episode_wall_s_mean"] for l in lat_labels])
            wall_sd = np.array([summ[l]["episode_wall_s_std"] for l in lat_labels])
            pol = np.array([summ[l]["episode_policy_s_mean"] for l in lat_labels])
            sim = np.clip(wall - pol, 0.0, None)
            ax1.barh(ypos, pol, height=0.6, color=lat_cols, edgecolor="white", linewidth=2, zorder=3)
            ax1.barh(ypos, sim, left=pol, height=0.6, color=lat_cols, alpha=0.28, hatch="///",
                     edgecolor="white", linewidth=2, zorder=3)
            ax1.errorbar(wall, ypos, xerr=wall_sd, fmt="none", ecolor=NAVY_SOFT, elinewidth=1.1,
                         capsize=4, capthick=1.1, zorder=4)
            xmax = float(np.max(wall + wall_sd)) or 1.0
            for y, w, s, p in zip(ypos, wall, wall_sd, pol):
                share = p / w if w > 0 else 0.0
                ax1.text(w + s + 0.02 * xmax, y, f"{w:.3g} s  ({share:.0%} deciding)", va="center",
                         fontsize=9, color=NAVY)
            ax1.set_xlim(0, xmax * 1.45)
            ax1.set_yticks(ypos)
            ax1.set_yticklabels(lat_names, fontsize=10)
            ax1.set_xlabel("Seconds per episode (mean ± SD)")
            ax1.set_title("Wall-clock time per episode")
            ax1.grid(axis="y", visible=False)
            ax1.legend(handles=[
                Patch(facecolor=NAVY_SOFT, edgecolor="white", label="policy decision time"),
                Patch(facecolor=NAVY_SOFT, alpha=0.28, hatch="///", edgecolor="white",
                      label="environment simulation"),
            ], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9, labelcolor=NAVY_SOFT)

            data = [np.clip(np.asarray(latency[l]["decision_ms"], dtype=float), 1e-4, None) for l in lat_labels]
            keep = [i for i, d in enumerate(data) if len(d)]
            if keep:
                box_kw = dict(positions=ypos[keep], widths=0.55, patch_artist=True, showfliers=False,
                              whis=(5, 95), medianprops=dict(color=NAVY, lw=2),
                              whiskerprops=dict(color=NAVY_SOFT, lw=1.1),
                              capprops=dict(color=NAVY_SOFT, lw=1.1), boxprops=dict(lw=1.4))
                try:  # matplotlib >= 3.10
                    bp = ax2.boxplot([data[i] for i in keep], orientation="horizontal", **box_kw)
                except TypeError:
                    bp = ax2.boxplot([data[i] for i in keep], vert=False, **box_kw)
                for patch, i in zip(bp["boxes"], keep):
                    patch.set_facecolor(lat_cols[i])
                    patch.set_alpha(0.35)
                    patch.set_edgecolor(lat_cols[i])
                for i in keep:
                    d = data[i]
                    sample = d if len(d) <= 400 else rng.choice(d, 400, replace=False)
                    ax2.scatter(sample, ypos[i] + rng.uniform(-0.17, 0.17, len(sample)), s=7,
                                color=lat_cols[i], alpha=0.35, linewidth=0, zorder=4)
            ax2.set_xscale("log")
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            trans = blended_transform_factory(ax2.transAxes, ax2.transData)
            for i, l in enumerate(lat_labels):
                s = summ[l]
                ax2.text(1.02, ypos[i], f"median {s['decision_ms_median']:.3g} ms\np95 {s['decision_ms_p95']:.3g} ms",
                         transform=trans, va="center", ha="left", fontsize=8.5, color=NAVY, linespacing=1.3)
            ax2.set_yticks(ypos)
            ax2.set_yticklabels([])
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_xlabel("Milliseconds per decision (log scale · whiskers 5th–95th pct)")
            ax2.set_title("Per-decision latency")
            ax2.grid(axis="y", visible=False)
            ax2.grid(axis="x", which="minor", color=GRID, linewidth=0.5)
            fig.tight_layout(rect=(0, 0, 0.93, top), w_pad=3)
            fig.savefig(out_dir / "latency.png", dpi=170)
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

    policies = {}  
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

    series = build_series(policies, loaded)

    rows, rows_by_group, rows_by_label, latency = [], {}, {}, {}
    try:
        for label, (group, policy) in policies.items():
            timed = TimedPolicy(policy)
            lat = {"episode_wall_s": [], "episode_policy_s": [], "decision_ms": [], "total_s": 0.0}
            results = []
            t0 = time.perf_counter()
            
            for env in eval_envs:
                first_call = len(timed.latencies)
                e0 = time.perf_counter()
                env_results = list(evaluate([env], timed))
                wall = time.perf_counter() - e0
                calls = timed.latencies[first_call:]
                k = max(1, len(env_results))
                lat["episode_wall_s"] += [wall / k] * k
                lat["episode_policy_s"] += [sum(calls) / k] * k
                results.extend(env_results)
            lat["total_s"] = time.perf_counter() - t0
            lat["decision_ms"] = [1000.0 * t for t in timed.latencies]
            latency[label] = lat
            print(f"  {series[label]['name']:<32} {len(results)} episodes in {lat['total_s']:6.1f}s")
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
    paired_by_label = paired_vs_idle(rows_by_label)
    latency_by_label = {label: latency_summary(lat) for label, lat in latency.items()}

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
    print("\nLatency (wall-clock):")
    for label, s in latency_by_label.items():
        print(f"  {series[label]['name']:<32} {s['episode_wall_s_mean']:8.3f} s/episode | "
              f"decision median {s['decision_ms_median']:.3g} ms, p95 {s['decision_ms_p95']:.3g} ms "
              f"| {s['decisions']} decisions")

    name = args.name or f"benchmark_n{config.fleet_size}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = PROJECT_ROOT / "evaluations" / "results" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    display = {label: s["name"] for label, s in series.items()}
    (out_dir / "summary.json").write_text(json.dumps({
        "fleet_size": config.fleet_size, "split": split, "scenario_seeds": eval_seeds,
        "env_config": eval_config.to_dict(), "by_group": by_group, "by_checkpoint": by_label,
        "display_names": display, "paired_vs_idle": paired, "paired_vs_idle_by_checkpoint": paired_by_label,
        "latency": latency_by_label,
    }, indent=2))
    (out_dir / "summary.md").write_text(markdown_table(by_group) + "\n")
    (out_dir / "summary_by_checkpoint.md").write_text(
        markdown_table({display.get(l, l): s for l, s in by_label.items()}) + "\n")
    (out_dir / "paired_vs_idle.json").write_text(json.dumps(paired, indent=2))
    (out_dir / "latency.json").write_text(json.dumps(latency_by_label, indent=2))
    with open(out_dir / "latency.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "display_name", "scenario_seed", "episode_wall_s", "episode_policy_s"])
        for label, lat in latency.items():
            seeds = [r["scenario_seed"] for r in rows_by_label.get(label, [])]
            for seed, w, p in zip(seeds, lat["episode_wall_s"], lat["episode_policy_s"]):
                writer.writerow([label, display.get(label, label), seed, f"{w:.6f}", f"{p:.6f}"])
    with open(out_dir / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_plots(rows_by_label, by_label, paired_by_label, latency, series, out_dir)
    print(f"\nSaved results in {out_dir}")
    return by_group


if __name__ == "__main__":
    main()
