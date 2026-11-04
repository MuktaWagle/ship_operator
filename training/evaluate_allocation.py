"""Compare a trained DQN allocator against the reference policies.

By default the scenarios exclude every ship set used for training (and,
for fleets of <= 4 ships, come from the test split).

The environment is rebuilt from the config stored inside the checkpoint, so
the comparison uses exactly the fleet size, budget, delay and reward weights
the agent was trained with.  
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.baselines import make_baselines  
from training.dqn import MaskedDQNAgent  
from training.evaluation import DQNPolicy, evaluate, format_table, summarise  
from training.features import ObservationEncoder  
from training.scenarios import EnvConfig, build_unique_envs  

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--split", choices=("train", "val", "test"), default=None,
                   help="default: test for fleets of <= 4 ships, otherwise unseen train combinations")
    p.add_argument("--scenarios", type=int, default=4)
    p.add_argument("--seed-offset", type=int, default=20_000,
                   help="scenario seeds start here (kept away from training seeds)")
    p.add_argument("--baselines", nargs="*", default=None, help="subset of: idle random_valid greedy_nearest risk_threshold")
    args = p.parse_args(argv)

    checkpoint = Path(args.checkpoint)
    agent = MaskedDQNAgent.load(checkpoint)
    extra = agent.extra
    env_config = EnvConfig(**extra["env_config"])
    split = args.split or ("test" if env_config.fleet_size <= 4 else "train")
    env_config = EnvConfig(**{**env_config.to_dict(), "split": split})
    
    exclude = (extra.get("train_ship_sets", []) + extra.get("eval_ship_sets", [])) if split == "train" else []
    envs = build_unique_envs(env_config, args.scenarios, args.seed_offset, exclude_sets=exclude)
    seeds = [env.scenario_seed for env in envs]
    encoder = ObservationEncoder.for_env(envs[0], **extra["encoder"])

    policies = {"dqn": DQNPolicy(agent, encoder)}
    baselines = make_baselines()
    for name in (args.baselines if args.baselines is not None else baselines):
        policies[name] = baselines[name]

    summaries, rows = {}, []
    try:
        for name, policy in policies.items():
            results = evaluate(envs, policy)
            summaries[name] = summarise(results)
            rows += [{"policy": name, **r} for r in results]
    finally:
        for env in envs:
            env.close()

    print(f"\nFleet {env_config.fleet_size} ships | split={split} | scenarios={seeds}")
    print(format_table(summaries))

    out_json = checkpoint.parent / f"eval_{split}.json"
    out_json.write_text(json.dumps({"split": split, "seeds": seeds, "summaries": summaries}, indent=2))
    out_csv = checkpoint.parent / f"eval_{split}_episodes.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out_json.name} and {out_csv.name} in {checkpoint.parent}")
    return summaries


if __name__ == "__main__":
    main()
