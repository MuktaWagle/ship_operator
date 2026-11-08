"""
Action-masked PPO for the operator allocation environment 

Command for training:
    python -m evaluations.train_ppo --run-name ppo_s0 --seed 0
    python -m evaluations.train_ppo --run-name ppo_s1 --seed 1

"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

MASK_FILL = -1e9


def _mlp(in_dim: int, out_dim: int, hidden: Sequence[int], out_gain: float) -> nn.Sequential:
    layers = []
    d = in_dim
    for width in hidden:
        linear = nn.Linear(d, width)
        nn.init.orthogonal_(linear.weight, gain=float(np.sqrt(2)))
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
        d = width
    head = nn.Linear(d, out_dim)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.actor = _mlp(obs_dim, num_actions, hidden, out_gain=0.01)
        self.critic = _mlp(obs_dim, 1, hidden, out_gain=1.0)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs).masked_fill(~mask, MASK_FILL)
        value = self.critic(obs).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 2048
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: Tuple[int, ...] = (256, 256)
    seed: int = 0


def compute_gae(rewards, values, ends, last_value, gamma, lam):
    """Generalised advantage estimation over a rollout that may span episodes"""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    ends = np.asarray(ends, dtype=np.float32)
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float32)
    running = 0.0
    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        non_terminal = 1.0 - ends[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        running = delta + gamma * lam * non_terminal * running
        advantages[t] = running
    returns = advantages + values
    return advantages, returns


class PPOAgent:
    def __init__(self, obs_dim: int, num_actions: int, config: Optional[PPOConfig] = None, device: str = "cpu"):
        self.config = config or PPOConfig()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)
        self.net = ActorCritic(obs_dim, num_actions, self.config.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, eps=1e-5)
        self.env_steps = 0
        self.updates = 0
        self.extra: Dict = {}

    def _tensors(self, obs, mask):
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
        if o.dim() == 1:
            o, m = o.unsqueeze(0), m.unsqueeze(0)
        return o, m

    def step(self, obs, mask):
        """Sample an action for data collection. Returns (action, log_prob, value)."""
        with torch.no_grad():
            o, m = self._tensors(obs, mask)
            logits, value = self.net(o, m)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def value(self, obs) -> float:
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.net.critic(o).squeeze().item())

    def act(self, obs, mask, explore: bool = False) -> int:
        """Greedy (default) or sampled action for evaluation"""
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 1:
            return int(np.flatnonzero(mask)[0])
        if explore:
            return self.step(obs, mask)[0]
        with torch.no_grad():
            o, m = self._tensors(obs, mask)
            logits, _ = self.net(o, m)
            return int(torch.argmax(logits, dim=1).item())

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        c = self.config
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        obs = t(batch["obs"])
        masks = t(batch["masks"], torch.bool)
        actions = t(batch["actions"], torch.long)
        old_logp = t(batch["logp"])
        advantages = t(batch["advantages"])
        returns = t(batch["returns"])

        n = obs.shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        count = 0
        for _ in range(c.epochs):
            order = torch.as_tensor(self.rng.permutation(n), device=self.device)
            for start in range(0, n, c.minibatch_size):
                idx = order[start:start + c.minibatch_size]
                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8) if len(idx) > 1 else adv
                logits, value = self.net(obs[idx], masks[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                surrogate = torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1.0 - c.clip, 1.0 + c.clip) * adv,
                )
                policy_loss = -surrogate.mean()
                value_loss = 0.5 * (value - returns[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] += float((old_logp[idx] - logp).mean().item())
                count += 1
        self.updates += 1
        return {k: v / max(1, count) for k, v in stats.items()}

    def save(self, path, extra: Optional[Dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "config": asdict(self.config),
            "net": self.net.state_dict(),
            "env_steps": self.env_steps,
            "updates": self.updates,
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "PPOAgent":
        saved = torch.load(path, map_location=device, weights_only=False)
        config = PPOConfig(**{**saved["config"], "hidden": tuple(saved["config"]["hidden"])})
        agent = cls(saved["obs_dim"], saved["num_actions"], config, device)
        agent.net.load_state_dict(saved["net"])
        agent.env_steps = saved["env_steps"]
        agent.updates = saved["updates"]
        agent.extra = saved.get("extra", {})
        return agent


class PPOPolicy:

    def __init__(self, agent: PPOAgent, encoder):
        self.agent = agent
        self.encoder = encoder

    def __call__(self, observation, info, env):
        x = self.encoder.encode(observation, info["fleet_state"], env.vessel_ids)
        return self.agent.act(x, info["action_mask"], explore=False)


__all__ = ["ActorCritic", "PPOAgent", "PPOConfig", "PPOPolicy", "compute_gae"]
