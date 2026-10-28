from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Masking helpers (pure functions, easy to unit-test and reuse in baselines)
# --------------------------------------------------------------------------

def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    """Greedy action among valid actions. Action 0 is always valid in this env."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Action mask has no valid action.")
    q = np.where(mask, np.asarray(q_values, dtype=np.float64), -np.inf)
    return int(np.argmax(q))


def masked_random_action(mask: np.ndarray, rng: np.random.Generator, idle_prob: float = None) -> int:
    
    mask = np.asarray(mask, dtype=bool)
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError("Action mask has no valid action.")
    if idle_prob is None:
        return int(rng.choice(valid))
    vessels = valid[valid != 0]
    if mask[0] and (vessels.size == 0 or rng.random() < idle_prob):
        return 0
    return int(rng.choice(vessels)) if vessels.size else int(valid[0])


def apply_mask_torch(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min / 2)



class QNetwork(nn.Module):
    

    def __init__(self, obs_dim: int, num_actions: int, hidden: Sequence[int] = (256, 256), dueling: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for width in hidden:
            layers += [nn.Linear(in_dim, width), nn.ReLU()]
            in_dim = width
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(in_dim, 1)
            self.advantage = nn.Linear(in_dim, num_actions)
        else:
            self.head = nn.Linear(in_dim, num_actions)

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.body(obs)
        if not self.dueling:
            return self.head(features)

        value = self.value(features)
        advantage = self.advantage(features)
        # Centre the advantage over valid actions only, so invalid actions
        # do not shift the value estimate.
        if mask is None:
            centre = advantage.mean(dim=1, keepdim=True)
        else:
            maskf = mask.float()
            centre = (advantage * maskf).sum(dim=1, keepdim=True) / maskf.sum(dim=1, keepdim=True).clamp(min=1.0)
        return value + advantage - centre



# Replay buffer (stores the next-state mask for masked targets)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, num_actions: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)  # gamma^k * (1 - terminated)
        self.masks = np.zeros((capacity, num_actions), dtype=bool)
        self.next_masks = np.zeros((capacity, num_actions), dtype=bool)
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, mask, action, reward, discount, next_obs, next_mask):
        i = self.index
        self.obs[i] = obs
        self.masks[i] = mask
        self.actions[i] = action
        self.rewards[i] = reward
        self.discounts[i] = discount
        self.next_obs[i] = next_obs
        self.next_masks[i] = next_mask
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "masks": self.masks[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "discounts": self.discounts[idx],
            "next_obs": self.next_obs[idx],
            "next_masks": self.next_masks[idx],
        }

    def __len__(self):
        return self.size


class NStepAccumulator:

    def __init__(self, n: int, gamma: float):
        if n < 1:
            raise ValueError("n_step must be >= 1.")
        self.n = n
        self.gamma = gamma
        self.queue: Deque[Tuple] = deque()

    def push(self, obs, mask, action, reward, next_obs, next_mask, terminated, truncated):
        self.queue.append((obs, mask, action, reward))
        ready = []
        episode_over = terminated or truncated
        if len(self.queue) >= self.n:
            ready.append(self._emit(next_obs, next_mask, terminated))
        if episode_over:
            while self.queue:
                ready.append(self._emit(next_obs, next_mask, terminated))
        return ready

    def _emit(self, next_obs, next_mask, terminated):
        ret, k = 0.0, 0
        for k, (_, _, _, r) in enumerate(self.queue):
            ret += (self.gamma ** k) * r
        obs, mask, action, _ = self.queue.popleft()
        discount = 0.0 if terminated else self.gamma ** (k + 1)
        return obs, mask, action, ret, discount, next_obs, next_mask

    def reset(self):
        self.queue.clear()



# Agent


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 2_000
    train_every: int = 1
    gradient_steps: int = 1
    tau: float = 0.005               
    n_step: int = 3
    hidden: Tuple[int, ...] = (256, 256)
    dueling: bool = True
    double: bool = True
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 30_000
    explore_idle_prob: float = 0.9   
    max_grad_norm: float = 10.0
    seed: int = 0


class MaskedDQNAgent:
    def __init__(self, obs_dim: int, num_actions: int, config: Optional[DQNConfig] = None, device: str = "cpu"):
        self.config = config or DQNConfig()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.device = torch.device(device)

        torch.manual_seed(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)

        c = self.config
        self.online = QNetwork(obs_dim, num_actions, c.hidden, c.dueling).to(self.device)
        self.target = QNetwork(obs_dim, num_actions, c.hidden, c.dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=c.lr)

        self.buffer = ReplayBuffer(c.buffer_size, obs_dim, num_actions, seed=c.seed)
        self.nstep = NStepAccumulator(c.n_step, c.gamma)
        self.env_steps = 0
        self.updates = 0

  
    @property
    def epsilon(self) -> float:
        c = self.config
        frac = min(1.0, self.env_steps / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def q_values(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            return self.online(o, m).squeeze(0).cpu().numpy()

    def act(self, obs: np.ndarray, mask: np.ndarray, explore: bool = True) -> int:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 1:            # forced (operator busy / no budget)
            return int(np.flatnonzero(mask)[0])
        if explore and self.rng.random() < self.epsilon:
            return masked_random_action(mask, self.rng, self.config.explore_idle_prob)
        return masked_argmax(self.q_values(obs, mask), mask)

    def observe(self, obs, mask, action, reward, next_obs, next_mask, terminated, truncated) -> Optional[Dict[str, float]]:
        """Store a transition and train if due. Returns loss stats when trained."""
        self.env_steps += 1
        for transition in self.nstep.push(obs, mask, action, reward, next_obs, next_mask, terminated, truncated):
            self.buffer.add(*transition)
        if terminated or truncated:
            self.nstep.reset()

        c = self.config
        if len(self.buffer) < max(c.learning_starts, c.batch_size) or self.env_steps % c.train_every:
            return None
        stats = None
        for _ in range(c.gradient_steps):
            stats = self.update()
        return stats

    def update(self) -> Dict[str, float]:
        c = self.config
        batch = self.buffer.sample(c.batch_size)
        t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=self.device)
        obs, next_obs = t(batch["obs"]), t(batch["next_obs"])
        masks, next_masks = t(batch["masks"], torch.bool), t(batch["next_masks"], torch.bool)
        actions = t(batch["actions"], torch.long)
        rewards, discounts = t(batch["rewards"]), t(batch["discounts"])

        q = self.online(obs, masks).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_q = self.target(next_obs, next_masks)
            if c.double:
                online_next = apply_mask_torch(self.online(next_obs, next_masks), next_masks)
                next_actions = online_next.argmax(dim=1, keepdim=True)
            else:
                next_actions = apply_mask_torch(target_q, next_masks).argmax(dim=1, keepdim=True)
            next_value = target_q.gather(1, next_actions).squeeze(1)
            target = rewards + discounts * next_value

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), c.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            for p, tp in zip(self.online.parameters(), self.target.parameters()):
                tp.mul_(1.0 - c.tau).add_(c.tau * p)

        self.updates += 1
        return {
            "loss": float(loss.item()),
            "q_mean": float(q.mean().item()),
            "grad_norm": float(grad_norm),
        }

    
    def save(self, path, extra: Optional[Dict] = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "config": asdict(self.config),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "env_steps": self.env_steps,
            "updates": self.updates,
            "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "MaskedDQNAgent":
        saved = torch.load(path, map_location=device, weights_only=False)
        config = DQNConfig(**{**saved["config"], "hidden": tuple(saved["config"]["hidden"])})
        agent = cls(saved["obs_dim"], saved["num_actions"], config, device)
        agent.online.load_state_dict(saved["online"])
        agent.target.load_state_dict(saved["target"])
        agent.env_steps = saved["env_steps"]
        agent.updates = saved["updates"]
        agent.extra = saved.get("extra", {})
        return agent


__all__ = [
    "DQNConfig",
    "MaskedDQNAgent",
    "NStepAccumulator",
    "QNetwork",
    "ReplayBuffer",
    "apply_mask_torch",
    "masked_argmax",
    "masked_random_action",
]
