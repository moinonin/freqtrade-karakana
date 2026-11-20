"""Pure Dual-Cosine Trainer for continuous & discrete policy optimization without PPO/GRPO."""

from __future__ import annotations

import uuid

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from karakana_engine.trainers.grg import GRGController
from torch.distributions import Categorical, Normal

from karakana.config import get_config
from karakana.torch_dual_cosine import AnchoredDualCosineSGOLoss, DualCosineSGOLoss


class DualCosinePolicy(nn.Module):
    """Simple MLP Policy with actor/critic heads for discrete environments."""

    def __init__(self, obs_size: int, action_count: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
        self.actor = nn.Linear(32, action_count)
        self.critic = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.actor(h), self.critic(h)

    def action_probs(self, x: torch.Tensor):
        logits, _ = self.forward(x)
        return functional.softmax(logits, dim=-1)


class DualCosineContinuousPolicy(nn.Module):
    """Gaussian Actor-Critic Policy for continuous action spaces."""

    def __init__(self, obs_size: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
        self.mu_head = nn.Linear(32, action_dim)
        self.critic = nn.Linear(32, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.mu_head(h)
        std = torch.exp(self.log_std)
        value = self.critic(h).squeeze(-1)
        return mu, std, value


class PureDualCosineTrainer:
    """
    Pure Dual-Cosine Policy Trainer.
    Trains policies using differentiable Dual-Cosine Structural Geometry Optimization
    on trajectory-level signed trade/reward outcomes, bypassing PPO/GRPO policy gradients.
    Supports both discrete (gym.spaces.Discrete) and continuous (gym.spaces.Box) environments.

    Uses the ANCHORED variant (hinge magnitude anchor, margin/w_hinge): the pure
    loss is scale-invariant, so its gradient dies once outcome chunks homogenize
    (the single-sign dead zone) and training stalls short of the optimum. The
    anchor keeps the gradient alive while any outcome is below margin. The
    alpha-health signal (``current_alpha``, feeding GRGController steering) is
    still computed from the PURE dual-cosine value so its semantics stay stable.
    Set ``w_hinge=0`` to recover the pure-loss objective exactly.
    """

    def __init__(
        self,
        env_name: str,
        lr: float = 0.01,
        k: int = 2,
        utility_power: float = 1.1,
        w_alpha: float = 1.0,
        w_coupling: float = 0.1,
        margin: float | None = None,
        w_hinge: float | None = None,
        steering_enabled: bool = True,
        profile_name: str = "gym_rl",
        num_epochs: int | None = None,
        env: gym.Env | None = None,
        **kwargs,
    ):
        self.env_name = env_name
        self.env = env if env is not None else gym.make(env_name)
        obs_size = int(self.env.observation_space.shape[0])
        self.is_continuous = isinstance(self.env.action_space, gym.spaces.Box)

        if self.is_continuous:
            action_dim = int(self.env.action_space.shape[0])
            self.action_size = action_dim
            self.action_count = action_dim
            self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
            self.policy = DualCosineContinuousPolicy(obs_size, action_dim)
        else:
            action_count = int(self.env.action_space.n)
            self.action_size = action_count
            self.action_count = action_count
            self.policy = DualCosinePolicy(obs_size, action_count)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Config resolution
        k_val = int(k if k is not None else get_config.get("coupling_k", 2))
        p_val = float(utility_power if utility_power is not None else get_config.get("utility_power", 1.1))
        w_a = float(w_alpha if w_alpha is not None else get_config.get("w_alpha", 1.0))
        w_c = float(w_coupling if w_coupling is not None else get_config.get("w_coupling", 0.1))
        m_val = float(margin if margin is not None else get_config.get("sgo_margin", 0.0))
        wh_val = float(w_hinge if w_hinge is not None else get_config.get("w_hinge", 1.0))
        # Dominance rule (rule of thumb): w_coupling <= 0.1 * w_alpha. The coupling
        # term is symmetric in P_u and N_u, so if it dominates early training when
        # N_u > P_u it pushes P_u -> 0 (all losses). Enforce, don't just document.
        if w_c > 0.1 * w_a:
            w_c = 0.1 * w_a

        # Training objective: anchored loss (pure + hinge magnitude anchor).
        # w_hinge=0 degenerates to the pure loss exactly.
        self.loss_fn = AnchoredDualCosineSGOLoss(
            k=k_val, p=p_val, w_alpha=w_a, w_coupling=w_c, margin=m_val, w_hinge=wh_val
        )
        # Alpha-health metric stays on the PURE loss: current_alpha feeds
        # GRGController steering and should keep its pre-anchor semantics.
        self.eval_loss_fn = DualCosineSGOLoss(k=k_val, p=p_val, w_alpha=w_a, w_coupling=w_c)
        self.grg = GRGController(self.optimizer, base_lr=lr, profile=profile_name)
        self.steering_enabled = steering_enabled
        self.num_epochs = num_epochs
        self.session_id = f"dual-cosine-{uuid.uuid4().hex[:8]}"

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self._last_alpha = 1.0

    @property
    def current_alpha(self) -> float:
        return self._last_alpha

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        if self.is_continuous:
            mu, std, _ = self.policy(state_t)
            dist = Normal(mu, std)
            action = dist.sample()
            clamped = torch.clamp(action, torch.as_tensor(self.action_low), torch.as_tensor(self.action_high))
            return clamped.detach().cpu().numpy()
        else:
            logits, _ = self.policy(state_t)
            probs = functional.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item()

    def train_episode(self):
        state, _ = self.env.reset()
        done = False
        truncated = False
        states, actions, rewards = [], [], []

        while not (done or truncated):
            action = self.select_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        episode_reward = sum(rewards)
        if len(rewards) < self.loss_fn.k:
            return float(episode_reward), 0.0

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)

        # Baseline centered rewards
        centered_r = rewards_t - rewards_t.mean()

        if self.is_continuous:
            actions_t = torch.tensor(np.array(actions), dtype=torch.float32)
            mu, std, _ = self.policy(states_t)
            dist = Normal(mu, std)
            log_probs = dist.log_prob(actions_t).sum(dim=-1)
            margin = torch.tanh(log_probs - log_probs.mean())
            diff_outcomes = margin * centered_r
        else:
            actions_t = torch.tensor(np.array(actions), dtype=torch.long)
            logits, _ = self.policy(states_t)
            chosen_logits = logits.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
            mean_logits = logits.mean(dim=-1)
            logit_margin = torch.tanh(chosen_logits - mean_logits)
            diff_outcomes = logit_margin * centered_r

        self.optimizer.zero_grad()
        loss = self.loss_fn(diff_outcomes)
        loss.backward()
        self.optimizer.step()

        # Update alpha health score (pure dual-cosine value, anchor excluded)
        with torch.no_grad():
            pure_loss = self.eval_loss_fn(diff_outcomes)
            self._last_alpha = float(max(0.0, 1.0 - pure_loss.detach().item() / 2.0))

        if self.steering_enabled:
            self.grg.step(self._last_alpha)

        return float(episode_reward), float(loss.item())

    def should_stop(self) -> bool:
        return False
