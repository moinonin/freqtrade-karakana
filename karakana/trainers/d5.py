"""D5 Trainer: Decoupled Actor-Critic + GAE + PPO Clip + Pure Dual-Cosine SGO.

Supports discrete vector environments, continuous Box environments, custom
tabular environments, and Atari pixel environments (NoFrameskip-v4) through
one unified structural geometry objective.
"""

from __future__ import annotations

import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from karakana_engine.trainers.grg import GRGController
from torch.distributions import Categorical, Normal

from karakana.config import get_config
from karakana.torch_dual_cosine import DualCosineSGOLoss


def _layer_init(layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _select_device(is_image: bool) -> torch.device:
    """Use MPS/CUDA for CNN workloads; CPU for small MLPs to avoid transfer overhead."""
    if not is_image:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _AtariFrameStackWrapper(gym.Wrapper):
    """Wraps AtariWrapper with a 4-frame (4, 84, 84) uint8 stack and raw_reward tracking."""

    def __init__(self, env: gym.Env, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = int(n_stack)
        self._frames: deque[np.ndarray] = deque(maxlen=self.n_stack)
        self.raw_reward: float = 0.0
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.n_stack, 84, 84), dtype=np.uint8
        )

    @staticmethod
    def _squeeze_frame(obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[:, :, 0]
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def _get_obs(self) -> np.ndarray:
        return np.stack(list(self._frames), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_reward = 0.0
        frame = self._squeeze_frame(obs)
        self._frames.clear()
        for _ in range(self.n_stack):
            self._frames.append(frame)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.raw_reward += float(reward)
        self._frames.append(self._squeeze_frame(obs))
        return self._get_obs(), reward, done, truncated, info


class _FireResetWrapper(gym.Wrapper):
    """Presses FIRE on reset for Atari games that require it to start."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings() if hasattr(env.unwrapped, "get_action_meanings") else []
        self._needs_fire = len(meanings) >= 3 and meanings[1] == "FIRE"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._needs_fire:
            obs, _, terminated, truncated, info = self.env.step(1)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
            obs, _, terminated, truncated, info = self.env.step(2)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


def _wrap_atari_env(env: gym.Env, n_stack: int = 4) -> gym.Env:
    try:
        from stable_baselines3.common.atari_wrappers import AtariWrapper

        wrapped: gym.Env = AtariWrapper(env)
    except ImportError:
        from gymnasium.wrappers import AtariPreprocessing, TransformReward

        wrapped = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            scale_obs=False,
        )
        wrapped = _FireResetWrapper(wrapped)
        wrapped = TransformReward(wrapped, lambda r: float(np.sign(r)))

    return _AtariFrameStackWrapper(wrapped, n_stack=n_stack)


class D5DiscretePolicy(nn.Module):
    """Decoupled Actor-Critic MLP for discrete vector environments."""

    def __init__(self, obs_size: int, action_count: int):
        super().__init__()
        self.actor_net = nn.Sequential(
            _layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
            _layer_init(nn.Linear(32, action_count), std=0.01),
        )
        self.critic_net = nn.Sequential(
            _layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
            _layer_init(nn.Linear(32, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor):
        return self.actor_net(x), self.critic_net(x).squeeze(-1)

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.actor_net(x)
        return F.softmax(logits, dim=-1)


class D5ContinuousPolicy(nn.Module):
    """Decoupled Gaussian Actor-Critic MLP for continuous Box environments."""

    def __init__(self, obs_size: int, action_dim: int):
        super().__init__()
        self.actor_net = nn.Sequential(
            _layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
            _layer_init(nn.Linear(32, action_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_net = nn.Sequential(
            _layer_init(nn.Linear(obs_size, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
            _layer_init(nn.Linear(32, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor):
        mu = self.actor_net(x)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 2.0))
        value = self.critic_net(x).squeeze(-1)
        return mu, std, value


class D5AtariPolicy(nn.Module):
    """NatureCNN Actor-Critic for stacked grayscale Atari pixel observations (C, 84, 84)."""

    def __init__(self, in_channels: int, action_count: int):
        super().__init__()
        self.encoder = nn.Sequential(
            _layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.actor = _layer_init(nn.Linear(512, action_count), std=0.01)
        self.critic = _layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x: torch.Tensor):
        h = self.encoder(x / 255.0)
        return self.actor(h), self.critic(h).squeeze(-1)

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


class D5Trainer:
    """D5: Clipped Actor-Critic + GAE + Pure Dual-Cosine SGO Regularizer.

    Resolves defaults from ``config/karakana-config.json`` (``global.d5``).
    Works across discrete vector envs, continuous Box envs, custom tabular
    envs, and Atari NoFrameskip pixel envs.
    """

    def __init__(
        self,
        env_name: str,
        lr: float | None = None,
        gamma: float | None = None,
        gae_lambda: float | None = None,
        clip_eps: float | None = None,
        ppo_epochs: int | None = None,
        lambda_sgo: float | None = None,
        entropy_coef: float | None = None,
        vf_coef: float | None = None,
        max_grad_norm: float | None = None,
        minibatch_size: int | None = None,
        k: int | None = None,
        utility_power: float | None = None,
        w_alpha: float | None = None,
        w_coupling: float | None = None,
        steering_enabled: bool | None = None,
        profile_name: str = "gym_rl",
        num_epochs: int | None = None,
        env: gym.Env | None = None,
        device: str | torch.device | None = None,
        **kwargs,
    ):
        self.env_name = env_name
        defaults = get_config.get_d5_defaults(env_name)

        self.is_atari = bool(env_name and "NoFrameskip" in env_name)
        if env is not None:
            self.env = env
        else:
            if self.is_atari:
                import ale_py

                gym.register_envs(ale_py)
                self.env = _wrap_atari_env(gym.make(env_name), n_stack=4)
            else:
                self.env = gym.make(env_name)

        obs_shape = self.env.observation_space.shape
        self.is_image = bool(self.is_atari or (obs_shape is not None and len(obs_shape) == 3))
        self.is_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        self.device = torch.device(device) if device is not None else _select_device(self.is_image)

        # Resolve hyperparameters from config/karakana-config.json defaults
        self.lr = float(lr if lr is not None else defaults["learning_rate"])
        self.gamma = float(gamma if gamma is not None else defaults["gamma"])
        self.gae_lambda = float(gae_lambda if gae_lambda is not None else defaults["gae_lambda"])
        self.clip_eps = float(clip_eps if clip_eps is not None else defaults["clip_eps"])
        self.ppo_epochs = max(1, int(ppo_epochs if ppo_epochs is not None else defaults["ppo_epochs"]))
        self.lambda_sgo = float(lambda_sgo if lambda_sgo is not None else defaults["lambda_sgo"])
        self.entropy_coef = float(entropy_coef if entropy_coef is not None else defaults["entropy_coef"])
        self.vf_coef = float(vf_coef if vf_coef is not None else defaults["vf_coef"])
        self.max_grad_norm = float(max_grad_norm if max_grad_norm is not None else defaults["max_grad_norm"])
        self.minibatch_size = max(16, int(minibatch_size if minibatch_size is not None else defaults["minibatch_size"]))
        self.steering_enabled = bool(
            steering_enabled if steering_enabled is not None else defaults["steering_enabled"]
        )
        self.num_epochs = num_epochs

        k_val = max(1, int(k if k is not None else defaults["coupling_k"]))
        p_val = float(utility_power if utility_power is not None else defaults["utility_power"])
        w_a = float(w_alpha if w_alpha is not None else defaults["w_alpha"])
        w_c = float(w_coupling if w_coupling is not None else defaults["w_coupling"])
        if w_c > 0.1 * w_a:
            w_c = 0.1 * w_a

        if self.is_image:
            in_channels = int(obs_shape[0])
            action_count = int(self.env.action_space.n)
            self.action_size = action_count
            self.action_count = action_count
            self.policy = D5AtariPolicy(in_channels, action_count).to(self.device)
            self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
            self.critic_optimizer = None
        elif self.is_continuous:
            obs_size = int(obs_shape[0])
            action_dim = int(self.env.action_space.shape[0])
            self.action_size = action_dim
            self.action_count = action_dim
            self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
            self.policy = D5ContinuousPolicy(obs_size, action_dim).to(self.device)
            self.actor_optimizer = torch.optim.Adam(
                list(self.policy.actor_net.parameters()) + [self.policy.log_std], lr=self.lr
            )
            self.critic_optimizer = torch.optim.Adam(self.policy.critic_net.parameters(), lr=self.lr)
        else:
            obs_size = int(obs_shape[0])
            action_count = int(self.env.action_space.n)
            self.action_size = action_count
            self.action_count = action_count
            self.policy = D5DiscretePolicy(obs_size, action_count).to(self.device)
            self.actor_optimizer = torch.optim.Adam(self.policy.actor_net.parameters(), lr=self.lr)
            self.critic_optimizer = torch.optim.Adam(self.policy.critic_net.parameters(), lr=self.lr)

        # Exposed for GRGController and checkpoint resume plumbing
        self.optimizer = self.actor_optimizer
        self.loss_fn = DualCosineSGOLoss(k=k_val, p=p_val, w_alpha=w_a, w_coupling=w_c)
        self.grg = GRGController(self.optimizer, base_lr=self.lr, profile=profile_name)
        self.session_id = f"d5-{uuid.uuid4().hex[:8]}"
        self._last_alpha = 1.0
        self.cancel_requested: Callable[[], bool] | None = None

    @property
    def current_alpha(self) -> float:
        return self._last_alpha

    def preprocess_observation(self, obs: np.ndarray, update: bool = False) -> torch.Tensor:
        del update
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _compute_gae(
        self,
        rewards: list[float],
        values: np.ndarray,
        next_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            nv = float(values[t + 1]) if t + 1 < n else float(next_value)
            delta = float(rewards[t]) + self.gamma * nv - float(values[t])
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            adv[t] = last_gae
        returns = adv + values
        return (
            torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            torch.as_tensor(adv, dtype=torch.float32, device=self.device),
        )

    def _update_from_batch(
        self,
        states_t: torch.Tensor,
        actions_t: torch.Tensor,
        old_logp_t: torch.Tensor,
        returns_t: torch.Tensor,
        adv_t: torch.Tensor,
    ) -> tuple[float, float]:
        n = states_t.shape[0]
        mb_size = min(self.minibatch_size, n)
        last_loss = 0.0
        last_sgo = 0.0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                if idx.numel() < self.loss_fn.k:
                    continue

                mb_s = states_t[idx]
                mb_a = actions_t[idx]
                mb_old_lp = old_logp_t[idx]
                mb_ret = returns_t[idx]
                mb_adv = adv_t[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                if self.is_continuous:
                    mu, std, vals = self.policy(mb_s)
                    dist = Normal(mu, std)
                    new_lp = dist.log_prob(mb_a).sum(dim=-1)
                    ent = dist.entropy().sum(dim=-1).mean()
                    margin = torch.tanh(new_lp - new_lp.mean())
                else:
                    logits, vals = self.policy(mb_s)
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(mb_a)
                    ent = dist.entropy().mean()
                    chosen = logits.gather(1, mb_a.unsqueeze(-1)).squeeze(-1)
                    margin = torch.tanh(chosen - logits.mean(dim=-1))

                ratio = torch.exp(torch.clamp(new_lp - mb_old_lp, -20.0, 20.0))
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                pg_loss = -torch.min(s1, s2).mean()

                sgo_l = self.loss_fn(margin * mb_adv)
                vf_loss = F.mse_loss(vals, mb_ret)

                if self.critic_optimizer is None:
                    # Shared CNN encoder (Atari)
                    total_loss = pg_loss + self.lambda_sgo * sgo_l + self.vf_coef * vf_loss - self.entropy_coef * ent
                    self.actor_optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    last_loss = float(total_loss.item())
                else:
                    # Decoupled actor & critic MLPs
                    actor_loss = pg_loss + self.lambda_sgo * sgo_l - self.entropy_coef * ent
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.actor_net.parameters(), self.max_grad_norm)
                    if self.is_continuous:
                        nn.utils.clip_grad_norm_([self.policy.log_std], self.max_grad_norm)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.critic_net.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()
                    last_loss = float(actor_loss.item())

                last_sgo = float(sgo_l.detach().item())

        return last_loss, last_sgo

    def train_episode(self) -> tuple[float, float]:
        obs, _ = self.env.reset()
        done = False
        truncated = False
        states: list[np.ndarray] = []
        actions: list[Any] = []
        logps: list[float] = []
        values: list[float] = []
        rewards: list[float] = []

        max_steps = 4000 if self.is_atari else 10000
        step_count = 0

        while not (done or truncated) and step_count < max_steps:
            step_count += 1
            if self.cancel_requested and step_count % 50 == 0 and self.cancel_requested():
                break

            obs_t = self.preprocess_observation(obs).unsqueeze(0)
            with torch.no_grad():
                if self.is_continuous:
                    mu, std, val = self.policy(obs_t)
                    dist = Normal(mu, std)
                    act_t = dist.sample()
                    lp = dist.log_prob(act_t).sum(dim=-1)
                    act_raw = act_t.squeeze(0).cpu().numpy()
                    act_env = np.clip(act_raw, self.action_low, self.action_high).astype(np.float32)
                    actions.append(act_raw)
                else:
                    logits, val = self.policy(obs_t)
                    dist = Categorical(logits=logits)
                    act_t = dist.sample()
                    lp = dist.log_prob(act_t)
                    act_env = int(act_t.item())
                    actions.append(act_env)

            next_obs, reward, done, truncated, _ = self.env.step(act_env)
            states.append(np.asarray(obs))
            logps.append(float(lp.item()))
            values.append(float(val.squeeze().item()))
            rewards.append(float(reward))
            obs = next_obs

        episode_reward = float(sum(rewards))
        if len(rewards) < self.loss_fn.k:
            return episode_reward, 0.0

        with torch.no_grad():
            if truncated and not done:
                last_obs_t = self.preprocess_observation(obs).unsqueeze(0)
                if self.is_continuous:
                    _, _, next_v = self.policy(last_obs_t)
                else:
                    _, next_v = self.policy(last_obs_t)
                next_value = float(next_v.squeeze().item())
            else:
                next_value = 0.0

        returns_t, adv_t = self._compute_gae(rewards, np.asarray(values, dtype=np.float32), next_value)
        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(
            np.asarray(actions),
            dtype=torch.float32 if self.is_continuous else torch.long,
            device=self.device,
        )
        old_logp_t = torch.as_tensor(np.asarray(logps, dtype=np.float32), dtype=torch.float32, device=self.device)

        last_loss, last_sgo = self._update_from_batch(states_t, actions_t, old_logp_t, returns_t, adv_t)
        self._last_alpha = float(max(0.0, min(1.0, 1.0 - last_sgo / 2.0)))

        if self.steering_enabled:
            self.grg.step(self._last_alpha)

        return episode_reward, last_loss

    def train_timesteps(
        self,
        total_timesteps: int,
        n_envs: int = 8,
        n_steps: int = 128,
        seed: int = 0,
        log_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Vectorized rollout trainer for step-budgeted Atari / pixel runs."""
        import time

        from stable_baselines3.common.atari_wrappers import AtariWrapper
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage

        if self.is_atari:
            import ale_py

            gym.register_envs(ale_py)

        torch.manual_seed(seed)
        np.random.seed(seed)

        env_name = self.env_name
        is_atari = self.is_atari

        def _make_env(rank: int):
            def _init():
                if is_atari:
                    import ale_py

                    gym.register_envs(ale_py)
                e = gym.make(env_name)
                e = Monitor(e)
                if is_atari:
                    e = AtariWrapper(e)
                e.reset(seed=seed + rank)
                return e

            return _init

        vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        vec = vec_cls([_make_env(i) for i in range(n_envs)])
        if self.is_atari:
            vec = VecTransposeImage(VecFrameStack(vec, n_stack=4))

        batch_size = n_envs * n_steps
        n_updates = max(1, total_timesteps // batch_size)
        obs_shape = vec.observation_space.shape

        obs_buf = np.zeros((n_steps, n_envs, *obs_shape), dtype=vec.observation_space.dtype)
        act_buf = np.zeros((n_steps, n_envs), dtype=np.int64)
        logp_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        rew_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        done_buf = np.zeros((n_steps, n_envs), dtype=np.float32)
        val_buf = np.zeros((n_steps, n_envs), dtype=np.float32)

        ep_returns: list[float] = []
        recent_returns: deque[float] = deque(maxlen=20)
        history: list[dict[str, Any]] = []
        obs = vec.reset()
        t0 = time.time()

        try:
            for update in range(1, n_updates + 1):
                for step in range(n_steps):
                    obs_buf[step] = obs
                    with torch.no_grad():
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                        logits, val = self.policy(obs_t)
                        dist = Categorical(logits=logits)
                        act = dist.sample()
                        logp = dist.log_prob(act)

                    act_np = act.cpu().numpy()
                    next_obs, rew, dones, infos = vec.step(act_np)
                    act_buf[step] = act_np
                    logp_buf[step] = logp.cpu().numpy()
                    rew_buf[step] = rew
                    done_buf[step] = dones.astype(np.float32)
                    val_buf[step] = val.cpu().numpy()

                    for info in infos:
                        if "episode" in info:
                            r_val = float(info["episode"]["r"])
                            ep_returns.append(r_val)
                            recent_returns.append(r_val)
                    obs = next_obs

                with torch.no_grad():
                    _, next_val = self.policy(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
                    next_val_np = next_val.cpu().numpy()

                adv = np.zeros_like(rew_buf)
                last_gae = np.zeros(n_envs, dtype=np.float32)
                for t in reversed(range(n_steps)):
                    nonterm = 1.0 - done_buf[t]
                    nv = next_val_np if t == n_steps - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + self.gamma * nv * nonterm - val_buf[t]
                    last_gae = delta + self.gamma * self.gae_lambda * nonterm * last_gae
                    adv[t] = last_gae
                ret = adv + val_buf

                states_t = torch.as_tensor(
                    obs_buf.reshape((-1, *obs_shape)), dtype=torch.float32, device=self.device
                )
                actions_t = torch.as_tensor(act_buf.reshape(-1), dtype=torch.long, device=self.device)
                old_logp_t = torch.as_tensor(logp_buf.reshape(-1), dtype=torch.float32, device=self.device)
                returns_t = torch.as_tensor(ret.reshape(-1), dtype=torch.float32, device=self.device)
                adv_t = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=self.device)

                last_loss, last_sgo = self._update_from_batch(states_t, actions_t, old_logp_t, returns_t, adv_t)
                self._last_alpha = float(max(0.0, min(1.0, 1.0 - last_sgo / 2.0)))
                if self.steering_enabled:
                    self.grg.step(self._last_alpha)

                steps_done = update * batch_size
                sps = steps_done / max(time.time() - t0, 1e-6)
                row = {
                    "update": update,
                    "timesteps": steps_done,
                    "episodes_completed": len(ep_returns),
                    "mean_return_20": float(np.mean(recent_returns)) if recent_returns else 0.0,
                    "best_return": float(np.max(ep_returns)) if ep_returns else 0.0,
                    "loss": last_loss,
                    "sgo_loss": last_sgo,
                    "alpha": self._last_alpha,
                    "lr": self.grg.current_lr,
                    "sps": float(sps),
                }
                history.append(row)
                if log_callback is not None:
                    log_callback(row)
        finally:
            vec.close()

        return {
            "env_name": self.env_name,
            "total_timesteps": n_updates * batch_size,
            "wall_seconds": time.time() - t0,
            "episodes_completed": len(ep_returns),
            "mean_return_20": float(np.mean(recent_returns)) if recent_returns else 0.0,
            "best_return": float(np.max(ep_returns)) if ep_returns else 0.0,
            "final_alpha": self._last_alpha,
            "history": history,
        }

    def should_stop(self) -> bool:
        return bool(self.steering_enabled and self.grg.should_stop())
