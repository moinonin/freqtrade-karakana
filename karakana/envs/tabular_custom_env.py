"""Declarative Custom Gymnasium Environment built from user CSV telemetry.

Allows users to turn any numeric tabular dataset into an interactive Gym
environment (without writing Python) for training with ``PureDualCosineTrainer``.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TabularCustomEnv(gym.Env):
    """Parameterised Gym environment backed by a user-uploaded DataFrame."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        observation_columns: list[str],
        reward_column: str,
        action_mode: str = "discrete_2",
        episode_length: int = 100,
        seed: int = 42,
    ):
        super().__init__()
        if not observation_columns:
            raise ValueError("observation_columns must contain at least one numeric column")
        if reward_column not in df.columns:
            raise ValueError(f"reward_column '{reward_column}' not found in dataset")
        missing_obs = [c for c in observation_columns if c not in df.columns]
        if missing_obs:
            raise ValueError(f"Missing observation columns: {missing_obs}")

        self.observation_columns = list(observation_columns)
        self.reward_column = reward_column
        self.action_mode = action_mode
        self.episode_length = max(10, int(episode_length))
        self._rng = np.random.default_rng(seed)

        obs_df = df[self.observation_columns].apply(pd.to_numeric, errors="coerce")
        obs_df = obs_df.fillna(obs_df.median(numeric_only=True)).fillna(0.0)
        raw_obs = obs_df.to_numpy(dtype=np.float32)

        # Standardize observations for stable policy learning
        mean = raw_obs.mean(axis=0, keepdims=True)
        std = raw_obs.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        self._obs = ((raw_obs - mean) / std).astype(np.float32)

        rewards = pd.to_numeric(df[self.reward_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        self._rewards = rewards
        self._n_rows = int(len(self._obs))
        if self._n_rows < 2:
            raise ValueError("Custom environment dataset must contain at least 2 rows")

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.observation_columns),),
            dtype=np.float32,
        )
        if action_mode == "discrete_2":
            self.action_space = spaces.Discrete(2)
        elif action_mode == "discrete_3":
            self.action_space = spaces.Discrete(3)
        elif action_mode == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action_mode '{action_mode}'. Choose discrete_2, discrete_3, or continuous.")

        self._cursor = 0
        self._step_in_ep = 0

    @classmethod
    def from_csv_b64(
        cls,
        dataset_csv_b64: str,
        *,
        observation_columns: list[str],
        reward_column: str,
        action_mode: str = "discrete_2",
        episode_length: int = 100,
        seed: int = 42,
    ) -> TabularCustomEnv:
        raw = base64.b64decode(dataset_csv_b64, validate=True)
        df = pd.read_csv(io.BytesIO(raw))
        return cls(
            df,
            observation_columns=observation_columns,
            reward_column=reward_column,
            action_mode=action_mode,
            episode_length=episode_length,
            seed=seed,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = max(0, self._n_rows - self.episode_length - 1)
        self._cursor = int(self._rng.integers(0, max_start + 1)) if max_start > 0 else 0
        self._step_in_ep = 0
        return self._obs[self._cursor].copy(), {}

    def step(self, action):
        base_r = float(self._rewards[self._cursor])
        if self.action_mode == "discrete_2":
            # 0 = Skip (0.0), 1 = Take (+base_r)
            reward = base_r if int(action) == 1 else 0.0
        elif self.action_mode == "discrete_3":
            # 0 = Short (-base_r), 1 = Flat (0.0), 2 = Long (+base_r)
            act = int(action)
            reward = -base_r if act == 0 else (base_r if act == 2 else 0.0)
        else:
            # Continuous scalar position in [-1, 1]
            scale = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
            reward = base_r * scale

        self._cursor = (self._cursor + 1) % self._n_rows
        self._step_in_ep += 1
        terminated = self._step_in_ep >= min(self.episode_length, self._n_rows - 1)
        truncated = False
        return self._obs[self._cursor].copy(), float(reward), terminated, truncated, {}
