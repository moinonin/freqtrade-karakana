import ast
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class HistoricalTradingEnv(gym.Env):
    """
    HistoricalTrading-v0: A custom trading environment that replays the CSV data.

    Actions:
    0: do_nothing
    1: go_long
    2: go_short
    """

    def __init__(self, seed=None):
        super().__init__()

        # Load dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(base_dir, "artifacts", "cleaned_trade_history.csv")
        self.df = pd.read_csv(csv_path)

        # We need to map string actions to ints for the expert
        self.action_map = {"do_nothing": 0, "go_long": 1, "go_short": 2}

        # Define Action Space (3 discrete actions)
        self.action_space = spaces.Discrete(3)

        # Define Observation Space (f1, f2, f3, f4, is_short) -> 5 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.episode_length = 1000
        self.current_step = 0
        self.start_index = 0
        self.max_steps = len(self.df) - 1
        self.seed_val = seed

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        if seed is None and self.seed_val is not None:
            seed = self.seed_val
        super().reset(seed=seed)

        # Randomly select a starting point so the agent sees different market regimes
        max_start = max(0, self.max_steps - self.episode_length - 1)
        if hasattr(self, "np_random"):
            self.start_index = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_index = int(np.random.randint(0, max_start + 1))

        self.current_step = self.start_index
        return self._get_observation(self.current_step), {}

    def _get_observation(self, step_idx):
        row = self.df.iloc[step_idx]

        # Parse state tuple string
        state_tuple = ast.literal_eval(row["state"])

        # is_short flag based on trade_position
        is_short = 1.0 if row["trade_position"] == "short" else 0.0

        # Combine into [f1, f2, f3, f4, is_short]
        obs = np.array([state_tuple[0], state_tuple[1], state_tuple[2], state_tuple[3], is_short], dtype=np.float32)
        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]
        expert_action_str = row["action"]
        expert_reward = float(row["reward"])

        # Deduce the true market return from the expert's position
        market_return = 0.0
        if expert_action_str == "go_long":
            market_return = expert_reward
        elif expert_action_str == "go_short":
            market_return = -expert_reward
        else:
            # If expert did nothing, we cannot know the exact market tick return from the CSV.
            # We assume it was 0 (neutral), so if the RL agent takes a position here, it just pays a small penalty.
            market_return = 0.0

        # Calculate the RL Agent's reward based on ITS action, not the expert's action
        penalty = -0.0001  # Cost of time / flat penalty
        if action == 0:  # do_nothing
            reward = penalty
        elif action == 1:  # go_long
            reward = market_return
        elif action == 2:  # go_short
            reward = -market_return
        else:
            reward = penalty

        self.current_step += 1
        steps_taken = self.current_step - self.start_index
        terminated = steps_taken >= self.episode_length or self.current_step >= self.max_steps
        truncated = False

        obs = np.zeros(5, dtype=np.float32)
        if not terminated:
            obs = self._get_observation(self.current_step)

        return obs, reward, terminated, truncated, {}


def register_historical_env():
    from gymnasium.envs.registration import register

    try:
        register(
            id="HistoricalTrading-v0",
            entry_point="karakana.envs.historical_trading_env:HistoricalTradingEnv",
            max_episode_steps=None,  # Handles its own termination
        )
        print("[SGO] Custom HistoricalTrading-v0 environment registered successfully!", flush=True)
    except Exception:
        pass  # Already registered
