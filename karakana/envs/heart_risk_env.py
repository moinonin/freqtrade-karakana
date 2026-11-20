import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class HeartRiskEnv(gym.Env):
    """
    HeartRisk-v0: A cardiovascular risk prediction environment using Framingham data.

    The agent acts as a diagnostician. For each patient, it sees their physiological
    structure and must predict whether they will suffer a cardiovascular event (TenYearCHD).

    Actions:
    0: predict_low_risk
    1: predict_high_risk
    """

    def __init__(self, seed=None):
        super().__init__()

        # Load dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(base_dir, "examples", "heart_bmi", "framingham.csv")
        self.df = pd.read_csv(csv_path)

        # Drop NaNs to keep things simple for the RL environment
        self.df = self.df.dropna()

        # We need the first 4 features to be the structural geometry [f1, f2, f3, f4]
        # Based on correlation: f1=age, f2=sysBP, f3=BMI, f4=glucose

        self.df["f1"] = (self.df["age"] - self.df["age"].mean()) / self.df["age"].std()
        self.df["f2"] = (self.df["sysBP"] - self.df["sysBP"].mean()) / self.df["sysBP"].std()
        self.df["f3"] = (self.df["BMI"] - self.df["BMI"].mean()) / self.df["BMI"].std()
        self.df["f4"] = (self.df["glucose"] - self.df["glucose"].mean()) / self.df["glucose"].std()

        # Add male and currentSmoker as extra features (obs size = 6)
        self.df["f5"] = self.df["male"]
        self.df["f6"] = self.df["currentSmoker"]

        self.obs_cols = ["f1", "f2", "f3", "f4", "f5", "f6"]
        self.target_col = "TenYearCHD"

        # Define Action Space (2 discrete actions)
        self.action_space = spaces.Discrete(2)

        # Define Observation Space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32)

        # Training mechanics: process chunks of patients per episode
        self.episode_length = 200
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

        # Randomly select a starting point in the patient database
        max_start = max(0, self.max_steps - self.episode_length - 1)
        if hasattr(self, "np_random"):
            self.start_index = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_index = int(np.random.randint(0, max_start + 1))

        self.current_step = self.start_index
        return self._get_observation(self.current_step), {}

    def _get_observation(self, step_idx):
        row = self.df.iloc[step_idx]
        return np.array([row[c] for c in self.obs_cols], dtype=np.float32)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        actual_chd = int(row[self.target_col])

        # Medical Reward Matrix
        # CHD=1 (Heart Attack will happen)
        # CHD=0 (Patient remains healthy)
        if actual_chd == 1:
            if action == 1:
                reward = 1.0  # True Positive (Caught the heart attack)
            else:
                reward = -1.0  # False Negative (Missed a heart attack - fatal)
        else:
            if action == 0:
                reward = 0.1  # True Negative (Correctly dismissed, small reward to balance classes)
            else:
                reward = -0.5  # False Positive (False alarm - stressful but less fatal than a miss)

        self.current_step += 1
        steps_taken = self.current_step - self.start_index
        terminated = steps_taken >= self.episode_length or self.current_step >= self.max_steps

        obs = (
            self._get_observation(self.current_step)
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        return obs, reward, terminated, False, {"actual_chd": actual_chd}


from gymnasium.envs.registration import register


def register_heart_risk_env():
    # Only register if it hasn't been registered yet
    if "HeartRisk-v0" not in gym.envs.registry:
        register(
            id="HeartRisk-v0",
            entry_point="karakana.envs.heart_risk_env:HeartRiskEnv",
        )
