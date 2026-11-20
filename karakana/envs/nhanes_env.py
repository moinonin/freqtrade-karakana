import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


def register_nhanes_env():
    # Registration is called lazily per trainer build; gym rejects duplicate ids.
    if "NHANES-v0" in gym.envs.registry:
        return
    gym.register(
        id="NHANES-v0",
        entry_point="karakana.envs.nhanes_env:NHANESEnv",
        max_episode_steps=7000,
    )


class NHANESEnv(gym.Env):
    """
    SGO Environment for NHANES sociological and physiological data.
    Evaluates the structural geometry of diabetes risk across diverse domains.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, seed=None):
        super().__init__()

        # Load dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        csv_path = os.path.join(base_dir, "examples", "heart_bmi", "nhanes.csv")
        self.df = pd.read_csv(csv_path, low_memory=False)

        self.features = ["Age", "BMI", "BPSysAve", "TotChol", "SleepHrsNight", "Poverty"]
        target = "Diabetes"

        # Drop missing values
        self.df = self.df.dropna(subset=self.features + [target]).copy()

        # Map target (Yes -> 1, No -> 0)
        # Drop borderline cases like 'Borderline' or 'Refused' if they exist, or map them to 0
        self.df = self.df[self.df[target].isin(["Yes", "No"])]
        self.df[target] = self.df[target].map({"Yes": 1, "No": 0}).astype(int)

        # Scale features using standard normalization (z-score)
        for f in self.features:
            # Convert to numeric just in case
            self.df[f] = pd.to_numeric(self.df[f], errors="coerce")
            self.df = self.df.dropna(subset=[f])

            mean = self.df[f].mean()
            std = self.df[f].std()
            if std > 0:
                self.df[f] = (self.df[f] - mean) / std

        # State: patient's features
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(len(self.features),), dtype=np.float32)

        # Action: Diagnose Diabetes (1) or No Diabetes (0)
        self.action_space = spaces.Discrete(2)

        # RL State Variables
        self.current_step = 0
        self.indices = self.df.index.tolist()

        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.np_random.shuffle(self.indices)
        else:
            np.random.shuffle(self.indices)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.np_random.shuffle(self.indices)
        else:
            np.random.shuffle(self.indices)

        return self._get_obs(), {}

    def _get_obs(self):
        idx = self.indices[self.current_step]
        row = self.df.loc[idx, self.features]
        return row.values.astype(np.float32)

    def step(self, action):
        idx = self.indices[self.current_step]
        true_outcome = self.df.loc[idx, "Diabetes"]

        # Reward Matrix for SGO Medical Geometry
        # True Positive (Caught Diabetes) -> High Reward
        # False Negative (Missed Diabetes) -> Heavy Penalty
        if action == 1 and true_outcome == 1:
            reward = 5.0
        elif action == 0 and true_outcome == 1:
            reward = -2.0
        elif action == 1 and true_outcome == 0:
            reward = -0.5
        else:
            reward = 0.1  # True Negative

        self.current_step += 1
        done = self.current_step >= len(self.indices)
        truncated = False

        obs = self._get_obs() if not done else np.zeros(len(self.features), dtype=np.float32)

        return obs, reward, done, truncated, {"true_label": true_outcome}
