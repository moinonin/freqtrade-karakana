import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StructuralTrapEnv(gym.Env):
    """
    StructuralTrap-v0: A high-tail-risk environment.

    Actions:
    0: Stay (Safe, reward = 0.1)
    1: Gamble (Fragile, reward = +1 99% of time, -1000 1% of time)

    Expected Value:
    Action 0: 0.1
    Action 1: (0.99 * 1) + (0.01 * -1000) = 0.99 - 10.0 = -9.01

    Wait, if the EV is negative, naive PPO will avoid it anyway.
    Let's make Action 1 have POSITIVE EV but high risk.

    Action 1: +10 (99%), -50 (1%) -> EV = 9.9 - 0.5 = +9.4
    Action 0: +1 (100%) -> EV = +1.0

    Here, a naive reward maximizer should pick Action 1 (9.4 > 1.0).
    However, Action 1 is structurally fragile (asymmetric risk).
    """

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.state = np.array([0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        if action == 0:  # Safe
            reward = 1.0
        else:  # Gamble
            if self.np_random.random() < 0.90:
                reward = 10.0
            else:
                reward = -50.0  # The 'Trap' (10% frequency)

        # One-step environment
        terminated = True
        truncated = False
        return self.state, reward, terminated, truncated, {}


def register_envs():
    from gymnasium.envs.registration import register

    try:
        register(
            id="StructuralTrap-v0",
            entry_point="karakana.envs.structural_trap:StructuralTrapEnv",
            max_episode_steps=1,
        )
    except Exception:
        pass  # Already registered
