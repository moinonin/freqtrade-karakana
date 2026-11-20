import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StructuralTradingEnv(gym.Env):
    """
    SGO-Trading-v0: A custom trading environment designed to benchmark
    Structural Geometry Optimization (SGO) against high tail-risk events.

    Actions:
    0: Stay Flat / Safe (small steady yield, reward = 0.5)
    1: Take Position (fragile: reward = +5.0 92% of the time, -50.0 8% of the time)

    Expected Value (EV):
    Action 0: 0.5
    Action 1: (0.92 * 5) + (0.08 * -50) = 4.6 - 4.0 = +0.6

    A standard RL agent maximizing raw expected value will choose Action 1 (0.6 > 0.5).
    However, Action 1 is structurally fragile (deep tail-risk loss of -50.0).
    Under SGO regularization, the agent learns to prioritize Action 0 because the
    SGO loss penalizes the extreme negative cofactors of Action 1's payoff distribution.
    """

    def __init__(self, seed=None):
        super().__init__()
        # Discrete actions: 0 = Safe Flat, 1 = Fragile Position
        self.action_space = spaces.Discrete(2)
        # Observations: [position_flag, last_payoff_scaled, normalized_time]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 10
        self.seed_val = seed
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        if seed is None and self.seed_val is not None:
            seed = self.seed_val
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1

        if action == 0:  # Stay Flat (Safe yield)
            reward = 0.5
            self.state = np.array([0.0, 0.5, self.step_count / self.max_steps], dtype=np.float32)
        else:  # Take Position (High reward, asymmetric downside risk)
            if self.np_random.random() < 0.92:
                reward = 5.0  # Normal market regime gain
            else:
                reward = -50.0  # Black Swan crash risk (8% frequency)
            self.state = np.array([1.0, reward / 50.0, self.step_count / self.max_steps], dtype=np.float32)

        terminated = self.step_count >= self.max_steps
        truncated = False
        return self.state, reward, terminated, truncated, {}


def register_trading_env():
    from gymnasium.envs.registration import register

    try:
        register(
            id="SGO-Trading-v0",
            entry_point="karakana.envs.trading_env:StructuralTradingEnv",
            max_episode_steps=10,
        )
        print("[SGO] Custom SGO-Trading-v0 environment registered successfully!", flush=True)
    except Exception:
        pass  # Already registered
