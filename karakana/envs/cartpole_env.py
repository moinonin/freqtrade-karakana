import gymnasium as gym
import numpy as np


class CartPoleSGOEnv(gym.Wrapper):
    """
    SGO-optimized CartPole environment wrapper.

    1. Adds potential-based reward shaping over the normalized cart position and
       pole angle, so the per-step reward varies with the state.
    2. Preserves the true underlying raw environment reward in info['raw_reward'].
    3. Flags time-limit survival in info['is_success'].

    Why this exists: CartPole pays +1 on every tick, so a mean-centered
    trajectory objective (Dual-Cosine SGO) sees c_t = r_t - mean(r) = 0 for
    every step and its gradient is identically zero. The shaping term

        r' = r + gamma * phi(s') - phi(s)
        phi(s) = -(|x| / x_threshold + |theta| / theta_threshold)

    is potential-based, so it leaves the optimal policy unchanged
    (Ng, Harada & Russell, 1999) while restoring a live reward carrier.
    Observations and action space are untouched, so existing checkpoints remain
    compatible with the simulator and hosted inference.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = float(gamma)
        self.raw_reward = 0.0
        self._prev_potential = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_reward = 0.0
        self._prev_potential = self._potential(obs)
        return obs, info

    def step(self, action: int):
        obs, reward, done, truncated, info = self.env.step(action)
        raw = float(reward)
        self.raw_reward += raw

        potential = self._potential(obs)
        shaped_reward = raw + self.gamma * potential - self._prev_potential
        self._prev_potential = potential

        info["raw_reward"] = self.raw_reward
        # CartPole terminates when the pole falls or the cart leaves the track;
        # surviving to the time limit is the only "success" event it has.
        info["is_success"] = bool(truncated and not done)

        return obs, shaped_reward, done, truncated, info

    @staticmethod
    def _potential(obs: np.ndarray) -> float:
        """Negative normalized distance from the upright/centered state."""
        x, _, theta, _ = (float(v) for v in obs)
        return -(abs(x) / 2.4 + abs(theta) / 0.2095)
