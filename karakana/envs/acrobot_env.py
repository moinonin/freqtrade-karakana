import gymnasium as gym
import numpy as np


class AcrobotSGOEnv(gym.Wrapper):
    """
    SGO-optimized Acrobot environment wrapper.

    1. Adds potential-based reward shaping over the free-end (tip) height, so
       the per-step reward varies before the first successful swing-up.
    2. Preserves the true underlying raw environment reward in info['raw_reward'].
    3. Exposes the tip height in info['height'] and flags goal reach in
       info['is_success'].

    Why this exists: Acrobot pays -1 on every tick and 0 only on the terminal
    step, so a mean-centered trajectory objective (Dual-Cosine SGO) sees
    c_t = r_t - mean(r) = 0 for every step of a failed episode; the gradient is
    identically zero. The shaping term

        r' = r + gamma * phi(s') - phi(s)
        phi(s) = -cos(theta1) - cos(theta1 + theta2)   (tip height)

    is potential-based, so it leaves the optimal policy unchanged
    (Ng, Harada & Russell, 1999) while restoring a live reward carrier. The
    potential is exactly Gymnasium's terminal predicate (`phi(s) > 1.0`), so the
    shaping reward is largest when the free end crosses the goal height.
    Observations and action space are untouched.
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

        height = self._potential(obs)
        shaped_reward = raw + self.gamma * height - self._prev_potential
        self._prev_potential = height

        info["raw_reward"] = self.raw_reward
        info["height"] = height
        info["is_success"] = bool(done and height > 1.0)

        return obs, shaped_reward, done, truncated, info

    @staticmethod
    def _potential(obs: np.ndarray) -> float:
        """Tip height: -cos(theta1) - cos(theta1 + theta2)."""
        cos1, sin1, cos2, sin2 = (float(v) for v in obs[:4])
        return -cos1 - (cos1 * cos2 - sin1 * sin2)
