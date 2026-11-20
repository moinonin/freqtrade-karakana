import gymnasium as gym
import numpy as np


class MountainCarSGOEnv(gym.Wrapper):
    """
    SGO-optimized MountainCar environment wrapper.
    1. Normalizes observation: position to [-1, 1] and velocity to [-1, 1].
    2. Action persistence / frame skip (default 4): converts single-tick jitter into sustained thrust.
    3. Potential-based mechanical energy reward shaping: kinetic energy + potential height + momentum alignment.
    4. Preserves true underlying raw environment reward in info['raw_reward'].
    """

    def __init__(self, env: gym.Env, frame_skip: int = 4):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.raw_reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_reward = 0.0
        return self._normalize_obs(obs), info

    def step(self, action: int):
        total_raw_reward = 0.0
        done = False
        truncated = False
        info = {}

        for _ in range(self.frame_skip):
            obs, r, d, t, info = self.env.step(action)
            total_raw_reward += r
            if d or t:
                done = d
                truncated = t
                break

        self.raw_reward += total_raw_reward
        info["raw_reward"] = self.raw_reward

        pos, vel = float(obs[0]), float(obs[1])
        height = float(np.sin(3.0 * pos))

        # Momentum alignment: pushing in the direction of velocity accelerates energy buildup
        vel_sign = 1.0 if vel > 0.001 else (-1.0 if vel < -0.001 else 0.0)
        act_dir = 1.0 if action == 2 else (-1.0 if action == 0 else 0.0)
        alignment = 1.0 if (vel_sign * act_dir > 0) else (-0.5 if (vel_sign * act_dir < 0) else 0.0)

        # Potential energy + kinetic energy shaping
        shaped_reward = total_raw_reward + 20.0 * abs(vel) + 2.0 * (height + 1.0) + 2.0 * alignment

        if done and pos >= 0.5:
            shaped_reward += 100.0  # Goal completion upside bonus
            info["is_success"] = True
        else:
            info["is_success"] = False

        return self._normalize_obs(obs), shaped_reward, done, truncated, info

    @staticmethod
    def _normalize_obs(obs: np.ndarray) -> np.ndarray:
        """
        Normalize position [-1.2, 0.6] and velocity [-0.07, 0.07] to [-1, 1].
        Prevents position from drowning out velocity by 100x.
        """
        p_norm = (float(obs[0]) - (-0.3)) / 0.9
        v_norm = float(obs[1]) / 0.07
        return np.array([p_norm, v_norm], dtype=np.float32)
