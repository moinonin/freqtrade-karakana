"""Register gymnasium environments including Atari and MuJoCo via ale_py."""


def _register_ale_envs():
    """Register ALE/Atari environments with gymnasium."""
    try:
        from ale_py.registration import register_v0_v4_envs, register_v5_envs

        register_v0_v4_envs()
        register_v5_envs()
    except ImportError:
        pass


_register_ale_envs()

from karakana.envs.acrobot_env import AcrobotSGOEnv
from karakana.envs.cartpole_env import CartPoleSGOEnv
from karakana.envs.mountain_car_env import MountainCarSGOEnv

__all__ = ["AcrobotSGOEnv", "CartPoleSGOEnv", "MountainCarSGOEnv"]
