"""
Karakana Trainers compatibility facade.
Delegates to:
  - karakana_engine.trainers (independent open-core package)
  - karakana_api.trainers (proprietary trading & execution trainers)
"""

import importlib
import sys

# Map submodule names to their actual package locations
_MODULE_MAP = {
    # Open-core engine trainers
    "spg": "karakana_engine.trainers.spg",
    "sgpo": "karakana_engine.trainers.sgpo",
    "grpo": "karakana_engine.trainers.grpo",
    "dual_cosine": "karakana.trainers.dual_cosine",
    "d5": "karakana.trainers.d5",
    "continuous_spg": "karakana_engine.trainers.continuous_spg",
    "continuous_sgpo": "karakana_engine.trainers.continuous_sgpo",
    "atari_trainer": "karakana_engine.trainers.atari_trainer",
    "dataset": "karakana_engine.trainers.dataset",
    "sb3": "karakana_engine.trainers.sb3",
    "grg": "karakana_engine.trainers.grg",
    # Proprietary SAaaS trainers
    "offline_trading": "karakana_api.trainers.offline_trading",
    "offline_trading_v2": "karakana_api.trainers.offline_trading_v2",
    "evx": "karakana_api.trainers.evx",
}

for _sub_name, _target_path in _MODULE_MAP.items():
    try:
        _mod = importlib.import_module(_target_path)
        sys.modules[f"karakana.trainers.{_sub_name}"] = _mod
        for _attr in dir(_mod):
            if not _attr.startswith("_"):
                globals()[_attr] = getattr(_mod, _attr)
    except ImportError:
        pass
