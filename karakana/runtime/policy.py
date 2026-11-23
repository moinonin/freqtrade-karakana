"""Inference bridge for Karakana offline trading policy artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

ACTION_LABELS = ("go_long", "go_short", "do_nothing")
ACTION_LABELS_V2 = ("go_long", "go_short")
SUPPORTED_ACTION_LABELS = (ACTION_LABELS, ACTION_LABELS_V2)
SCALE_INVARIANT_V1 = "scale_invariant_v1"
DEFAULT_LONG_ACTION_MARGIN = 1.3
DEFAULT_SHORT_ACTION_MARGIN = 0.07
DEFAULT_MIN_ACTION_MARGIN = DEFAULT_SHORT_ACTION_MARGIN
DEFAULT_STRATEGY_HEALTH_PATH = Path("user_data/reports/directional_strategy_health.csv")


def _transform_scale_invariant_v1(ask: float, bid: float, sma_compare: float, is_short: float) -> list[float]:
    mid_price = (ask + bid) / 2.0
    spread = bid - ask
    log_mid = math.copysign(math.log1p(abs(mid_price)), mid_price) if mid_price != 0.0 else 0.0
    rel_spread = max(-2.0, min(2.0, spread / max(abs(mid_price), 1e-8)))
    sma_clipped = max(-10.0, min(10.0, float(sma_compare)))
    return [float(log_mid), float(rel_spread), float(sma_clipped), float(is_short)]


def _torch_load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class KarakanaOfflinePolicy:
    """Load a Karakana offline policy and emit standard action labels."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        checkpoint_type: str = "best",
        side_consistent: bool = True,
        min_action_margin: float | None = None,
        min_long_action_margin: float | None = None,
        min_short_action_margin: float | None = None,
        threshold_governor_enabled: bool | None = None,
        strategy_health_path: str | Path | None = None,
    ):
        from karakana.trainers.offline_trading import OfflineTradingQPolicy

        self.model_dir = Path(model_dir)
        self.checkpoint_type = checkpoint_type
        self.side_consistent = bool(side_consistent)
        legacy_margin = None if min_action_margin is None else float(min_action_margin)
        if legacy_margin is None and "KARAKANA_MIN_ACTION_MARGIN" in os.environ:
            legacy_margin = float(os.environ["KARAKANA_MIN_ACTION_MARGIN"])
        self.min_long_action_margin = self._resolve_margin(
            explicit=min_long_action_margin,
            env_name="KARAKANA_MIN_LONG_ACTION_MARGIN",
            default=legacy_margin if legacy_margin is not None else DEFAULT_LONG_ACTION_MARGIN,
            label="min_long_action_margin",
        )
        self.min_short_action_margin = self._resolve_margin(
            explicit=min_short_action_margin,
            env_name="KARAKANA_MIN_SHORT_ACTION_MARGIN",
            default=legacy_margin if legacy_margin is not None else DEFAULT_SHORT_ACTION_MARGIN,
            label="min_short_action_margin",
        )
        self.min_action_margin = self.min_short_action_margin
        self.threshold_governor_enabled = self._resolve_bool(
            threshold_governor_enabled,
            env_name="KARAKANA_THRESHOLD_GOVERNOR_ENABLED",
            default=False,
        )
        self.strategy_health_path = Path(
            strategy_health_path or os.getenv("KARAKANA_STRATEGY_HEALTH_PATH", str(DEFAULT_STRATEGY_HEALTH_PATH))
        ).expanduser()
        self._strategy_health_snapshot = self._load_strategy_health_snapshot()

        checkpoint_name = "best_policy.pt" if checkpoint_type == "best" else "latest_policy.pt"
        checkpoint_path = self.model_dir / checkpoint_name
        preprocessing_path = self.model_dir / "offline_preprocessing.json"
        manifest_path = self.model_dir / "manifest.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Karakana checkpoint not found: {checkpoint_path}")
        if not preprocessing_path.exists():
            raise FileNotFoundError(f"Karakana preprocessing contract not found: {preprocessing_path}")

        self.preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        self.action_labels = tuple(self.preprocessing.get("action_labels", ACTION_LABELS))
        if self.action_labels not in SUPPORTED_ACTION_LABELS:
            raise ValueError(
                f"Unsupported action labels {self.action_labels}; expected one of {SUPPORTED_ACTION_LABELS}"
            )
        self.feature_transform = self.preprocessing.get("feature_transform")
        self.observation_layout = tuple(self.preprocessing.get("observation_layout", ()))

        means = self.preprocessing.get("means")
        scales = self.preprocessing.get("scales")
        if not isinstance(means, list) or not isinstance(scales, list) or len(means) != 4 or len(scales) != 4:
            raise ValueError("offline_preprocessing.json must contain four means and four scales")

        self.means = torch.tensor(means, dtype=torch.float32)
        self.scales = torch.tensor(scales, dtype=torch.float32)
        if not torch.isfinite(self.means).all() or not torch.isfinite(self.scales).all():
            raise ValueError("Karakana preprocessing means/scales must be finite")
        if torch.any(self.scales <= 0):
            raise ValueError("Karakana preprocessing scales must be positive")

        self.policy = OfflineTradingQPolicy(action_count=len(self.action_labels))
        self.policy.load_state_dict(_torch_load_state_dict(checkpoint_path))
        self.policy.eval()

    @staticmethod
    def _resolve_margin(*, explicit: float | None, env_name: str, default: float, label: str) -> float:
        value = explicit
        if value is None:
            value = float(os.getenv(env_name, default))
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
        return value

    @staticmethod
    def _resolve_bool(value: bool | None, *, env_name: str, default: bool) -> bool:
        if value is not None:
            return bool(value)
        raw = os.getenv(env_name)
        if raw is None:
            return bool(default)
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) and parsed >= 0.0 else default

    def _load_strategy_health_snapshot(self) -> dict[str, Any] | None:
        if not self.threshold_governor_enabled:
            return None
        if not self.strategy_health_path.exists():
            return None
        last_row: dict[str, str] | None = None
        with self.strategy_health_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                last_row = row
        if not last_row:
            return None
        long_threshold = self._safe_float(last_row.get("effective_long_threshold"), self.min_long_action_margin)
        short_threshold = self._safe_float(last_row.get("effective_short_threshold"), self.min_short_action_margin)
        return {
            **last_row,
            "effective_long_threshold": long_threshold,
            "effective_short_threshold": short_threshold,
            "source_path": str(self.strategy_health_path),
        }

    def _validate_state(self, ask: float, bid: float, sma_compare: float | int, is_short: int) -> list[float]:
        state = [float(ask), float(bid), float(sma_compare), float(is_short)]
        if len(state) != 4 or not all(math.isfinite(value) for value in state):
            raise ValueError("Karakana inference requires four finite state values")
        if int(state[3]) not in (0, 1):
            raise ValueError("is_short must be 0 or 1")
        if self.feature_transform == SCALE_INVARIANT_V1:
            short_feat = 0.0 if len(self.action_labels) == 2 else float(int(state[3]))
            return _transform_scale_invariant_v1(state[0], state[1], state[2], short_feat)
        state[2] = float(int(state[2]))
        state[3] = float(int(state[3]))
        return state

    def _directional_margin(self, q_values: torch.Tensor, is_short: int) -> dict[str, float | str]:
        directional_action = "go_short" if int(is_short) else "go_long"
        directional_q = float(q_values[self.action_labels.index(directional_action)].item())
        if "do_nothing" in self.action_labels:
            baseline_q = float(q_values[self.action_labels.index("do_nothing")].item())
        else:
            opposite_action = "go_long" if int(is_short) else "go_short"
            baseline_q = float(q_values[self.action_labels.index(opposite_action)].item())
        return {
            "directional_action": directional_action,
            "directional_q": directional_q,
            "do_nothing_q": baseline_q,
            "action_margin": directional_q - baseline_q,
        }

    def _threshold_for_side(self, is_short: int) -> float:
        if self._strategy_health_snapshot:
            key = "effective_short_threshold" if int(is_short) else "effective_long_threshold"
            return float(self._strategy_health_snapshot[key])
        return self.min_short_action_margin if int(is_short) else self.min_long_action_margin

    def _select_action(self, q_values: torch.Tensor, is_short: int) -> str:
        if not self.side_consistent:
            return self.action_labels[int(torch.argmax(q_values).item())]
        margin_info = self._directional_margin(q_values, is_short)
        if float(margin_info["action_margin"]) >= self._threshold_for_side(is_short):
            return str(margin_info["directional_action"])
        return "do_nothing"

    def predict(self, ask: float, bid: float, sma_compare: int, is_short: int) -> dict[str, Any]:
        raw_state = self._validate_state(ask, bid, sma_compare, is_short)
        state_tensor = torch.tensor(raw_state, dtype=torch.float32)
        normalized_state = (state_tensor - self.means) / self.scales

        with torch.inference_mode():
            q_values = self.policy(normalized_state.unsqueeze(0)).squeeze(0)

        action_values = {label: float(q_values[index].item()) for index, label in enumerate(self.action_labels)}
        selected_action = self._select_action(q_values, int(is_short))
        margin_info = self._directional_margin(q_values, int(is_short))
        allowed_actions = ["go_short", "do_nothing"] if int(is_short) else ["go_long", "do_nothing"]
        required_margin = self._threshold_for_side(int(is_short))

        return {
            "state_recognized": True,
            "backend": "karakana_offline",
            "checkpoint_type": self.checkpoint_type,
            "side_consistent": self.side_consistent,
            "min_action_margin": self.min_action_margin,
            "min_long_action_margin": self.min_long_action_margin,
            "min_short_action_margin": self.min_short_action_margin,
            "threshold_governor_enabled": self.threshold_governor_enabled,
            "threshold_governor_active": self._strategy_health_snapshot is not None,
            "strategy_health_path": str(self.strategy_health_path),
            "strategy_health_snapshot": self._strategy_health_snapshot,
            "raw_state": raw_state,
            "normalized_state": [float(value) for value in normalized_state.tolist()],
            "action": selected_action,
            "selected_action": selected_action,
            "allowed_actions": allowed_actions if self.side_consistent else list(self.action_labels),
            "directional_action": margin_info["directional_action"],
            "directional_q": margin_info["directional_q"],
            "do_nothing_q": margin_info["do_nothing_q"],
            "action_margin": margin_info["action_margin"],
            "required_action_margin": required_margin,
            "margin_passed": bool(float(margin_info["action_margin"]) >= required_margin),
            "q_values": action_values,
            "conviction": float(max(action_values.values()) - min(action_values.values())),
        }
