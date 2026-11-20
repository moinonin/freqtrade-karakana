"""Anchored Dual-Cosine Dataset Trainer.

Extends ``DatasetTrainer`` so BYOD (non-gym) CSV training uses the standalone
``AnchoredDualCosineSGOLoss`` objective while preserving 100% of the portable
``DatasetPolicy``, preprocessing contract, and bundle export pipeline.
"""

from __future__ import annotations

import torch
from karakana_engine.trainers.dataset import DatasetTrainer

from karakana.metrics import evaluate_structural_geometry
from karakana.torch_dual_cosine import AnchoredDualCosineSGOLoss


class DualCosineDatasetTrainer(DatasetTrainer):
    """DatasetTrainer driven by AnchoredDualCosineSGOLoss."""

    def __init__(
        self,
        dataset_csv_b64: str,
        target_col: str,
        task_type: str = "classification",
        lr: float = 0.01,
        k: int = 2,
        utility_power: float = 1.1,
        w_alpha: float = 1.0,
        w_coupling: float = 0.1,
        margin: float = 0.0,
        w_hinge: float = 1.0,
        ranking_profile: str = "balanced",
        steering_enabled: bool = True,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__(
            dataset_csv_b64=dataset_csv_b64,
            target_col=target_col,
            task_type=task_type,
            lambda_sgo=0.0,
            lr=lr,
            loss_module="dual_cosine",
            ranking_profile=ranking_profile,
            steering_enabled=steering_enabled,
            random_seed=random_seed,
            **kwargs,
        )
        # Dominance rule: w_coupling <= 0.1 * w_alpha
        w_a = float(w_alpha)
        w_c = min(float(w_coupling), 0.1 * w_a) if w_a > 0 else float(w_coupling)
        self.anchored_loss_fn = AnchoredDualCosineSGOLoss(
            k=max(2, int(k)),
            p=float(utility_power),
            w_alpha=w_a,
            w_coupling=w_c,
            margin=float(margin),
            w_hinge=float(w_hinge),
        )

    def train_epoch(self, batch_size: int = 32):
        """Run one training epoch using AnchoredDualCosineSGOLoss."""
        if self._sample_weights is not None:
            indices = torch.multinomial(
                self._sample_weights,
                num_samples=self.X.size(0),
                replacement=True,
                generator=self._sampling_generator,
            )
        else:
            indices = torch.randperm(
                self.X.size(0),
                generator=self._sampling_generator,
            )

        epoch_loss = 0.0
        n_batches = 0
        self.policy.train()

        for i in range(0, self.X.size(0), batch_size):
            idx = indices[i : i + batch_size]
            batch_X, batch_y = self.X[idx], self.y[idx]
            if batch_X.size(0) < 2:
                continue

            self.optimizer.zero_grad()
            outputs = self.policy(batch_X)

            if self.task_type == "classification":
                true_logits = outputs.gather(1, batch_y.unsqueeze(1)).squeeze(1)
                competing_logits = outputs.clone()
                competing_logits.scatter_(1, batch_y.unsqueeze(1), float("-inf"))
                scores = true_logits - competing_logits.max(dim=1).values
            else:
                absolute_error = torch.abs(outputs - batch_y).view(-1)
                scores = absolute_error.mean().detach() - absolute_error

            total_loss = self.anchored_loss_fn(scores)
            total_loss.backward()
            self.optimizer.step()

            epoch_loss += float(total_loss.item())
            n_batches += 1

            with torch.no_grad():
                pos = scores[scores > 0].cpu().numpy()
                neg = scores[scores <= 0].cpu().numpy()
                if pos.size > 0 and neg.size > 0:
                    res = evaluate_structural_geometry(pos, neg)
                    self._last_alpha = max(0.0, min(1.0, float(res.get("alpha_health", 1.0))))
                elif pos.size > 0 and neg.size == 0:
                    self._last_alpha = 1.0

        if self.steering_enabled:
            self.grg.step(self._last_alpha)

        val_metrics = self.validation_metrics()
        reward = float(val_metrics.get(self.selection_metric_name, 0.0))
        avg_loss = epoch_loss / max(1, n_batches)
        return reward, avg_loss
