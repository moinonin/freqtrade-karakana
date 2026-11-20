import base64

import numpy as np
import pandas as pd
import torch
from karakana.trainers.dataset import DatasetTrainer


def _csv_b64(frame: pd.DataFrame) -> str:
    return base64.b64encode(frame.to_csv(index=False).encode("utf-8")).decode("ascii")


def test_dataset_trainer_filters_text_and_prediction_output_columns():
    rows = 20
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.1, 0.9, rows),
            "feature_b": np.linspace(1.0, 3.5, rows),
            "filename": [f"sample_{index}.jpg" for index in range(rows)],
            "probability": np.linspace(0.1, 0.95, rows),
            "predicted_class": ["human"] * 10 + ["ai"] * 10,
            "confidence": np.linspace(0.8, 0.99, rows),
            "is_ai": [False] * 10 + [True] * 10,
        }
    )

    trainer = DatasetTrainer(_csv_b64(frame), target_col="is_ai")

    assert trainer.feature_columns == ["feature_a", "feature_b"]
    assert set(trainer.ignored_columns) == {
        "filename",
        "probability",
        "predicted_class",
        "confidence",
    }
    assert trainer.X.dtype.is_floating_point
    assert np.isfinite(trainer.X.numpy()).all()


def test_dataset_trainer_excludes_exported_dataframe_index_columns():
    frame = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(40),
            "Index": np.arange(40),
            "ticket_id": [f"TICKET_{i}" for i in range(40)],
            "feature_a": np.linspace(0.0, 1.0, 40),
            "feature_b": np.linspace(10.0, 20.0, 40),
            "target": ["human"] * 20 + ["ai"] * 20,
        }
    )

    trainer = DatasetTrainer(_csv_b64(frame), target_col="target")

    assert trainer.feature_columns == ["feature_a", "feature_b"]
    assert "Unnamed: 0" in trainer.ignored_columns
    assert "Index" in trainer.ignored_columns
    assert "ticket_id" in trainer.ignored_columns


def test_dataset_trainer_auto_vectorizes_text_and_categoricals():
    rows = 40
    plans = ["Enterprise", "Pro", "Free", "Starter", "Business"] * 8
    subjects = [f"unique issue report number {i} for support team" for i in range(rows)]
    frame = pd.DataFrame(
        {
            "Index": np.arange(rows),
            "plan": plans,
            "subject": subjects,
            "target": ["high"] * 20 + ["low"] * 20,
        }
    )

    trainer = DatasetTrainer(_csv_b64(frame), target_col="target", auto_vectorize=True, max_text_features=16)

    assert "Index" not in trainer.feature_columns
    assert any("plan_" in col for col in trainer.feature_columns)
    assert any("subject_" in col for col in trainer.feature_columns)
    loss, alpha = trainer.train_epoch(batch_size=16)
    assert np.isfinite(loss)
    assert 0.0 <= alpha <= 1.0


def test_dataset_trainer_balances_classification_sampling_reproducibly():
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, 100),
            "target": ["majority"] * 90 + ["minority"] * 10,
        }
    )
    first = DatasetTrainer(_csv_b64(frame), target_col="target", random_seed=17)
    second = DatasetTrainer(_csv_b64(frame), target_col="target", random_seed=17)

    assert first.sampling_strategy == "inverse_frequency_oversampling"
    assert first.training_class_counts == {"majority": 63, "minority": 7}

    first_indices = torch.multinomial(
        first._sample_weights,
        num_samples=7000,
        replacement=True,
        generator=first._sampling_generator,
    )
    second_indices = torch.multinomial(
        second._sample_weights,
        num_samples=7000,
        replacement=True,
        generator=second._sampling_generator,
    )
    assert torch.equal(first_indices, second_indices)

    sampled_counts = torch.bincount(first.y[first_indices], minlength=2)
    sampled_ratio = sampled_counts.min().item() / sampled_counts.max().item()
    assert sampled_ratio > 0.9


def test_dataset_regression_uses_seeded_shuffle_without_resampling():
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, 40),
            "target": np.linspace(10.0, 20.0, 40),
        }
    )

    trainer = DatasetTrainer(
        _csv_b64(frame),
        target_col="target",
        task_type="regression",
        random_seed=11,
    )

    assert trainer.sampling_strategy == "seeded_shuffle"
    assert trainer._sample_weights is None
    transform = trainer.preprocessing_contract()["target_transform"]
    assert transform["method"] == "standard"
    assert transform["scale"] > 0


def test_dataset_regression_decodes_metrics_to_original_target_units():
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, 80),
            "target": np.linspace(0.65, 1.10, 80),
        }
    )
    trainer = DatasetTrainer(
        _csv_b64(frame),
        target_col="target",
        task_type="regression",
        lambda_sgo=0.05,
        steering_enabled=False,
        random_seed=19,
    )

    for _ in range(40):
        trainer.train_epoch(batch_size=32)

    metrics = trainer.validation_metrics()
    assert metrics["mae"] < 0.08
    assert 0.6 < metrics["target_range"][0] < 0.8
    assert 1.0 < metrics["target_range"][1] < 1.2


def test_dataset_trainer_runs_classification_epoch_with_mixed_csv():
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.0, 1.0, 40),
            "feature_b": np.linspace(10.0, 20.0, 40),
            "filename": [f"sample_{index}.jpg" for index in range(40)],
            "target": ["human"] * 20 + ["ai"] * 20,
        }
    )

    trainer = DatasetTrainer(_csv_b64(frame), target_col="target", steering_enabled=False)
    loss, alpha = trainer.train_epoch(batch_size=16)

    assert np.isfinite(loss)
    assert np.isfinite(alpha)
    assert 0.0 <= alpha <= 1.0


def test_dataset_trainer_excludes_exact_classification_target_proxy():
    rows = 20
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.1, 0.9, rows),
            "is_ai": [False] * 10 + [True] * 10,
            "ground_truth": ["human"] * 10 + ["ai"] * 10,
        }
    )

    trainer = DatasetTrainer(_csv_b64(frame), target_col="ground_truth")

    assert trainer.feature_columns == ["feature_a"]
    assert "is_ai" in trainer.ignored_columns


def test_dataset_moment_loss_uses_selected_ranking_profile():
    rows = 20
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(0.1, 0.9, rows),
            "target": ["human"] * 10 + ["ai"] * 10,
        }
    )

    trainer = DatasetTrainer(
        _csv_b64(frame),
        target_col="target",
        loss_module="moment",
        ranking_profile="aggressive",
    )

    assert trainer.ranking_profile == "aggressive"
    assert trainer.sgo_loss_fn.metric_key == "ranking_score"
    assert trainer.sgo_loss_fn.ranking_profile == "aggressive"


def test_dataset_preprocessing_is_fit_on_training_split_only():
    frame = pd.DataFrame(
        {
            "feature_a": np.arange(40, dtype=float),
            "feature_b": np.linspace(10.0, 20.0, 40),
            "target": ["human"] * 20 + ["ai"] * 20,
        }
    )

    trainer = DatasetTrainer(_csv_b64(frame), target_col="target", random_seed=7)

    assert trainer.split_sizes() == {"train": 28, "validation": 6, "test": 6}
    assert trainer.X.shape[0] == 28
    assert trainer.validation_X.shape[0] == 6
    assert trainer.test_X.shape[0] == 6
    assert trainer.preprocessing_contract()["feature_columns"] == ["feature_a", "feature_b"]


def test_dataset_validation_and_test_metrics_are_task_appropriate():
    frame = pd.DataFrame(
        {
            "feature_a": np.linspace(-1.0, 1.0, 40),
            "feature_b": np.linspace(2.0, 4.0, 40),
            "target": ["left"] * 20 + ["right"] * 20,
        }
    )
    trainer = DatasetTrainer(_csv_b64(frame), target_col="target", steering_enabled=False)
    trainer.train_epoch(batch_size=16)

    validation = trainer.validation_metrics()
    test = trainer.test_metrics()

    assert {"accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix"} <= validation.keys()
    assert {"accuracy", "balanced_accuracy", "macro_f1", "confusion_matrix"} <= test.keys()
