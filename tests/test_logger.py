"""Tests for ExperimentLogger class."""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from expsetup import ExperimentLogger, Config


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def test_experiment_logger_creation():
    """Test ExperimentLogger initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(
            experiment_name="test_exp",
            base_dir=tmpdir,
            use_tensorboard=False
        )

        assert logger.exp_dir.exists()
        assert logger.checkpoint_dir.exists()
        assert logger.log_dir.exists()
        assert logger.config_dir.exists()

        logger.close()


def test_log_config():
    """Test logging configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        config = {"lr": 0.001, "batch_size": 32}
        logger.log_config(config)

        config_file = logger.config_dir / "config.yaml"
        assert config_file.exists()

        logger.close()


def test_log_metrics():
    """Test logging metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        logger.log_metrics({"loss": 0.5, "accuracy": 0.85}, step=1)
        logger.log_metrics({"loss": 0.4, "accuracy": 0.90}, step=2)

        # Check metrics file
        metrics_file = logger.log_dir / "metrics.csv"
        assert metrics_file.exists()

        logger.close()


def test_log_evaluation():
    """Test logging evaluation metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        logger.log_evaluation(
            {"loss": 0.3, "accuracy": 0.92},
            epoch=5,
            split="validation"
        )

        eval_file = logger.log_dir / "validation_results.json"
        assert eval_file.exists()

        logger.close()


def test_save_and_load_checkpoint():
    """Test saving and loading checkpoints through logger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        checkpoint_path = logger.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            metrics={"accuracy": 0.95}
        )

        assert checkpoint_path.exists()

        # Load checkpoint
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())

        checkpoint_data = logger.load_checkpoint(
            checkpoint_path,
            new_model,
            new_optimizer
        )

        assert checkpoint_data["epoch"] == 5

        logger.close()


def test_get_best_checkpoint():
    """Test getting best checkpoint through logger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        model = SimpleModel()

        # Save multiple checkpoints
        logger.save_checkpoint(model, epoch=1, metrics={"accuracy": 0.8})
        logger.save_checkpoint(model, epoch=2, metrics={"accuracy": 0.9})
        logger.save_checkpoint(model, epoch=3, metrics={"accuracy": 0.85})

        best_checkpoint = logger.get_best_checkpoint(metric="accuracy", mode="max")
        assert best_checkpoint is not None

        logger.close()


def test_list_checkpoints():
    """Test listing checkpoints through logger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        model = SimpleModel()

        logger.save_checkpoint(model, epoch=1)
        logger.save_checkpoint(model, epoch=2)

        df = logger.list_checkpoints()
        assert len(df) >= 2

        logger.close()


def test_cleanup_checkpoints():
    """Test checkpoint cleanup through logger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        model = SimpleModel()

        # Save 10 checkpoints
        for i in range(10):
            logger.save_checkpoint(
                model,
                epoch=i,
                metrics={"loss": 1.0 - i * 0.05}
            )

        logger.cleanup_checkpoints(keep_best=3, keep_last=2, metric="loss", mode="min")

        df = logger.list_checkpoints()
        # Should have at most 5 checkpoints (might be less due to overlap)
        assert len(df) <= 6  # Including potential best_checkpoint.pt

        logger.close()


def test_context_manager():
    """Test using ExperimentLogger as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False) as logger:
            logger.log_config({"lr": 0.001})
            assert logger.exp_dir.exists()

        # Logger should be closed after context
        assert logger.metrics_file.exists()


def test_resume_from_directory():
    """Test resuming experiment from existing directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create initial experiment
        logger1 = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)
        exp_dir = logger1.exp_dir
        logger1.log_config({"lr": 0.001})
        logger1.close()

        # Resume from same directory
        logger2 = ExperimentLogger(
            "test_exp",
            base_dir=tmpdir,
            resume_from=exp_dir,
            use_tensorboard=False
        )

        assert logger2.exp_dir == exp_dir
        assert (logger2.config_dir / "config.yaml").exists()

        logger2.close()


def test_log_metrics_with_tensors():
    """Test logging metrics with PyTorch tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        # Log with tensor values
        logger.log_metrics({
            "loss": torch.tensor(0.5),
            "accuracy": torch.tensor(0.85)
        }, step=1)

        # Should convert tensors to scalars
        assert logger.metrics_file.exists()

        logger.close()


def test_metadata_file_creation():
    """Test that metadata file is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger("test_exp", base_dir=tmpdir, use_tensorboard=False)

        metadata_file = logger.exp_dir / "metadata.json"
        assert metadata_file.exists()

        logger.close()
