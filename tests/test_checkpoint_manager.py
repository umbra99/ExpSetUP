"""Tests for CheckpointManager class."""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from expsetup import CheckpointManager


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


def test_checkpoint_manager_creation():
    """Test CheckpointManager creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        assert manager.checkpoint_dir.exists()
        assert manager.metadata_file.exists()


def test_save_checkpoint():
    """Test saving a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()

        checkpoint_path = manager.save_checkpoint(
            model=model,
            epoch=5,
            metrics={"accuracy": 0.95}
        )

        assert checkpoint_path.exists()
        assert checkpoint_path.name in manager.metadata


def test_load_checkpoint():
    """Test loading a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            epoch=5,
            metrics={"accuracy": 0.95}
        )

        # Load checkpoint
        new_model = SimpleModel()
        checkpoint_data = manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model
        )

        assert checkpoint_data["epoch"] == 5
        assert checkpoint_data["metrics"]["accuracy"] == 0.95


def test_best_checkpoint():
    """Test getting best checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()

        # Save multiple checkpoints
        manager.save_checkpoint(model, epoch=1, metrics={"accuracy": 0.8})
        manager.save_checkpoint(model, epoch=2, metrics={"accuracy": 0.9})
        manager.save_checkpoint(model, epoch=3, metrics={"accuracy": 0.85})

        # Get best by accuracy (max)
        best_path = manager.get_best_checkpoint(metric="accuracy", mode="max")
        assert best_path is not None

        # Load and check
        checkpoint = torch.load(best_path)
        assert checkpoint["metrics"]["accuracy"] == 0.9


def test_best_checkpoint_min_mode():
    """Test getting best checkpoint with min mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()

        # Save checkpoints with loss metric
        manager.save_checkpoint(model, epoch=1, metrics={"loss": 0.5})
        manager.save_checkpoint(model, epoch=2, metrics={"loss": 0.3})
        manager.save_checkpoint(model, epoch=3, metrics={"loss": 0.4})

        # Get best by loss (min)
        best_path = manager.get_best_checkpoint(metric="loss", mode="min")
        checkpoint = torch.load(best_path)
        assert checkpoint["metrics"]["loss"] == 0.3


def test_list_checkpoints():
    """Test listing checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()

        # Save multiple checkpoints
        manager.save_checkpoint(model, epoch=1, metrics={"accuracy": 0.8})
        manager.save_checkpoint(model, epoch=2, metrics={"accuracy": 0.9})

        df = manager.list_checkpoints()
        assert len(df) >= 2
        assert "epoch" in df.columns
        assert "metric_accuracy" in df.columns


def test_cleanup_checkpoints():
    """Test cleanup of old checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()

        # Save 10 checkpoints
        for i in range(10):
            manager.save_checkpoint(
                model,
                epoch=i,
                metrics={"accuracy": 0.7 + i * 0.01}
            )

        # Cleanup: keep 3 best and 2 most recent
        manager.cleanup_checkpoints(
            keep_best=3,
            keep_last=2,
            metric="accuracy",
            mode="max"
        )

        # Should have at most 5 checkpoints (some might overlap)
        remaining = len([f for f in Path(tmpdir).glob("checkpoint_*.pt")])
        assert remaining <= 5


def test_get_latest_checkpoint():
    """Test getting the latest checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()

        # Save multiple checkpoints
        manager.save_checkpoint(model, epoch=1)
        import time
        time.sleep(0.1)
        latest_path = manager.save_checkpoint(model, epoch=2)

        retrieved_latest = manager.get_latest_checkpoint()
        assert retrieved_latest == latest_path


def test_save_checkpoint_with_optimizer():
    """Test saving checkpoint with optimizer state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5
        )

        # Load and verify
        checkpoint = torch.load(checkpoint_path)
        assert "optimizer_state_dict" in checkpoint


def test_is_best_flag():
    """Test is_best flag and best_checkpoint.pt creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = SimpleModel()

        manager.save_checkpoint(
            model,
            epoch=5,
            metrics={"accuracy": 0.95},
            is_best=True
        )

        best_checkpoint = Path(tmpdir) / "best_checkpoint.pt"
        assert best_checkpoint.exists()
