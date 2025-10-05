"""
Main experiment logger for tracking configurations, metrics, and checkpoints.
"""

import json
import torch
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from expsetup.config import Config
from expsetup.checkpoint_manager import CheckpointManager


class ExperimentLogger:
    """
    Main logger for experiment tracking.

    Handles configuration logging, metrics tracking, evaluation results,
    and checkpoint management for PyTorch experiments.

    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for all experiment outputs
        use_tensorboard: Whether to use TensorBoard logging
        resume_from: Path to experiment directory to resume from

    Example:
        >>> logger = ExperimentLogger("mnist_training", base_dir="./experiments")
        >>> logger.log_config({"lr": 0.001, "epochs": 10})
        >>> for epoch in range(10):
        ...     logger.log_metrics({"loss": 0.5}, step=epoch)
        ...     logger.save_checkpoint(model, optimizer, epoch=epoch)
    """

    def __init__(
        self,
        experiment_name: str,
        base_dir: Union[str, Path] = "./experiments",
        use_tensorboard: bool = True,
        resume_from: Optional[Union[str, Path]] = None,
    ):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)

        # Setup experiment directory
        if resume_from:
            self.exp_dir = Path(resume_from)
            if not self.exp_dir.exists():
                raise ValueError(f"Resume directory does not exist: {resume_from}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_dir = self.base_dir / f"{experiment_name}_{timestamp}"
            self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.config_dir = self.exp_dir / "config"

        for directory in [self.checkpoint_dir, self.log_dir, self.config_dir]:
            directory.mkdir(exist_ok=True)

        # Initialize components
        self.config: Optional[Config] = None
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)

        # TensorBoard writer
        self.tb_writer: Optional[SummaryWriter] = None
        if use_tensorboard:
            tb_dir = self.exp_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

        # Metrics tracking
        self.metrics_history: Dict[str, list] = {}
        self.metrics_file = self.log_dir / "metrics.csv"

        # Create metadata file
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Save experiment metadata."""
        metadata = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "exp_dir": str(self.exp_dir),
        }
        metadata_file = self.exp_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def log_config(
        self,
        config: Union[Dict[str, Any], Config, str, Path],
        name: str = "experiment_config"
    ) -> None:
        """
        Log experiment configuration.

        Args:
            config: Configuration as dict, Config object, or path to config file
            name: Name for the configuration
        """
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config(config, name=name)

        # Save configuration
        config_path = self.config_dir / "config.yaml"
        self.config.save(config_path)

        # Log to tensorboard if available
        if self.tb_writer:
            config_text = json.dumps(self.config._config, indent=2)
            self.tb_writer.add_text("config", config_text, 0)

        print(f"Configuration saved to {config_path}")

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log training or evaluation metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Current step/iteration/epoch
            prefix: Optional prefix for metric names (e.g., 'train/', 'val/')
        """
        # Convert tensors to Python scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            processed_metrics[key] = value

        # Add to history
        for key, value in processed_metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            if full_key not in self.metrics_history:
                self.metrics_history[full_key] = []
            self.metrics_history[full_key].append({
                'step': step,
                'value': value,
                'timestamp': datetime.now().isoformat()
            })

        # Log to TensorBoard
        if self.tb_writer and step is not None:
            for key, value in processed_metrics.items():
                full_key = f"{prefix}{key}" if prefix else key
                self.tb_writer.add_scalar(full_key, value, step)

        # Save to CSV
        self._save_metrics_csv()

    def _save_metrics_csv(self) -> None:
        """Save metrics history to CSV file."""
        rows = []
        for metric_name, history in self.metrics_history.items():
            for entry in history:
                rows.append({
                    'metric': metric_name,
                    'step': entry['step'],
                    'value': entry['value'],
                    'timestamp': entry['timestamp']
                })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.metrics_file, index=False)

    def log_evaluation(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        epoch: Optional[int] = None,
        split: str = "validation"
    ) -> None:
        """
        Log evaluation metrics.

        Args:
            metrics: Dictionary of evaluation metrics
            epoch: Current epoch number
            split: Data split name (e.g., 'validation', 'test')
        """
        prefix = f"{split}/"
        self.log_metrics(metrics, step=epoch, prefix=prefix)

        # Save separate evaluation results
        eval_file = self.log_dir / f"{split}_results.json"
        eval_data = {}
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)

        epoch_key = f"epoch_{epoch}" if epoch is not None else "final"
        eval_data[epoch_key] = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in metrics.items()
        }
        eval_data[epoch_key]['timestamp'] = datetime.now().isoformat()

        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, indent=2)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        **extra_state
    ) -> Path:
        """
        Save model checkpoint with metadata.

        Args:
            model: PyTorch model to save
            optimizer: Optional optimizer state
            scheduler: Optional learning rate scheduler state
            epoch: Current epoch
            step: Current training step
            metrics: Performance metrics at checkpoint time
            is_best: Whether this is the best model so far
            **extra_state: Additional state to save

        Returns:
            Path to saved checkpoint file
        """
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            metrics=metrics,
            config=self.config._config if self.config else None,
            is_best=is_best,
            **extra_state
        )

        print(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to map tensors to

        Returns:
            Dictionary containing checkpoint metadata
        """
        return self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

    def get_best_checkpoint(self, metric: str, mode: str = "max") -> Optional[Path]:
        """
        Get path to best checkpoint based on a metric.

        Args:
            metric: Metric name to compare
            mode: 'max' for higher is better, 'min' for lower is better

        Returns:
            Path to best checkpoint or None if no checkpoints found
        """
        return self.checkpoint_manager.get_best_checkpoint(metric=metric, mode=mode)

    def list_checkpoints(self) -> pd.DataFrame:
        """
        List all checkpoints with their metadata.

        Returns:
            DataFrame with checkpoint information
        """
        return self.checkpoint_manager.list_checkpoints()

    def cleanup_checkpoints(
        self,
        keep_best: int = 3,
        keep_last: int = 2,
        metric: str = "loss",
        mode: str = "min"
    ) -> None:
        """
        Clean up old checkpoints, keeping only the best and most recent.

        Args:
            keep_best: Number of best checkpoints to keep
            keep_last: Number of most recent checkpoints to keep
            metric: Metric to use for determining best checkpoints
            mode: 'max' or 'min' for the metric
        """
        self.checkpoint_manager.cleanup_checkpoints(
            keep_best=keep_best,
            keep_last=keep_last,
            metric=metric,
            mode=mode
        )

    def close(self) -> None:
        """Close logger and cleanup resources."""
        if self.tb_writer:
            self.tb_writer.close()

        # Save final metrics
        self._save_metrics_csv()
        print(f"Experiment data saved to {self.exp_dir}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
