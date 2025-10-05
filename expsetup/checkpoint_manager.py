"""
Checkpoint management with performance-based utilities.
"""

import torch
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints with metadata and performance tracking.

    Provides utilities for saving, loading, and managing checkpoints based
    on performance metrics.

    Args:
        checkpoint_dir: Directory to store checkpoints

    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> manager.save_checkpoint(model, optimizer, epoch=5, metrics={"acc": 0.92})
        >>> best_ckpt = manager.get_best_checkpoint(metric="acc", mode="max")
    """

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"

        # Load existing metadata
        self.metadata: Dict[str, Dict[str, Any]] = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        checkpoint_name: Optional[str] = None,
        **extra_state
    ) -> Path:
        """
        Save a checkpoint with comprehensive metadata.

        Args:
            model: PyTorch model
            optimizer: Optimizer state
            scheduler: LR scheduler state
            epoch: Current epoch
            step: Current step
            metrics: Performance metrics
            config: Training configuration
            is_best: Flag for best model
            checkpoint_name: Custom checkpoint name
            **extra_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if epoch is not None:
                checkpoint_name = f"checkpoint_epoch{epoch}_{timestamp}.pt"
            elif step is not None:
                checkpoint_name = f"checkpoint_step{step}_{timestamp}.pt"
            else:
                checkpoint_name = f"checkpoint_{timestamp}.pt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'config': config or {},
        }

        # Add optimizer state
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Add extra state
        checkpoint.update(extra_state)

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update metadata
        self.metadata[checkpoint_name] = {
            'path': str(checkpoint_path),
            'epoch': epoch,
            'step': step,
            'timestamp': checkpoint['timestamp'],
            'metrics': metrics or {},
            'is_best': is_best,
            'size_bytes': checkpoint_path.stat().st_size
        }

        # Save best checkpoint copy
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            shutil.copy2(checkpoint_path, best_path)
            self.metadata['best_checkpoint.pt'] = self.metadata[checkpoint_name].copy()
            self.metadata['best_checkpoint.pt']['path'] = str(best_path)

        # Save metadata
        self._save_metadata()

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
        Load a checkpoint and restore model/optimizer/scheduler states.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to map tensors to

        Returns:
            Checkpoint metadata and extra state
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Step: {checkpoint.get('step', 'N/A')}")
        if 'metrics' in checkpoint and checkpoint['metrics']:
            print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint

    def get_best_checkpoint(
        self,
        metric: str,
        mode: str = "max"
    ) -> Optional[Path]:
        """
        Get the best checkpoint based on a specific metric.

        Args:
            metric: Metric name to use for comparison
            mode: 'max' for higher is better, 'min' for lower is better

        Returns:
            Path to best checkpoint or None
        """
        best_checkpoint = None
        best_value = float('-inf') if mode == "max" else float('inf')

        for ckpt_name, ckpt_meta in self.metadata.items():
            if ckpt_name == 'best_checkpoint.pt':
                continue

            metrics = ckpt_meta.get('metrics', {})
            if metric not in metrics:
                continue

            value = metrics[metric]

            if mode == "max" and value > best_value:
                best_value = value
                best_checkpoint = ckpt_meta['path']
            elif mode == "min" and value < best_value:
                best_value = value
                best_checkpoint = ckpt_meta['path']

        return Path(best_checkpoint) if best_checkpoint else None

    def list_checkpoints(self) -> pd.DataFrame:
        """
        List all checkpoints with their metadata.

        Returns:
            DataFrame containing checkpoint information
        """
        if not self.metadata:
            return pd.DataFrame()

        rows = []
        for ckpt_name, ckpt_meta in self.metadata.items():
            row = {
                'name': ckpt_name,
                'epoch': ckpt_meta.get('epoch'),
                'step': ckpt_meta.get('step'),
                'timestamp': ckpt_meta.get('timestamp'),
                'is_best': ckpt_meta.get('is_best', False),
                'size_mb': ckpt_meta.get('size_bytes', 0) / 1024 / 1024
            }

            # Add metrics as separate columns
            metrics = ckpt_meta.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                row[f'metric_{metric_name}'] = metric_value

            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values('timestamp', ascending=False) if not df.empty else df

    def cleanup_checkpoints(
        self,
        keep_best: int = 3,
        keep_last: int = 2,
        metric: str = "loss",
        mode: str = "min"
    ) -> None:
        """
        Remove old checkpoints, keeping only the best and most recent.

        Args:
            keep_best: Number of best checkpoints to keep based on metric
            keep_last: Number of most recent checkpoints to keep
            metric: Metric to use for selecting best checkpoints
            mode: 'max' or 'min' for the metric
        """
        if not self.metadata:
            return

        # Get list of checkpoints (excluding best_checkpoint.pt)
        checkpoints = [
            (name, meta) for name, meta in self.metadata.items()
            if name != 'best_checkpoint.pt' and Path(meta['path']).exists()
        ]

        if len(checkpoints) <= (keep_best + keep_last):
            return

        # Sort by timestamp for recent
        checkpoints_by_time = sorted(
            checkpoints,
            key=lambda x: x[1].get('timestamp', ''),
            reverse=True
        )
        keep_recent = set(name for name, _ in checkpoints_by_time[:keep_last])

        # Sort by metric for best
        checkpoints_with_metric = [
            (name, meta) for name, meta in checkpoints
            if metric in meta.get('metrics', {})
        ]
        checkpoints_with_metric.sort(
            key=lambda x: x[1]['metrics'][metric],
            reverse=(mode == "max")
        )
        keep_best_set = set(name for name, _ in checkpoints_with_metric[:keep_best])

        # Combine keep sets
        keep_all = keep_recent | keep_best_set

        # Remove checkpoints not in keep set
        removed_count = 0
        for name, meta in checkpoints:
            if name not in keep_all:
                checkpoint_path = Path(meta['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    removed_count += 1
                del self.metadata[name]

        # Save updated metadata
        self._save_metadata()
        print(f"Removed {removed_count} old checkpoints")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint."""
        if not self.metadata:
            return None

        latest = max(
            ((name, meta) for name, meta in self.metadata.items()
             if name != 'best_checkpoint.pt'),
            key=lambda x: x[1].get('timestamp', ''),
            default=None
        )

        return Path(latest[1]['path']) if latest else None

    def delete_checkpoint(self, checkpoint_name: str) -> None:
        """Delete a specific checkpoint."""
        if checkpoint_name not in self.metadata:
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")

        checkpoint_path = Path(self.metadata[checkpoint_name]['path'])
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        del self.metadata[checkpoint_name]
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Save checkpoint metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
