"""
ExpSetUP - Experiment Setup and Tracking for PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A comprehensive experiment tracking and checkpoint management package for PyTorch models.

Basic usage:

    >>> from expsetup import ExperimentLogger
    >>> logger = ExperimentLogger(experiment_name="my_experiment", base_dir="./experiments")
    >>> logger.log_config({"learning_rate": 0.001, "batch_size": 32})
    >>> logger.log_metrics({"train_loss": 0.5, "train_acc": 0.85}, step=100)
    >>> logger.save_checkpoint(model, optimizer, epoch=1, metrics={"val_acc": 0.88})

:copyright: (c) 2024 by ExpSetUP Contributors.
:license: MIT, see LICENSE for more details.
"""

__version__ = "0.1.0"
__author__ = "ExpSetUP Contributors"

from expsetup.logger import ExperimentLogger
from expsetup.checkpoint_manager import CheckpointManager
from expsetup.config import Config

__all__ = ["ExperimentLogger", "CheckpointManager", "Config"]
