# ExpSetUP - Experiment Setup and Tracking for PyTorch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive experiment tracking and checkpoint management package for PyTorch models. ExpSetUP helps you track configurations, training metrics, evaluation performance, and manage model checkpoints efficiently.

## Features

- 📊 **Comprehensive Logging**: Track configurations, training metrics, and evaluation results
- 💾 **Smart Checkpoint Management**: Automatic checkpoint saving with performance-based retention
- 📈 **TensorBoard Integration**: Built-in TensorBoard logging for visualization
- 🔧 **Flexible Configuration**: Support for YAML, JSON, and Python dict configurations
- 🎯 **Performance Tracking**: Easy retrieval of best models based on metrics
- 📦 **Clean API**: Simple, intuitive interface for PyTorch workflows
- 🔄 **Resume Training**: Easy experiment resumption from checkpoints

## Installation

### From GitHub (Private Repository)

1. Generate a GitHub Personal Access Token (PAT) with `repo` scope:
   - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click "Generate new token (classic)"
   - Select the `repo` scope
   - Copy the generated token

2. Install using pip:

```bash
# Option 1: Install directly from GitHub
pip install git+https://github.com/yourusername/ExpSetUP.git

# Option 2: Using SSH (if you have SSH keys set up)
pip install git+ssh://git@github.com/yourusername/ExpSetUP.git

# Option 3: Install with authentication token
pip install git+https://<YOUR_TOKEN>@github.com/yourusername/ExpSetUP.git
```

3. Add to your `requirements.txt`:

```
git+https://github.com/yourusername/ExpSetUP.git@main
```

### For Development

```bash
git clone https://github.com/yourusername/ExpSetUP.git
cd ExpSetUP
pip install -e .
```

## Quick Start

```python
import torch
import torch.nn as nn
from expsetup import ExperimentLogger

# Create logger
logger = ExperimentLogger(
    experiment_name="mnist_training",
    base_dir="./experiments",
    use_tensorboard=True
)

# Log configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "model": "resnet18"
}
logger.log_config(config)

# Your training loop
model = nn.Linear(784, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    # Training
    train_loss = 0.5  # Your training logic
    logger.log_metrics({"loss": train_loss, "lr": 0.001}, step=epoch, prefix="train/")

    # Validation
    val_loss = 0.4
    val_acc = 0.92
    logger.log_evaluation({"loss": val_loss, "accuracy": val_acc}, epoch=epoch)

    # Save checkpoint
    logger.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={"val_accuracy": val_acc, "val_loss": val_loss},
        is_best=(val_acc > 0.90)
    )

# Cleanup
logger.close()
```

## Core Components

### ExperimentLogger

The main interface for experiment tracking.

```python
from expsetup import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="my_experiment",
    base_dir="./experiments",
    use_tensorboard=True,
    resume_from=None  # or path to resume from
)
```

**Key Methods:**

- `log_config(config)`: Log experiment configuration
- `log_metrics(metrics, step, prefix)`: Log training metrics
- `log_evaluation(metrics, epoch, split)`: Log evaluation results
- `save_checkpoint(model, optimizer, ...)`: Save model checkpoint
- `load_checkpoint(path, model, optimizer, ...)`: Load checkpoint
- `get_best_checkpoint(metric, mode)`: Get best model by metric
- `cleanup_checkpoints(keep_best, keep_last, ...)`: Clean old checkpoints

### CheckpointManager

Handles checkpoint saving, loading, and management.

```python
from expsetup import CheckpointManager

manager = CheckpointManager("./checkpoints")

# Save checkpoint
manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=5,
    metrics={"accuracy": 0.95}
)

# Get best checkpoint
best_ckpt = manager.get_best_checkpoint(metric="accuracy", mode="max")

# List all checkpoints
df = manager.list_checkpoints()
print(df)

# Cleanup old checkpoints
manager.cleanup_checkpoints(keep_best=3, keep_last=2, metric="accuracy")
```

### Config

Configuration management with multiple format support.

```python
from expsetup import Config

# From dictionary
config = Config({"lr": 0.001, "batch_size": 32})

# From file
config = Config("config.yaml")

# Access values
lr = config.lr  # or config["lr"] or config.get("lr")

# Save configuration
config.save("output_config.yaml")
```

## Directory Structure

When you create an experiment, ExpSetUP organizes everything in a structured directory:

```
experiments/
└── my_experiment_20240105_143022/
    ├── metadata.json              # Experiment metadata
    ├── config/
    │   └── config.yaml           # Saved configuration
    ├── checkpoints/
    │   ├── checkpoint_epoch5_*.pt
    │   ├── best_checkpoint.pt    # Best model copy
    │   └── checkpoints_metadata.json
    ├── logs/
    │   ├── metrics.csv           # All metrics in CSV
    │   ├── validation_results.json
    │   └── test_results.json
    └── tensorboard/              # TensorBoard logs
        └── events.out.tfevents.*
```

## Advanced Usage

### Context Manager

```python
with ExperimentLogger("my_experiment") as logger:
    logger.log_config(config)
    # Your training code
    logger.log_metrics({"loss": 0.5}, step=0)
# Automatically closes and saves
```

### Resume Training

```python
# Resume from existing experiment
logger = ExperimentLogger(
    experiment_name="my_experiment",
    resume_from="./experiments/my_experiment_20240105_143022"
)

# Load last checkpoint
latest_ckpt = logger.checkpoint_manager.get_latest_checkpoint()
checkpoint_data = logger.load_checkpoint(latest_ckpt, model, optimizer)

start_epoch = checkpoint_data['epoch'] + 1
# Continue training from start_epoch
```

### Custom Checkpoint Names

```python
logger.save_checkpoint(
    model=model,
    optimizer=optimizer,
    checkpoint_name="my_special_checkpoint.pt",
    metrics={"accuracy": 0.95}
)
```

### Multiple Metrics Tracking

```python
# Training metrics
logger.log_metrics({
    "loss": 0.5,
    "accuracy": 0.85,
    "learning_rate": 0.001,
    "gradient_norm": 2.3
}, step=100, prefix="train/")

# Validation metrics
logger.log_metrics({
    "loss": 0.4,
    "accuracy": 0.90,
    "f1_score": 0.88
}, step=100, prefix="val/")
```

### Performance-Based Checkpoint Cleanup

```python
# Keep 3 best by accuracy, 2 most recent
logger.cleanup_checkpoints(
    keep_best=3,
    keep_last=2,
    metric="accuracy",
    mode="max"
)

# Keep 5 best by loss, 1 most recent
logger.cleanup_checkpoints(
    keep_best=5,
    keep_last=1,
    metric="loss",
    mode="min"
)
```

### Saving Additional State

```python
logger.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=lr_scheduler,
    epoch=epoch,
    metrics={"accuracy": 0.95},
    # Additional custom state
    random_state=torch.get_rng_state(),
    scaler_state=scaler.state_dict(),
    best_metrics_so_far={"best_acc": 0.95}
)
```

## Integration Examples

### With PyTorch Lightning

```python
import pytorch_lightning as pl
from expsetup import ExperimentLogger

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.exp_logger = ExperimentLogger("lightning_exp")

    def training_step(self, batch, batch_idx):
        loss = ...  # Your loss
        self.exp_logger.log_metrics({"loss": loss}, step=self.global_step, prefix="train/")
        return loss
```

### With Hugging Face Transformers

```python
from transformers import Trainer, TrainingArguments
from expsetup import ExperimentLogger

logger = ExperimentLogger("transformer_training")

class CustomTrainer(Trainer):
    def log(self, logs):
        super().log(logs)
        logger.log_metrics(logs, step=self.state.global_step)
```

## API Reference

See the [examples](examples/) directory for detailed usage examples and the inline documentation in the source code for complete API details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions, please open an issue on the GitHub repository.
