# ExpSetUP Quick Start Guide

Get up and running with ExpSetUP in 5 minutes!

## Installation

```bash
pip install git+https://github.com/yourusername/ExpSetUP.git
```

## Basic Usage

### 1. Minimal Example

```python
from expsetup import ExperimentLogger
import torch.nn as nn
import torch.optim as optim

# Create logger
logger = ExperimentLogger("my_experiment")

# Log config
logger.log_config({"lr": 0.001, "epochs": 10})

# Your model and optimizer
model = nn.Linear(10, 5)
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(10):
    # ... your training code ...
    train_loss = 0.5  # your actual loss

    # Log metrics
    logger.log_metrics({"loss": train_loss}, step=epoch, prefix="train/")

    # Save checkpoint
    logger.save_checkpoint(model, optimizer, epoch=epoch)

logger.close()
```

### 2. With Validation

```python
from expsetup import ExperimentLogger

logger = ExperimentLogger("my_experiment")
logger.log_config(config)

for epoch in range(epochs):
    # Training
    train_loss = train_one_epoch(model, train_loader)
    logger.log_metrics({"loss": train_loss}, step=epoch, prefix="train/")

    # Validation
    val_loss, val_acc = validate(model, val_loader)
    logger.log_evaluation(
        {"loss": val_loss, "accuracy": val_acc},
        epoch=epoch,
        split="validation"
    )

    # Save with metrics
    logger.save_checkpoint(
        model, optimizer,
        epoch=epoch,
        metrics={"val_accuracy": val_acc},
        is_best=(val_acc > best_acc)
    )

logger.close()
```

### 3. Resume Training

```python
from expsetup import ExperimentLogger

# Resume from existing experiment
logger = ExperimentLogger(
    "my_experiment",
    resume_from="./experiments/my_experiment_20240105_143022"
)

# Load latest checkpoint
latest = logger.checkpoint_manager.get_latest_checkpoint()
checkpoint = logger.load_checkpoint(latest, model, optimizer)

# Continue from saved epoch
start_epoch = checkpoint['epoch'] + 1

for epoch in range(start_epoch, total_epochs):
    # ... continue training ...
    pass

logger.close()
```

## Key Features

### Configuration Management

```python
from expsetup import Config

# From dict
config = Config({"lr": 0.001, "batch_size": 32})

# From file
config = Config("config.yaml")

# Access values
lr = config.lr
batch_size = config["batch_size"]

# Save
config.save("output.yaml")
```

### Checkpoint Management

```python
# Save checkpoint
logger.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    metrics={"val_acc": 0.95},
    is_best=True
)

# Get best checkpoint
best = logger.get_best_checkpoint(metric="val_acc", mode="max")

# List all checkpoints
df = logger.list_checkpoints()

# Cleanup old checkpoints
logger.cleanup_checkpoints(
    keep_best=3,
    keep_last=2,
    metric="val_acc",
    mode="max"
)
```

### TensorBoard Integration

```python
# TensorBoard is enabled by default
logger = ExperimentLogger("my_experiment", use_tensorboard=True)

# Metrics are automatically logged to TensorBoard
logger.log_metrics({"loss": 0.5}, step=epoch)

# View with: tensorboard --logdir experiments/
```

### Context Manager

```python
with ExperimentLogger("my_experiment") as logger:
    logger.log_config(config)
    # Your training code
    logger.log_metrics({"loss": 0.5}, step=0)
# Automatically closed
```

## Directory Structure

After running an experiment, you'll have:

```
experiments/
└── my_experiment_20240105_143022/
    ├── metadata.json              # Experiment info
    ├── config/
    │   └── config.yaml           # Your configuration
    ├── checkpoints/
    │   ├── checkpoint_epoch0_*.pt
    │   ├── checkpoint_epoch1_*.pt
    │   ├── best_checkpoint.pt    # Best model
    │   └── checkpoints_metadata.json
    ├── logs/
    │   ├── metrics.csv           # All metrics
    │   ├── validation_results.json
    │   └── test_results.json
    └── tensorboard/              # TensorBoard logs
        └── events.out.tfevents.*
```

## Common Patterns

### Pattern 1: Simple Training

```python
with ExperimentLogger("quick_test") as logger:
    logger.log_config({"lr": 0.001})

    for epoch in range(10):
        loss = train_epoch(model, loader)
        logger.log_metrics({"loss": loss}, step=epoch)
        logger.save_checkpoint(model, epoch=epoch)
```

### Pattern 2: Production Training

```python
logger = ExperimentLogger("production", base_dir="/shared/experiments")
logger.log_config(config)

best_acc = 0.0
for epoch in range(100):
    train_loss = train_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    logger.log_metrics({"loss": train_loss}, step=epoch, prefix="train/")
    logger.log_evaluation({"loss": val_loss, "acc": val_acc}, epoch=epoch)

    is_best = val_acc > best_acc
    if is_best:
        best_acc = val_acc

    logger.save_checkpoint(
        model, optimizer, scheduler,
        epoch=epoch,
        metrics={"val_acc": val_acc},
        is_best=is_best
    )

logger.close()
```

### Pattern 3: Hyperparameter Search

```python
for lr in [0.001, 0.01, 0.1]:
    logger = ExperimentLogger(f"hp_search_lr{lr}")
    logger.log_config({"lr": lr, "batch_size": 32})

    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        loss = train_epoch(model, loader)
        logger.log_metrics({"loss": loss}, step=epoch)

    logger.close()
```

## Next Steps

- Read the [README.md](README.md) for detailed API reference
- Check [examples/](examples/) for complete working examples
- See [SETUP_GUIDE.md](SETUP_GUIDE.md) for installation details
- Review [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Tips

1. **Always use context manager** when possible for automatic cleanup
2. **Log configuration early** to track experiment settings
3. **Use meaningful experiment names** for easy identification
4. **Set `is_best=True`** for best checkpoints to create a copy
5. **Cleanup checkpoints regularly** to save disk space
6. **Use prefixes** (`train/`, `val/`) to organize metrics
7. **Enable TensorBoard** for visualization during training

## Troubleshooting

### Issue: Module not found

```bash
pip install git+https://github.com/yourusername/ExpSetUP.git
```

### Issue: Permission error on directory creation

```bash
# Use a writable directory
logger = ExperimentLogger("exp", base_dir="./my_experiments")
```

### Issue: CUDA out of memory when loading checkpoint

```python
# Specify CPU device
logger.load_checkpoint(path, model, device=torch.device('cpu'))
```

## Support

For more help:
- Check the [examples/](examples/) directory
- Read the [full documentation](README.md)
- Open an issue on GitHub

Happy experimenting! 🚀
