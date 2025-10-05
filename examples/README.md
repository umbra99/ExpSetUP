# ExpSetUP Examples

This directory contains example scripts demonstrating various features of the ExpSetUP package.

## Examples Overview

### 1. Basic Usage ([basic_usage.py](basic_usage.py))

Demonstrates the fundamental features of ExpSetUP:
- Creating an experiment logger
- Logging configuration
- Tracking training and validation metrics
- Saving checkpoints with metadata
- Loading best checkpoints
- Cleaning up old checkpoints

**Run:**
```bash
python examples/basic_usage.py
```

**Key Features:**
- Simple neural network training
- Metric logging during training
- Checkpoint management
- Best model selection

### 2. Resume Training ([resume_training.py](resume_training.py))

Shows how to resume interrupted training:
- Resuming an experiment from existing directory
- Loading the latest checkpoint
- Continuing training from saved state
- Preserving optimizer state

**Run:**
```bash
python examples/resume_training.py
```

**Key Features:**
- Two-phase training (initial + resume)
- Checkpoint loading
- State restoration
- Seamless continuation

### 3. Advanced Usage ([advanced_usage.py](advanced_usage.py))

Demonstrates advanced features:
- Using the Config class for configuration
- Learning rate scheduling
- Custom state saving/loading
- Multiple metric tracking
- Per-class accuracy tracking
- Context manager usage
- Performance-based checkpoint cleanup

**Run:**
```bash
python examples/advanced_usage.py
```

**Key Features:**
- Complex model architecture
- Comprehensive metric tracking
- LR scheduler integration
- Advanced checkpoint management
- Test set evaluation

## Running the Examples

### Prerequisites

Make sure ExpSetUP is installed:

```bash
# From the repository root
pip install -e .
```

### Run All Examples

```bash
# Run basic example
python examples/basic_usage.py

# Run resume training example
python examples/resume_training.py

# Run advanced example
python examples/advanced_usage.py
```

### Expected Output

Each example will:
1. Create an `experiments/` directory in the current location
2. Generate experiment subdirectories with timestamps
3. Save checkpoints, logs, and metrics
4. Display training progress in the console
5. Create TensorBoard logs (if enabled)

### Viewing Results

**View TensorBoard logs:**
```bash
tensorboard --logdir experiments/
```

**Inspect saved files:**
```bash
# List experiment directories
ls -la experiments/

# View a specific experiment
ls -la experiments/basic_example_*/

# View metrics
cat experiments/basic_example_*/logs/metrics.csv

# View configuration
cat experiments/basic_example_*/config/config.yaml
```

## Example Output Structure

After running the examples, you'll see a structure like:

```
experiments/
├── basic_example_20240105_143022/
│   ├── metadata.json
│   ├── config/
│   │   └── config.yaml
│   ├── checkpoints/
│   │   ├── checkpoint_epoch0_*.pt
│   │   ├── checkpoint_epoch1_*.pt
│   │   ├── best_checkpoint.pt
│   │   └── checkpoints_metadata.json
│   ├── logs/
│   │   ├── metrics.csv
│   │   └── validation_results.json
│   └── tensorboard/
│       └── events.out.tfevents.*
│
├── resume_example_20240105_143045/
│   └── ... (similar structure)
│
└── advanced_example_20240105_143110/
    └── ... (similar structure)
```

## Understanding the Examples

### Metric Tracking

All examples demonstrate metric tracking:

```python
# Training metrics
logger.log_metrics({
    "loss": train_loss,
    "accuracy": train_acc
}, step=epoch, prefix="train/")

# Validation metrics
logger.log_evaluation({
    "loss": val_loss,
    "accuracy": val_acc
}, epoch=epoch, split="validation")
```

### Checkpoint Management

Checkpoints are saved with metadata:

```python
logger.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={"val_accuracy": val_acc},
    is_best=is_best
)
```

### Best Model Selection

Retrieve the best model based on any metric:

```python
best_checkpoint = logger.get_best_checkpoint(
    metric="val_accuracy",
    mode="max"
)
```

## Customization

Feel free to modify these examples:

1. **Change the model architecture**: Replace `SimpleNet` or `DeepNet` with your own models
2. **Use real datasets**: Replace dummy data with actual PyTorch datasets
3. **Add custom metrics**: Track additional metrics specific to your task
4. **Modify training loops**: Adapt the training procedure to your needs
5. **Experiment with configurations**: Try different hyperparameters

## Common Patterns

### Pattern 1: Quick Experiment

```python
with ExperimentLogger("quick_test") as logger:
    logger.log_config(config)
    # Your training code
    logger.save_checkpoint(model, metrics=metrics)
```

### Pattern 2: Production Training

```python
logger = ExperimentLogger("production_model", base_dir="/shared/experiments")
logger.log_config(config)

try:
    # Long training loop
    for epoch in range(epochs):
        # Training code
        logger.save_checkpoint(model, epoch=epoch, metrics=metrics)
finally:
    logger.close()
```

### Pattern 3: Resume with Error Handling

```python
try:
    logger = ExperimentLogger(resume_from=exp_dir)
    latest = logger.checkpoint_manager.get_latest_checkpoint()
    logger.load_checkpoint(latest, model, optimizer)
except FileNotFoundError:
    logger = ExperimentLogger("new_experiment")
    logger.log_config(config)
```

## Troubleshooting

### Issue: "No module named 'expsetup'"

**Solution:** Install the package first:
```bash
pip install -e .
```

### Issue: CUDA out of memory

**Solution:** Reduce batch size in the config or add to the example:
```python
config["batch_size"] = 32  # Reduce from 64
```

### Issue: TensorBoard not showing data

**Solution:** Check the TensorBoard directory and run:
```bash
tensorboard --logdir experiments/your_experiment_dir/tensorboard/
```

## Next Steps

After running the examples:

1. Integrate ExpSetUP into your own projects
2. Customize the examples for your specific use cases
3. Explore the API documentation in the main README
4. Set up the package as described in SETUP_GUIDE.md

For questions or issues, please refer to the main [README.md](../README.md) or open an issue on GitHub.
