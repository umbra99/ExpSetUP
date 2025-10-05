# Getting Started with ExpSetUP

This guide will help you publish ExpSetUP to GitHub and start using it in your experiments.

## Step 1: Publish to GitHub

### 1.1 Update pyproject.toml

First, update the package information in [pyproject.toml](pyproject.toml):

```toml
[project]
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/ExpSetUP"
Repository = "https://github.com/YOUR_USERNAME/ExpSetUP"
Issues = "https://github.com/YOUR_USERNAME/ExpSetUP/issues"
```

Replace `YOUR_USERNAME` with your actual GitHub username.

### 1.2 Create GitHub Repository

1. Go to [GitHub](https://github.com)
2. Click "+" → "New repository"
3. Name: `ExpSetUP`
4. Visibility: Choose Private or Public
5. **Don't** initialize with README
6. Click "Create repository"

### 1.3 Push Code to GitHub

```bash
# Navigate to the project directory
cd /Users/gereonvienken/projects/ExpSetUP

# Verify git status
git status

# Add all files
git add .

# Commit
git commit -m "Initial commit: ExpSetUP package for PyTorch experiment tracking

Features:
- Comprehensive experiment logging
- Smart checkpoint management
- Configuration tracking
- TensorBoard integration
- Performance-based checkpoint cleanup
- Full documentation and examples"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ExpSetUP.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 1.4 Verify Upload

Visit `https://github.com/YOUR_USERNAME/ExpSetUP` and verify all files are present.

## Step 2: Set Up Authentication

### Option A: Personal Access Token (Recommended)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Name: "ExpSetUP Package Access"
4. Scopes: Select `repo` (full control)
5. Click "Generate token"
6. **Copy the token immediately**

Save it in your environment:

```bash
# Add to ~/.bashrc or ~/.zshrc
export GITHUB_TOKEN="ghp_your_token_here"

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

### Option B: SSH Keys (Alternative)

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

## Step 3: Install in Your Projects

### In Your Experiment Project

```bash
# Create/activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ExpSetUP
pip install git+https://github.com/YOUR_USERNAME/ExpSetUP.git

# Or with token
pip install git+https://${GITHUB_TOKEN}@github.com/YOUR_USERNAME/ExpSetUP.git
```

### Add to requirements.txt

```
# requirements.txt
torch>=1.9.0
numpy>=1.21.0
git+https://github.com/YOUR_USERNAME/ExpSetUP.git@main
```

Install with:
```bash
pip install -r requirements.txt
```

## Step 4: Verify Installation

Run the verification script:

```bash
python -m expsetup.verify_installation
# Or if you cloned the repo:
python verify_installation.py
```

You should see:
```
🎉 All tests passed! ExpSetUP is correctly installed.
```

## Step 5: Run Your First Experiment

Create a file `my_first_experiment.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from expsetup import ExperimentLogger

# Simple model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create experiment logger
with ExperimentLogger("my_first_experiment") as logger:
    # Log configuration
    logger.log_config({
        "model": "SimpleNN",
        "learning_rate": 0.001,
        "hidden_size": 128,
        "num_epochs": 5
    })

    # Simulate training
    for epoch in range(5):
        # Your training code here
        train_loss = 1.0 / (epoch + 1)  # Dummy loss
        val_acc = 0.5 + epoch * 0.08    # Dummy accuracy

        # Log metrics
        logger.log_metrics({"loss": train_loss}, step=epoch, prefix="train/")
        logger.log_evaluation({"accuracy": val_acc}, epoch=epoch)

        # Save checkpoint
        logger.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"val_accuracy": val_acc},
            is_best=(val_acc > 0.7)
        )

        print(f"Epoch {epoch + 1}/5 - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

print("\n✓ Experiment completed!")
print(f"Results saved to: experiments/")
```

Run it:
```bash
python my_first_experiment.py
```

## Step 6: View Results

### View Experiment Files

```bash
# List experiments
ls -la experiments/

# View specific experiment
ls -la experiments/my_first_experiment_*/

# View metrics
cat experiments/my_first_experiment_*/logs/metrics.csv

# View configuration
cat experiments/my_first_experiment_*/config/config.yaml
```

### View TensorBoard

```bash
tensorboard --logdir experiments/
```

Then open http://localhost:6006 in your browser.

## Step 7: Try the Examples

Run the provided examples to learn more:

```bash
# Basic usage
python examples/basic_usage.py

# Resume training
python examples/resume_training.py

# Advanced features
python examples/advanced_usage.py
```

## Common Workflows

### Workflow 1: Quick Experiment

```python
from expsetup import ExperimentLogger

with ExperimentLogger("quick_test") as logger:
    logger.log_config(config)
    for epoch in range(epochs):
        # Training code
        logger.log_metrics(metrics, step=epoch)
        logger.save_checkpoint(model, epoch=epoch)
```

### Workflow 2: Hyperparameter Search

```python
from expsetup import ExperimentLogger

for lr in [0.001, 0.01, 0.1]:
    exp_name = f"hp_search_lr_{lr}"
    with ExperimentLogger(exp_name) as logger:
        logger.log_config({"lr": lr})
        # Train and log
```

### Workflow 3: Resume Training

```python
from expsetup import ExperimentLogger

logger = ExperimentLogger(
    "my_experiment",
    resume_from="experiments/my_experiment_20240105_143022"
)

latest = logger.checkpoint_manager.get_latest_checkpoint()
checkpoint = logger.load_checkpoint(latest, model, optimizer)

for epoch in range(checkpoint['epoch'] + 1, total_epochs):
    # Continue training
    pass
```

## Next Steps

1. **Integrate into your project:** Replace your current logging with ExpSetUP
2. **Customize:** Adjust configurations for your specific needs
3. **Explore features:** Check out advanced features in the documentation
4. **Share with team:** Help teammates install and use the package

## Resources

- **Main Documentation:** [README.md](README.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **Setup Guide:** [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Examples:** [examples/](examples/)
- **API Reference:** Inline documentation in source code

## Troubleshooting

### Package not found
```bash
# Verify token is set
echo $GITHUB_TOKEN

# Try with explicit token
pip install git+https://${GITHUB_TOKEN}@github.com/YOUR_USERNAME/ExpSetUP.git
```

### Import errors
```bash
# Verify installation
pip show expsetup

# Reinstall
pip uninstall expsetup
pip install git+https://github.com/YOUR_USERNAME/ExpSetUP.git
```

### Permission errors
```bash
# Use local directory for experiments
logger = ExperimentLogger("exp", base_dir="./my_experiments")
```

## Getting Help

- Check the [documentation](README.md)
- Review [examples](examples/)
- Open an issue on GitHub
- Check existing issues for similar problems

## Best Practices

1. **Always log configuration:** Track what you're running
2. **Use meaningful names:** Name experiments descriptively
3. **Set is_best flag:** Mark best checkpoints explicitly
4. **Clean up regularly:** Use cleanup_checkpoints to save space
5. **Use context manager:** Ensures proper cleanup
6. **Track all metrics:** Log everything you might need later
7. **Document experiments:** Add notes in configuration

---

Happy experimenting with ExpSetUP! 🚀
