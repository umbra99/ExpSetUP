"""
Basic usage example of ExpSetUP package.

This example demonstrates the fundamental features of ExpSetUP:
- Creating an experiment logger
- Logging configuration
- Tracking metrics during training
- Saving and loading checkpoints
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from expsetup import ExperimentLogger


# Simple neural network for demonstration
class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """Create dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # Configuration
    config = {
        "model": "SimpleNet",
        "input_size": 784,
        "hidden_size": 128,
        "num_classes": 10,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Create experiment logger
    logger = ExperimentLogger(
        experiment_name="basic_example",
        base_dir="./experiments",
        use_tensorboard=True
    )

    # Log configuration
    logger.log_config(config)
    print(f"Experiment directory: {logger.exp_dir}")

    # Setup
    device = torch.device(config["device"])

    # Create model, optimizer, criterion
    model = SimpleNet(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_classes=config["num_classes"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Create dummy datasets
    train_dataset = create_dummy_data(num_samples=1000)
    val_dataset = create_dummy_data(num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(config["epochs"]):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Log training metrics
        logger.log_metrics({
            "loss": train_loss,
            "accuracy": train_acc
        }, step=epoch, prefix="train/")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log validation metrics
        logger.log_evaluation({
            "loss": val_loss,
            "accuracy": val_acc
        }, epoch=epoch, split="validation")

        # Print progress
        print(f"Epoch {epoch + 1}/{config['epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Check if this is the best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # Save checkpoint
        logger.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            },
            is_best=is_best
        )

    # Get best checkpoint
    best_checkpoint = logger.get_best_checkpoint(metric="val_accuracy", mode="max")
    print(f"\nBest checkpoint: {best_checkpoint}")

    # List all checkpoints
    print("\nAll checkpoints:")
    checkpoints_df = logger.list_checkpoints()
    print(checkpoints_df)

    # Cleanup old checkpoints (keep 3 best, 2 most recent)
    logger.cleanup_checkpoints(
        keep_best=3,
        keep_last=2,
        metric="val_accuracy",
        mode="max"
    )

    # Close logger
    logger.close()
    print(f"\nExperiment completed! Results saved to {logger.exp_dir}")


if __name__ == "__main__":
    main()
