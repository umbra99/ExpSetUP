"""
Example demonstrating how to resume training from a checkpoint.

This example shows:
- Resuming an experiment from an existing directory
- Loading the latest checkpoint
- Continuing training from the saved epoch
- Preserving optimizer and model state
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from expsetup import ExperimentLogger


class SimpleNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
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

    for data, target in dataloader:
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

    return total_loss / len(dataloader), correct / total


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

    return total_loss / len(dataloader), correct / total


def initial_training():
    """Run initial training that we'll interrupt."""
    print("=" * 60)
    print("PHASE 1: Initial Training (will stop after 5 epochs)")
    print("=" * 60)

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

    # Create new experiment
    logger = ExperimentLogger(
        experiment_name="resume_example",
        base_dir="./experiments",
        use_tensorboard=True
    )
    logger.log_config(config)

    device = torch.device(config["device"])
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_dataset = create_dummy_data(num_samples=1000)
    val_dataset = create_dummy_data(num_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Train for only 5 epochs (simulate interruption)
    for epoch in range(5):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.log_metrics({"loss": train_loss, "accuracy": train_acc},
                          step=epoch, prefix="train/")
        logger.log_evaluation({"loss": val_loss, "accuracy": val_acc},
                             epoch=epoch, split="validation")

        logger.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"val_loss": val_loss, "val_accuracy": val_acc}
        )

        print(f"Epoch {epoch + 1}/5: "
              f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    exp_dir = logger.exp_dir
    logger.close()

    print(f"\nInitial training stopped. Experiment saved at: {exp_dir}")
    return exp_dir


def resume_training(exp_dir):
    """Resume training from checkpoint."""
    print("\n" + "=" * 60)
    print("PHASE 2: Resuming Training")
    print("=" * 60)

    # Resume experiment from existing directory
    logger = ExperimentLogger(
        experiment_name="resume_example",
        resume_from=exp_dir,
        use_tensorboard=True
    )

    print(f"Resumed experiment from: {exp_dir}")

    # Recreate model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Load latest checkpoint
    latest_checkpoint_path = logger.checkpoint_manager.get_latest_checkpoint()
    print(f"Loading checkpoint: {latest_checkpoint_path}")

    checkpoint_data = logger.load_checkpoint(
        checkpoint_path=latest_checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device
    )

    # Get starting epoch (continue from where we left off)
    start_epoch = checkpoint_data['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")

    # Recreate data loaders
    train_dataset = create_dummy_data(num_samples=1000)
    val_dataset = create_dummy_data(num_samples=200)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Continue training for remaining epochs
    total_epochs = 10
    for epoch in range(start_epoch, total_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.log_metrics({"loss": train_loss, "accuracy": train_acc},
                          step=epoch, prefix="train/")
        logger.log_evaluation({"loss": val_loss, "accuracy": val_acc},
                             epoch=epoch, split="validation")

        logger.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"val_loss": val_loss, "val_accuracy": val_acc}
        )

        print(f"Epoch {epoch + 1}/{total_epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Show all checkpoints
    print("\nAll checkpoints:")
    print(logger.list_checkpoints())

    # Get best model
    best_checkpoint = logger.get_best_checkpoint(metric="val_accuracy", mode="max")
    print(f"\nBest checkpoint: {best_checkpoint}")

    logger.close()
    print("\nTraining completed!")


def main():
    # Run initial training
    exp_dir = initial_training()

    # Simulate program restart - resume training
    resume_training(exp_dir)


if __name__ == "__main__":
    main()
