"""
Advanced usage example of ExpSetUP package.

This example demonstrates advanced features:
- Custom checkpoint naming
- Learning rate scheduling with checkpoint saving
- Multiple metric tracking
- Custom state saving/loading
- Performance-based checkpoint cleanup
- Using context manager
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from expsetup import ExperimentLogger, Config


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x += residual
        return self.relu(x)


class DeepNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10, num_blocks=3):
        super(DeepNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


def create_dummy_data(num_samples=1000, input_size=784, num_classes=10):
    """Create dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def compute_gradient_norm(model):
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch with detailed metrics."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    gradient_norms = []

    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Track gradient norm
        grad_norm = compute_gradient_norm(model)
        gradient_norms.append(grad_norm)

        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    avg_grad_norm = sum(gradient_norms) / len(gradient_norms)

    return avg_loss, accuracy, avg_grad_norm


def validate(model, dataloader, criterion, device):
    """Validate with multiple metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # Per-class accuracy
            for i in range(len(target)):
                label = target[i].item()
                class_correct[label] += pred[i].eq(target[i]).item()
                class_total[label] += 1

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    # Calculate per-class accuracy
    per_class_acc = {
        f"class_{i}_acc": class_correct[i] / max(class_total[i], 1)
        for i in range(10)
    }

    return avg_loss, accuracy, per_class_acc


def main():
    # Create configuration using Config class
    config = Config({
        "model": {
            "name": "DeepNet",
            "input_size": 784,
            "hidden_size": 256,
            "num_classes": 10,
            "num_blocks": 3
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 15,
            "lr_step_size": 5,
            "lr_gamma": 0.5
        },
        "checkpoint": {
            "save_frequency": 2,  # Save every 2 epochs
            "keep_best": 3,
            "keep_last": 2
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

    # Use context manager for automatic cleanup
    with ExperimentLogger(
        experiment_name="advanced_example",
        base_dir="./experiments",
        use_tensorboard=True
    ) as logger:

        # Log configuration
        logger.log_config(config, name="advanced_config")
        print(f"Experiment directory: {logger.exp_dir}")

        # Setup
        device = torch.device(config.device)

        # Create model with config
        model = DeepNet(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_classes=config.model.num_classes,
            num_blocks=config.model.num_blocks
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

        # Learning rate scheduler
        scheduler = StepLR(
            optimizer,
            step_size=config.training.lr_step_size,
            gamma=config.training.lr_gamma
        )

        criterion = nn.CrossEntropyLoss()

        # Create datasets
        train_dataset = create_dummy_data(num_samples=2000)
        val_dataset = create_dummy_data(num_samples=400)
        test_dataset = create_dummy_data(num_samples=400)

        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        # Training state
        best_val_acc = 0.0
        patience_counter = 0

        # Training loop
        for epoch in range(config.training.epochs):
            # Train
            train_loss, train_acc, grad_norm = train_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log comprehensive training metrics
            logger.log_metrics({
                "loss": train_loss,
                "accuracy": train_acc,
                "learning_rate": current_lr,
                "gradient_norm": grad_norm
            }, step=epoch, prefix="train/")

            # Validate
            val_loss, val_acc, per_class_acc = validate(
                model, val_loader, criterion, device
            )

            # Log validation metrics
            val_metrics = {
                "loss": val_loss,
                "accuracy": val_acc,
                **per_class_acc
            }
            logger.log_evaluation(val_metrics, epoch=epoch, split="validation")

            # Print progress
            print(f"Epoch {epoch + 1}/{config.training.epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                  f"Grad Norm: {grad_norm:.4f}, LR: {current_lr:.6f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            # Check if best model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0
                print(f"  ✓ New best model! Accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1

            # Save checkpoint (with custom state)
            if epoch % config.checkpoint.save_frequency == 0 or is_best:
                logger.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,  # Save scheduler state
                    epoch=epoch,
                    metrics={
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc
                    },
                    is_best=is_best,
                    # Custom state
                    best_val_accuracy=best_val_acc,
                    patience_counter=patience_counter,
                    random_state=torch.get_rng_state()
                )

            # Step scheduler
            scheduler.step()

        # Final test evaluation
        print("\n" + "=" * 60)
        print("Final Test Evaluation")
        print("=" * 60)

        # Load best checkpoint for testing
        best_checkpoint = logger.get_best_checkpoint(metric="val_accuracy", mode="max")
        print(f"Loading best checkpoint: {best_checkpoint}")

        checkpoint_data = logger.load_checkpoint(
            checkpoint_path=best_checkpoint,
            model=model,
            device=device
        )

        test_loss, test_acc, test_per_class = validate(model, test_loader, criterion, device)

        # Log test results
        test_metrics = {
            "loss": test_loss,
            "accuracy": test_acc,
            **test_per_class
        }
        logger.log_evaluation(test_metrics, split="test")

        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"\nPer-class accuracy:")
        for class_id in range(10):
            acc = test_per_class[f"class_{class_id}_acc"]
            print(f"  Class {class_id}: {acc:.4f}")

        # Checkpoint management
        print("\n" + "=" * 60)
        print("Checkpoint Management")
        print("=" * 60)

        # List all checkpoints
        print("\nAll checkpoints before cleanup:")
        checkpoints_df = logger.list_checkpoints()
        print(checkpoints_df.to_string())

        # Cleanup old checkpoints
        logger.cleanup_checkpoints(
            keep_best=config.checkpoint.keep_best,
            keep_last=config.checkpoint.keep_last,
            metric="val_accuracy",
            mode="max"
        )

        print("\nCheckpoints after cleanup:")
        checkpoints_df = logger.list_checkpoints()
        print(checkpoints_df.to_string())

        print(f"\n✓ Experiment completed! Results saved to {logger.exp_dir}")


if __name__ == "__main__":
    main()
