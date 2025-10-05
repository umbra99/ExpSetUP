#!/usr/bin/env python
"""
Verification script for ExpSetUP installation.

Run this script after installation to verify that ExpSetUP is properly installed
and all components are working correctly.
"""

import sys
import tempfile
from pathlib import Path


def check_imports():
    """Check if all modules can be imported."""
    print("Checking imports...")
    try:
        import expsetup
        from expsetup import ExperimentLogger, CheckpointManager, Config
        print("✓ All modules imported successfully")
        print(f"  ExpSetUP version: {expsetup.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def check_dependencies():
    """Check if all dependencies are available."""
    print("\nChecking dependencies...")
    dependencies = [
        ("torch", "PyTorch"),
        ("yaml", "PyYAML"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("tensorboard", "TensorBoard")
    ]

    all_ok = True
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"✗ {display_name} is NOT installed")
            all_ok = False

    return all_ok


def test_config():
    """Test Config class functionality."""
    print("\nTesting Config class...")
    try:
        from expsetup import Config

        # Test dict creation
        config = Config({"lr": 0.001, "batch_size": 32})
        assert config.lr == 0.001
        assert config["batch_size"] == 32

        # Test save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            config.save(config_path)
            assert config_path.exists()

        print("✓ Config class works correctly")
        return True
    except Exception as e:
        print(f"✗ Config class test failed: {e}")
        return False


def test_checkpoint_manager():
    """Test CheckpointManager functionality."""
    print("\nTesting CheckpointManager...")
    try:
        import torch
        import torch.nn as nn
        from expsetup import CheckpointManager

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)
            model = SimpleModel()

            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                model=model,
                epoch=1,
                metrics={"accuracy": 0.95}
            )
            assert checkpoint_path.exists()

            # Load checkpoint
            new_model = SimpleModel()
            checkpoint_data = manager.load_checkpoint(checkpoint_path, new_model)
            assert checkpoint_data["epoch"] == 1

        print("✓ CheckpointManager works correctly")
        return True
    except Exception as e:
        print(f"✗ CheckpointManager test failed: {e}")
        return False


def test_experiment_logger():
    """Test ExperimentLogger functionality."""
    print("\nTesting ExperimentLogger...")
    try:
        import torch
        import torch.nn as nn
        from expsetup import ExperimentLogger

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                "test_exp",
                base_dir=tmpdir,
                use_tensorboard=False
            )

            # Test config logging
            logger.log_config({"lr": 0.001})
            assert (logger.config_dir / "config.yaml").exists()

            # Test metrics logging
            logger.log_metrics({"loss": 0.5}, step=1)

            # Test checkpoint saving
            model = SimpleModel()
            checkpoint_path = logger.save_checkpoint(model, epoch=1)
            assert checkpoint_path.exists()

            logger.close()

        print("✓ ExperimentLogger works correctly")
        return True
    except Exception as e:
        print(f"✗ ExperimentLogger test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("ExpSetUP Installation Verification")
    print("=" * 60)

    results = []

    # Run all checks
    results.append(("Import Check", check_imports()))
    results.append(("Dependencies Check", check_dependencies()))
    results.append(("Config Test", test_config()))
    results.append(("CheckpointManager Test", test_checkpoint_manager()))
    results.append(("ExperimentLogger Test", test_experiment_logger()))

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! ExpSetUP is correctly installed.")
        print("\nNext steps:")
        print("  1. Check out the examples: python examples/basic_usage.py")
        print("  2. Read the documentation: README.md")
        print("  3. Start tracking your experiments!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Make sure all dependencies are installed: pip install -e .[dev]")
        print("  2. Check that PyTorch is properly installed")
        print("  3. Verify Python version is 3.8 or higher")
        return 1


if __name__ == "__main__":
    sys.exit(main())
