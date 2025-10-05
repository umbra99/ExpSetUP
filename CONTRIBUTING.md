# Contributing to ExpSetUP

Thank you for your interest in contributing to ExpSetUP! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ExpSetUP.git
   cd ExpSetUP
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

## Running Tests

Run the test suite using pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=expsetup tests/

# Run specific test file
pytest tests/test_logger.py

# Run with verbose output
pytest -v tests/
```

## Code Style

We use several tools to maintain code quality:

### Black (Code Formatting)

```bash
# Format all code
black expsetup/ tests/ examples/

# Check formatting without changes
black --check expsetup/
```

### isort (Import Sorting)

```bash
# Sort imports
isort expsetup/ tests/ examples/

# Check import sorting
isort --check-only expsetup/
```

### flake8 (Linting)

```bash
# Run linter
flake8 expsetup/ tests/
```

### mypy (Type Checking)

```bash
# Run type checker
mypy expsetup/
```

### Run All Checks

```bash
# Format code
black expsetup/ tests/ examples/
isort expsetup/ tests/ examples/

# Run checks
flake8 expsetup/ tests/
mypy expsetup/
pytest tests/
```

## Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes:**
   - Write clear, concise code
   - Follow the existing code style
   - Add docstrings to new functions/classes
   - Update tests as needed

3. **Add tests:**
   - Add tests for new features
   - Ensure existing tests still pass
   - Aim for good test coverage

4. **Update documentation:**
   - Update README.md if needed
   - Add docstrings to new code
   - Update examples if relevant

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

6. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request:**
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Provide a clear description of your changes

## Pull Request Guidelines

- **Title:** Use a clear, descriptive title
- **Description:** Explain what changes you made and why
- **Tests:** Ensure all tests pass
- **Documentation:** Update docs if needed
- **Code Style:** Follow the project's code style
- **Commits:** Keep commits focused and well-described

## Commit Message Guidelines

Follow these conventions:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when relevant

Examples:
```
Add checkpoint compression feature

Fix issue with metric tracking in evaluation mode

Update documentation for Config class

Refactor checkpoint manager for better performance
```

## Adding New Features

When adding new features:

1. **Discuss first:** Open an issue to discuss major changes
2. **Keep it focused:** One feature per pull request
3. **Write tests:** Add comprehensive tests
4. **Document:** Update relevant documentation
5. **Examples:** Add examples if appropriate

## Reporting Bugs

When reporting bugs, include:

1. **Description:** Clear description of the bug
2. **Reproduction:** Steps to reproduce the behavior
3. **Expected behavior:** What you expected to happen
4. **Actual behavior:** What actually happened
5. **Environment:**
   - OS and version
   - Python version
   - PyTorch version
   - ExpSetUP version
6. **Code sample:** Minimal code to reproduce the issue

## Feature Requests

When requesting features:

1. **Use case:** Describe your use case
2. **Proposed solution:** How you envision it working
3. **Alternatives:** Alternative solutions you've considered
4. **Additional context:** Any other relevant information

## Code Review Process

1. Maintainers will review your pull request
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Your contribution will be included in the next release

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`

## Questions?

If you have questions:

- Open an issue on GitHub
- Check existing issues and pull requests
- Review the documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to ExpSetUP! 🎉
