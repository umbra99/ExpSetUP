# ExpSetUP - Setup and Installation Guide

This guide covers how to install ExpSetUP in your experiment projects.

## Table of Contents

1. [Setting Up GitHub Authentication](#setting-up-github-authentication)
2. [Installing in Your Projects](#installing-in-your-projects)
3. [Installing Specific Versions](#installing-specific-versions)
4. [Updating the Package](#updating-the-package)
5. [Using in Docker](#using-in-docker)
6. [Troubleshooting](#troubleshooting)

## Setting Up GitHub Authentication

To install private packages, you need to authenticate with GitHub.

### Option 1: Personal Access Token (Recommended)

1. **Generate a Personal Access Token (PAT)**:
   - Go to GitHub Settings → Developer settings
   - Click "Personal access tokens" → "Tokens (classic)"
   - Click "Generate new token (classic)"
   - Settings:
     - **Note**: "ExpSetUP Package Access"
     - **Expiration**: Choose appropriate duration
     - **Scopes**: Select `repo` (full control of private repositories)
   - Click "Generate token"
   - **IMPORTANT**: Copy the token immediately (you won't see it again)

2. **Save Token Securely**:
   ```bash
   # Store in environment variable (add to ~/.bashrc or ~/.zshrc)
   export GITHUB_TOKEN="ghp_your_token_here"
   ```

### Option 2: SSH Keys (Alternative)

1. **Generate SSH Key** (if you don't have one):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. **Add to SSH Agent**:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. **Add to GitHub**:
   - Go to GitHub Settings → SSH and GPG keys
   - Click "New SSH key"
   - Paste contents of `~/.ssh/id_ed25519.pub`
   - Click "Add SSH key"

## Installing in Your Projects

### Method 1: Direct Installation with pip

```bash
# Using HTTPS with token
pip install git+https://${GITHUB_TOKEN}@github.com/yourusername/ExpSetUP.git

# Using SSH (if SSH keys are set up)
pip install git+ssh://git@github.com/yourusername/ExpSetUP.git

# Using HTTPS (will prompt for credentials)
pip install git+https://github.com/yourusername/ExpSetUP.git
```

### Method 2: Using requirements.txt

Create or update your `requirements.txt`:

```
# requirements.txt

# Core dependencies
torch>=1.9.0
numpy>=1.21.0

# ExpSetUP package
git+https://github.com/yourusername/ExpSetUP.git@main

# Other dependencies...
```

Install all requirements:

```bash
# Using environment variable for token
pip install -r requirements.txt
```

### Method 3: Using pyproject.toml (Poetry/Modern Projects)

```toml
[tool.poetry.dependencies]
python = "^3.8"
expsetup = {git = "https://github.com/yourusername/ExpSetUP.git", branch = "main"}
```

Or with Poetry CLI:

```bash
poetry add git+https://github.com/yourusername/ExpSetUP.git
```

### Method 4: Using pip with .netrc (Automatic Authentication)

Create/edit `~/.netrc`:

```
machine github.com
login your-github-username
password ghp_your_token_here
```

Set permissions:

```bash
chmod 600 ~/.netrc
```

Then install without specifying token:

```bash
pip install git+https://github.com/yourusername/ExpSetUP.git
```

## Installing Specific Versions

### Install Specific Branch

```bash
pip install git+https://github.com/yourusername/ExpSetUP.git@develop
```

### Install Specific Tag/Release

```bash
pip install git+https://github.com/yourusername/ExpSetUP.git@v0.1.0
```

### Install Specific Commit

```bash
pip install git+https://github.com/yourusername/ExpSetUP.git@abc1234
```

## Updating the Package

### Update Version Number

When the package maintainer updates the version, edit `pyproject.toml`:

```toml
[project]
version = "0.2.0"  # Increment version
```

And create a git tag (optional but recommended):

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

### Update in Your Projects

```bash
# Upgrade to latest version
pip install --upgrade git+https://github.com/yourusername/ExpSetUP.git

# Or reinstall specific version
pip install --force-reinstall git+https://github.com/yourusername/ExpSetUP.git@v0.2.0
```

## Using in Docker

### Dockerfile Example

```dockerfile
FROM python:3.9-slim

# Set up git for pip install
RUN apt-get update && apt-get install -y git

# Copy requirements
COPY requirements.txt .

# Install dependencies (token passed as build arg)
ARG GITHUB_TOKEN
RUN pip install git+https://${GITHUB_TOKEN}@github.com/yourusername/ExpSetUP.git

# Rest of your Dockerfile...
```

Build with:

```bash
docker build --build-arg GITHUB_TOKEN=${GITHUB_TOKEN} -t myimage .
```

## Troubleshooting

### Issue: Authentication Failed

**Solution**:
- Verify token has `repo` scope
- Check token hasn't expired
- Ensure token is correctly set in environment variable

### Issue: Repository Not Found

**Solution**:
- Verify repository URL is correct
- Ensure you have access to the private repository
- Check if using correct username

### Issue: Permission Denied (SSH)

**Solution**:
- Verify SSH key is added to GitHub
- Test SSH connection: `ssh -T git@github.com`
- Check SSH agent is running: `ssh-add -l`

### Issue: Package Not Updating

**Solution**:
```bash
# Clear pip cache
pip cache purge

# Force reinstall
pip install --force-reinstall --no-cache-dir git+https://github.com/yourusername/ExpSetUP.git
```

### Issue: Import Error After Installation

**Solution**:
```bash
# Verify installation
pip show expsetup

# Check if in editable mode (development)
pip list | grep expsetup

# Reinstall if needed
pip uninstall expsetup
pip install git+https://github.com/yourusername/ExpSetUP.git
```

## Best Practices

1. **Version Control**: Always tag releases with semantic versioning
2. **Documentation**: Keep README.md updated with new features
3. **Testing**: Add tests before pushing updates
4. **Changelog**: Maintain a CHANGELOG.md for tracking changes
5. **Security**: Never commit tokens or credentials to the repository
6. **Dependencies**: Keep `pyproject.toml` dependencies up to date

## Quick Reference Commands

```bash
# Install latest version
pip install git+https://github.com/yourusername/ExpSetUP.git

# Install specific version
pip install git+https://github.com/yourusername/ExpSetUP.git@v0.1.0

# Update to latest
pip install --upgrade git+https://github.com/yourusername/ExpSetUP.git

# Uninstall
pip uninstall expsetup

# Development installation (editable)
cd /path/to/ExpSetUP
pip install -e .
```

## Next Steps

1. Install the package in your experiment project
2. Review the [README.md](README.md) for usage examples
3. Check out the [examples](examples/) directory
4. Start tracking your experiments!

For issues or questions, open an issue on the GitHub repository.
