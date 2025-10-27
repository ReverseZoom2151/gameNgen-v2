# Optional Dependencies Guide

This guide covers optional dependencies that enhance GameNGen but aren't required for basic functionality.

---

## I3D Model for Accurate FVD

The paper uses **FVD (Fréchet Video Distance)** with an **I3D (Inflated 3D ConvNet)** model for video quality evaluation.

### Quick Version (Simplified FVD)

**No installation needed!** The built-in implementation works out of the box with a simplified I3D.

```bash
# Works immediately
from src.utils.fvd import FVDCalculator
fvd_calc = FVDCalculator()
```

This gives reasonable FVD scores but isn't identical to the paper's implementation.

---

### Full Version (Paper-Accurate FVD)

For FVD scores matching the paper exactly, install the full I3D model:

#### Option 1: Install from Source (Recommended)

```bash
# Clone the I3D repository
git clone https://github.com/piergiaj/pytorch-i3d.git
cd pytorch-i3d

# Install
pip install -e .

# Download pretrained weights
# Follow instructions at: https://github.com/piergiaj/pytorch-i3d
```

#### Option 2: Manual Installation

```bash
# Download the I3D code directly
wget https://raw.githubusercontent.com/piergiaj/pytorch-i3d/master/pytorch_i3d.py

# Place in your Python path or project
```

#### Verification

```python
# Test if I3D is available
try:
    from pytorch_i3d import InceptionI3d
    print("✓ Full I3D available")
except ImportError:
    print("⚠ Using simplified I3D (functional but not paper-exact)")
```

---

## Paper's FVD Results

**With full I3D model:**
- 16 frames: FVD = 114.02
- 32 frames: FVD = 186.23

**With simplified I3D:**
- Results will be different but still useful for relative comparisons

---

## ViZDoom (Required for Tier 2 & 3)

```bash
pip install vizdoom
```

If installation fails, see: https://github.com/Farama-Foundation/ViZDoom

---

## Development Tools

```bash
pip install -r requirements-dev.txt
```

This includes:
- Code formatters (black, isort)
- Linters (flake8, pylint)
- Testing tools (pytest)
- Documentation tools (sphinx)
- Jupyter notebooks

---

## Weights & Biases (Optional Tracking)

For experiment tracking with W&B:

```bash
# Already in requirements.txt as optional
wandb login
```

Update config:
```yaml
logging:
  use_wandb: true
  wandb_project: "gamengen"
```

---

## Pre-commit Hooks (Optional)

Auto-format code before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Now code will be formatted automatically on commit
```

---

## Docker (Optional)

For containerized development:

```dockerfile
# Coming soon: Dockerfile for full setup
```

---

## Summary

**Required:**
- PyTorch with CUDA
- diffusers, stable-baselines3, gymnasium
- Basic metrics (lpips, scikit-image)

**Optional but Recommended:**
- ViZDoom (for Tier 2 & 3)
- Development tools (black, pytest, etc.)

**Optional for Research:**
- Full I3D (for paper-exact FVD)
- Weights & Biases (experiment tracking)

**Your code works without any optional dependencies!**
