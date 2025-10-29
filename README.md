# ECE531 Project

Installation instructions:
```bash
# Package setup
pip install uv
uv venv --python=3.10
source .venv/bin/activate
uv run python scripts/install_all.py

# Pytorch setup
python -m ensurepip --upgrade
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```