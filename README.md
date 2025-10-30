# ECE531 Project

Installation instructions:
```bash
# Clone this repository and install uv
cd /scratch/network/${USER}
git clone git@github.com:logflash/rpmml-project.git
cd rpmml-project

# Package setup
pip install uv --target ..
uv venv --python=3.10
source .venv/bin/activate
uv run --no-cache python scripts/install_all.py # if disk quota exceeded, run `pip cache purge` or `uv clean`

# .bashrc
echo "source /scratch/network/${USER}/rpmml-project/.venv/bin/activate" >> ~/.bashrc

# Pytorch setup
python -m ensurepip --upgrade
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```