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
uv run --no-cache python scripts/install_all.py
# if disk quota exceeded, run `pip cache purge` or `uv clean`

# Mujoco + .bashrc setup
cd ..
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xvf mujoco210-linux-x86_64.tar.gz
cd rpmml-project
mkdir -p ~/.mujoco
mv mujoco210 ~/.mujoco/
echo "export MUJOCO_PATH=$HOME/.mujoco/mujoco210" >> ~/.bashrc
echo "export MUJOCO_PLUGIN_PATH=$HOME/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ih2422/.mujoco/mujoco210/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
echo "source /scratch/network/${USER}/rpmml-project/.venv/bin/activate" >> ~/.bashrc

# Pytorch setup
python -m ensurepip --upgrade
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```