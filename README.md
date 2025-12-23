# ECE531 Project

## Installation

```bash
# Clone this repository and install uv
cd /scratch/network/${USER}
git clone git@github.com:logflash/rpmml-project.git
cd rpmml-project

# Package setup
pip install uv --target ..
uv venv --python=3.10
source .venv/bin/activate
uv run --no-cache python setup/install_all.py
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
uv pip install --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Offline Dataset Building

```bash
cd timeskip-diffuser/src/timeskip_diffuser/datasets/point_maze/offline_skip

# Build each offline dataset separately
python offline_dataset_generator.py --env_name "open"
python offline_dataset_generator.py --env_name "umaze"
python offline_dataset_generator.py --env_name "medium"
```

## Training Pipelines

```bash
cd timeskip-diffuser/src/timeskip_diffuser/pipelines

# Train each model separately
python train_diffuser.py --config configs/config_[...].yaml

# The result will be stored in runs/
```

## Run Experiments

```bash
cd timeskip-diffuser/src/timeskip_diffuser/pipelines

# Run the experiment pipeline
run_experiments_script.py --config experiments_config.yaml

# The result will be stored in runs/
```