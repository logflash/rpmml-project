Unfortunately, MMD is Python 3.8 whereas DiffuserLite is 3.10, so there will be
a lot of dependency issues, so let's just try to run it in our own conda environment like
the authors suggested

First, I suggest downloading miniconda3 to this project directory

Run `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
then `bash Miniconda3-latest-Linux-x86_64.sh`

Continue through the installation until it asks for a path, which you should type:
`/scratch/network/{netid}/rpmml-project/miniconda3`

Finally, say yes when it asks to rewrite your .bashrc with this new path for conda

now `cd mmd`

Now we can just follow what the authors of mmd wrote in their readme:

conda env create -f environment.yml
conda activate mmd

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

cd deps/torch_robotics
pip install -e .
cd ../experiment_launcher
pip install -e .
cd ../motion_planning_baselines
pip install -e .
cd ../..

pip install --upgrade setuptools==70.2.0 (my addition for the next steps to work)
 

pip install -e .

bash setup.sh


Now, let's test if you set this up properly. We'll perform a test inference with some of their pretrained models / collected trajectories

Run:

gdown --id 1Onw0s1pDsMLDfJVOAqmNme4eVVoAkkjz
tar -xJvf data_trajectories.tar.xz

gdown --id 1WO3tpvg-HU0m9RyDvGyfDamo7roBYMud
tar -xJvf data_trained_models.tar.xz

These should give you data_trained_models/ and data_trajectories/ in your mdd directory,
but for now (a bug randomly came up when I was doing this the second time), let's also copy
data_trained_models/ and data_trajectories/ so that they exist in your rpmml-project dir (one up from mmd/)

Then, let's test with `cd scripts/inference`
`python3 inference_multi_agent.py`

Now within the inference directory, you should see a gif of the run. (only the last nested directory will have a gif)

Good job on getting this far!