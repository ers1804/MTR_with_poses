#!/bin/bash
#SBATCH --job-name=create_venv
#SBATCH --output=/home/slurm/shared_folder/erik/mtr_env_creation.txt
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=1G
#SBATCH -D /home/slurm

# Set up a variable for the virtual environment directory
VENV_DIR="/home/slurm/venvs/mtr_pose"

# Create a new virtual environment; this directory must not exist
python3.8 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
pip install numpy torch torchvision tensorboardX easydict pyyaml scikit-image tqdm

# Optionally, run a Python script using this virtual environment
# python my_script.py

# Deactivate the virtual environment at the end
deactivate
