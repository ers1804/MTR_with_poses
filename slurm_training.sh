#!/bin/bash -l
#SBATCH --job-name=train_mtr
#SBATCH --output=/home/slurm/shared_folder/erik/mtr_training_%j.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -D /home/slurm
#SBATCH --gres=gpu:rtx4090:4
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Set up a variable for the virtual environment directory
VENV_DIR="/home/slurm/venvs/mtr_pose"

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Optionally, run a Python script using this virtual environment
# python my_script.py

set -x

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

cd /home/slurm/working_dir/MTR/tools

export OMP_NUM_THREADS=32

torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /home/slurm/working_dir/MTR/tools/cfgs/waymo/mtr+100_percent_data_slurm.yaml --batch_size=28 --epochs=120 --extra_tag=MTR_wo_poses_new --tcp_port=$PORT --workers=8

# Deactivate the virtual environment at the end
deactivate
