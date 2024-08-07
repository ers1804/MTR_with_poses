#!/bin/bash -l
#SBATCH --job-name=mtr_training
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/mtr_training%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4 -C a100_80
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Activate Conda
module add python
module add cuda/12.1.1
module add cudnn/8.9.6.50-12.x
#module add gcc/12.1.0
conda activate mtr_training

# Unpack training data to $TMPDIR
cd $TMPDIR
tar xzf $WORK/mtr_training_wo_poses.tar.gz

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

cd /home/atuin/v103fe/v103fe12/MTR/tools

export OMP_NUM_THREADS=32

torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data.yaml --batch_size=80 --epochs=30 --extra_tag=MTR_wo_poses_a100 --tcp_port=$PORT --workers=8 --set DATA_CONFIG.DATA_ROOT $TMPDIR

# Deactivate the virtual environment at the end
conda deactivate
