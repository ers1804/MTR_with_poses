#!/bin/bash -l
#SBATCH --job-name=mtr_training_w_poses
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/mtr_training_poses_mlp_mlp_60%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:8 -C a100_80
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Activate Conda
module add python
module add cuda/12.3.0
#module add gcc/12.1.0
source $WORK/mtr_venv/bin/activate

# Unpack training data to $TMPDIR
cd $TMPDIR
tar xzf $WORK/mtr_training.tar.gz

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

export OMP_NUM_THREADS=64

torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data_with_poses_MLP_MLP.yaml --batch_size=160 --epochs=60 --extra_tag=MTR_MLP_MLP_8a100_60 --tcp_port=$PORT --workers=8 --not_eval_with_train --max_ckpt_save_num=60 --set DATA_CONFIG.DATA_ROOT $TMPDIR DATA_CONFIG.POSE_DIR $TMPDIR/poses

# Deactivate the virtual environment at the end
deactivate
