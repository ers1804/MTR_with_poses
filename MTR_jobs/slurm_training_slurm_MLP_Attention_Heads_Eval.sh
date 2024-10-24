#!/bin/bash -l
#SBATCH --job-name=mtr_training_w_poses_eval
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/mtr_training_poses_8_heads_eval_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1 -C a100_80
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

export OMP_NUM_THREADS=12

torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${PORT} test.py --launcher pytorch --eval_all --save_to_file --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data_with_poses_MLP_Attention_Heads.yaml --batch_size=100 --extra_tag=MTR_MLP_Attention_8a100_8_Heads --eval_tag=MTR_MLP_Attention_8a100_8_Heads_eval --tcp_port=$PORT --workers=12 --set DATA_CONFIG.DATA_ROOT $TMPDIR DATA_CONFIG.POSE_DIR $TMPDIR/poses

# Deactivate the virtual environment at the end
deactivate
