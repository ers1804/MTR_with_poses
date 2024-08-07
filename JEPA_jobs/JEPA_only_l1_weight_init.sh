#!/bin/bash -l
#SBATCH --job-name=jepa_l1_layer_weight
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/jepa_pretraining_12_%j.txt
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

export OMP_NUM_THREADS=64

torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data_jepa_only_l1_layer.yaml --batch_size=232 --epochs=100 --extra_tag=L1_Layer_Init --tcp_port=$PORT --workers=8 --ckpt_save_interval=5 --max_ckpt_save_num=300 --set DATA_CONFIG.DATA_ROOT $TMPDIR

# Deactivate the virtual environment at the end
deactivate
