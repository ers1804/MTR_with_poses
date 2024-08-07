#!/bin/bash -l
#SBATCH --job-name=mtr_training_wo_poses_eval
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/mtr_training_wo_poses_eval_single_%j.txt
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

export OMP_NUM_THREADS=12

torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${PORT} test.py --launcher pytorch --ckpt /home/atuin/v103fe/v103fe12/MTR/output/home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data/MTR_wo_poses_8a100/ckpt/checkpoint_epoch_30.pth --ckpt_dir /home/atuin/v103fe/v103fe12/MTR/output/home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data/MTR_wo_poses_8a100/ckpt --save_to_file --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data.yaml --batch_size=1 --extra_tag=MTR_wo_poses_8a100 --eval_tag=MTR_wo_poses_8a100_eval_single --tcp_port=$PORT --workers=12 --single_scenario --single_eval_output /home/atuin/v103fe/v103fe12/MTR_wo_poses_8a100_single.json --scenario_log_file /home/atuin/v103fe/v103fe12/MTR_wo_poses_8a100_single.txt --set DATA_CONFIG.DATA_ROOT $TMPDIR

# Deactivate the virtual environment at the end
deactivate
