#!/bin/bash -l
#SBATCH --job-name=jepa_training_wo_poses
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/jepa_pretraining_attn_pooling_7_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2 -C a100_80
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Activate Conda
module add python
module add cuda/12.3.0
#module add gcc/12.1.0
source $WORK/mtr_venv/bin/activate

#mkdir -p $TMPDIR/processed_scenarios_training
#mkdir -p $TMPDIR/processed_scenarios_validation
# find the data
STORAGE_DIR="$(ws_find jepa_data)"
# the -P parameter defines the number of parallel processes, something like 4-8 should work well
ls -1 $STORAGE_DIR/validation | xargs -P 8 -I{} tar xzf $STORAGE_DIR/validation/{} -C $TMPDIR
ls -1 $STORAGE_DIR/training | xargs -P 8 -I{} tar xzf $STORAGE_DIR/training/{} -C $TMPDIR

cp $WORK/processed_scenarios_training_infos.pkl $TMPDIR/processed_scenarios_training_infos.pkl
cp $WORK/processed_scenarios_val_infos.pkl $TMPDIR/processed_scenarios_val_infos.pkl

# Unpack training data to $TMPDIR
#cd $TMPDIR
#tar xzf $WORK/mtr_training_wo_poses.tar.gz

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

torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data_jepa_attn_pooling.yaml --batch_size=58 --epochs=5 --extra_tag=Jepa_wo_poses_attn_pooling_eval_trial --tcp_port=$PORT --workers=8 --max_ckpt_save_num=5 --ckpt_save_interval=1 --eval_batch_size=58 --set DATA_CONFIG.DATA_ROOT $TMPDIR

# Deactivate the virtual environment at the end
deactivate
