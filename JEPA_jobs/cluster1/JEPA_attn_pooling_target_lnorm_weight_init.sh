#!/bin/bash -l
#SBATCH --job-name=jepa_training_wo_poses
#SBATCH --output=/mnt/md0/erik/jepa_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:a6000:6
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Activate Conda
#module add python
#module add cuda/12.3.0
#module add gcc/12.1.0
source ./mtr_venv/bin/activate

#mkdir -p $TMPDIR/processed_scenarios_training
#mkdir -p $TMPDIR/processed_scenarios_validation
# find the data
#STORAGE_DIR="$(ws_find jepa_data)"
# the -P parameter defines the number of parallel processes, something like 4-8 should work well
#ls -1 $STORAGE_DIR/validation | xargs -P 8 -I{} tar xzf $STORAGE_DIR/validation/{} -C $TMPDIR
#ls -1 $STORAGE_DIR/training | xargs -P 8 -I{} tar xzf $STORAGE_DIR/training/{} -C $TMPDIR

# cp $WORK/processed_scenarios_training_infos.pkl $TMPDIR/processed_scenarios_training_infos.pkl
# cp $WORK/processed_scenarios_val_infos.pkl $TMPDIR/processed_scenarios_val_infos.pkl

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

cd /mnt/md0/erik/MTR_with_poses/tools

export OMP_NUM_THREADS=36

torchrun --nproc_per_node=6 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /mnt/md0/erik/MTR_with_poses/tools/cfgs/waymo/jepa_loss_trial_no_timestamps_batch_norm.yaml --batch_size=120 --epochs=50 --extra_tag=Overfit_120 --tcp_port=$PORT --workers=6 --max_ckpt_save_num=300 --not_eval_with_train --single_overfit=120 --set DATA_CONFIG.DATA_ROOT /mnt/md0/erik/dataset

# Deactivate the virtual environment at the end
deactivate
