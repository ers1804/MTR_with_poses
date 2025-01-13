#!/bin/bash -l
#SBATCH --job-name=jepa_loss_trial
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/jepa_complete_time_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:h100:4
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Activate Conda
module add python
module add cuda/12.6.2
#module add gcc/12.1.0
source $WORK/mtr_venv_helma/bin/activate

#mkdir -p $TMPDIR/processed_scenarios_training
#mkdir -p $TMPDIR/processed_scenarios_validation
# find the data
STORAGE_DIR="$(ws_find jepa_data)"
# the -P parameter defines the number of parallel processes, something like 4-8 should work well
#ls -1 $STORAGE_DIR/archives_val | xargs -P 1 -I{} tar xzf $STORAGE_DIR/archives_val/{} -C $TMPDIR
#ls -1 $STORAGE_DIR/archives_train | xargs -P 1 -I{} tar xzf $STORAGE_DIR/archives_train/{} -C $TMPDIR

mkdir $TMPDIR/processed_scenarios_training
mkdir $TMPDIR/processed_scenarios_validation

find $STORAGE_DIR/archives_train -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'mkdir -p $TMPDIR/tmp_{} && tar xzf {} -C $TMPDIR/tmp_{} && mv $TMPDIR/tmp_{}/processed_scenarios_training/* $TMPDIR/processed_scenarios_training'
find $STORAGE_DIR/archives_train -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'rm -rf $TMPDIR/tmp_{}'

find $STORAGE_DIR/archives_val -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'mkdir -p $TMPDIR/tmp_{} && tar xzf {} -C $TMPDIR/tmp_{} && mv $TMPDIR/tmp_{}/processed_scenarios_validation/* $TMPDIR/processed_scenarios_validation'
find $STORAGE_DIR/archives_val -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'rm -rf $TMPDIR/tmp_{}'
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

cd /home/atuin/v103fe/v103fe12/MTR_helma/MTR_with_poses/tools

export OMP_NUM_THREADS=128


torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/jepa_loss_trial.yaml --batch_size=120 --epochs=150 --extra_tag=Training_1_1_0001_150_Epochs_512_No_Attn --tcp_port=$PORT --workers=16 --max_ckpt_save_num=150 --ckpt_save_interval=2 --set DATA_CONFIG.DATA_ROOT $TMPDIR MODEL.CONTEXT_ENCODER.USE_ATTN_POOL False MODEL.CONTEXT_ENCODER.D_MODEL 512 OPTIMIZATION.DECAY_STEP_LIST []


# Deactivate the virtual environment at the end
deactivate
