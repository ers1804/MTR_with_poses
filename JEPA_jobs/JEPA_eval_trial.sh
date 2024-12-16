#!/bin/bash -l
#SBATCH --job-name=jepa_training_wo_poses
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/jepa_pretraining_attn_pooling_7_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
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
ls -1 $STORAGE_DIR/archives_val | xargs -P 8 -I{} tar xzf $STORAGE_DIR/archives_val/{} -C $TMPDIR
ls -1 $STORAGE_DIR/archives_train | xargs -P 8 -I{} tar xzf $STORAGE_DIR/archives_train/{} -C $TMPDIR

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

export OMP_NUM_THREADS=16

torchrun --nproc_per_node=1 test.py --launcher pytorch --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size=50 --extra_tag=Full_Training_20_1_1_0001_100 --eval_tag=Full_Training_20_1_1_0001_100_eval --ckpt_dir=/home/atuin/v103fe/v103fe12/MTR/output/home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+20_percent_data_jepa_with_decoder/Full_Training_20_1_1_0001_100/ckpt --save_to_file --workers=8 --eval_all --set DATA_CONFIG.DATA_ROOT $TMPDIR

# Deactivate the virtual environment at the end
deactivate
