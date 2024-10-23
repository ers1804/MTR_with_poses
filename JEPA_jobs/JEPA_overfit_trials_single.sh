#!/bin/bash -l
#SBATCH --job-name=base_overfit_trial
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/base_overfit_%j.txt
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
#STORAGE_DIR="$(ws_find jepa_data)"
# the -P parameter defines the number of parallel processes, something like 4-8 should work well
#ls -1 $STORAGE_DIR/archives_val | xargs -P 8 -I{} tar xzf $STORAGE_DIR/archives_val/{} -C $TMPDIR
#ls -1 $STORAGE_DIR/archives_train | xargs -P 8 -I{} tar xzf $STORAGE_DIR/archives_train/{} -C $TMPDIR
cp $WORK/processed_scenarios_training_infos.pkl $TMPDIR/processed_scenarios_training_infos.pkl
cp $WORK/processed_scenarios_val_infos.pkl $TMPDIR/processed_scenarios_val_infos.pkl
mkdir $TMPDIR/processed_scenarios_training
cp $WORK/sample_103718491a34269.pkl $TMPDIR/processed_scenarios_training/sample_103718491a34269.pkl

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

export OMP_NUM_THREADS=8

torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data.yaml --batch_size=1 --epochs=10000 --extra_tag=Overfit_103718491a34269_CLR_10000 --tcp_port=$PORT --workers=8 --max_ckpt_save_num=1 --not_eval_with_train --scenario_id 103718491a34269 --set DATA_CONFIG.DATA_ROOT $TMPDIR DATA_CONFIG.JEPA True MODEL.CONTEXT_ENCODER.NUM_INPUT_ATTR_AGENT 109

# Deactivate the virtual environment at the end
deactivate
