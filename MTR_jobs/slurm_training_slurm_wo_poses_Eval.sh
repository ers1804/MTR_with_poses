#!/bin/bash -l
#SBATCH --job-name=jepa_eval
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/jepa_20_eval_%j.txt
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

export OMP_NUM_THREADS=12

torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${PORT} test.py --launcher pytorch --eval_all --save_to_file --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+20_percent_data_jepa_with_decoder.yaml --ckpt_dir=/home/atuin/v103fe/v103fe12/MTR/output/home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+20_percent_data_jepa_with_decoder/Normal_Loss_40/ckpt --batch_size=100 --extra_tag=Normal_Loss_40 --eval_tag=Normal_Loss_40_eval --tcp_port=$PORT --workers=12 --set DATA_CONFIG.DATA_ROOT $TMPDIR

# Deactivate the virtual environment at the end
deactivate
