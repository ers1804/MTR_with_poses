#!/bin/bash -l
#SBATCH --job-name=jepa_eval
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/jepa_eval_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Activate Conda
module add python
module add cuda/12.6.2
#module add gcc/12.1.0
#source $WORK/mtr_venv_helma/bin/activate
conda activate helma_conda

#mkdir -p $TMPDIR/processed_scenarios_training
#mkdir -p $TMPDIR/processed_scenarios_validation
# find the data
STORAGE_DIR="$(ws_find jepa_data)"
# the -P parameter defines the number of parallel processes, something like 4-8 should work well
#ls -1 $STORAGE_DIR/archives_val | xargs -P 1 -I{} tar xzf $STORAGE_DIR/archives_val/{} -C $TMPDIR
#ls -1 $STORAGE_DIR/archives_train | xargs -P 1 -I{} tar xzf $STORAGE_DIR/archives_train/{} -C $TMPDIR

#mkdir $TMPDIR/processed_scenarios_training
mkdir $TMPDIR/processed_scenarios_validation

#find $STORAGE_DIR/archives_train -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'mkdir -p $TMPDIR/tmp_{} && tar xzf {} -C $TMPDIR/tmp_{} && mv $TMPDIR/tmp_{}/processed_scenarios_training/* $TMPDIR/processed_scenarios_training'
#find $STORAGE_DIR/archives_train -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'rm -rf $TMPDIR/tmp_{}'

find $STORAGE_DIR/archives_val -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'mkdir -p $TMPDIR/tmp_{} && tar xzf {} -C $TMPDIR/tmp_{} && mv $TMPDIR/tmp_{}/processed_scenarios_validation/* $TMPDIR/processed_scenarios_validation'
find $STORAGE_DIR/archives_val -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'rm -rf $TMPDIR/tmp_{}'
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

export OMP_NUM_THREADS=32

torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${PORT} test.py --launcher pytorch --tcp_port=$PORT --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data_jepa_with_decoder.yaml --batch_size=30 --save_to_file --workers=0 --eval_all --extra_tag Full_Training_1_1_0001_40_Epochs_2_New --eval_tag Full_Training_1_1_0001_40_Epochs_2_New_eval --ckpt_dir /home/atuin/v103fe/v103fe12/MTR_helma/MTR_with_poses/output/home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/mtr+100_percent_data_jepa_with_decoder/Full_Training_1_1_0001_40_Epochs_2_New/ckpt --set DATA_CONFIG.DATA_ROOT $TMPDIR

# Deactivate the virtual environment at the end
conda deactivate
