#!/bin/bash -l
#SBATCH --job-name=jepa_loss_trial
#SBATCH --output=/home/atuin/v103fe/v103fe12/outputs/jepa_overfit_time_enc_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
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

export OMP_NUM_THREADS=8

torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch --cfg_file /home/atuin/v103fe/v103fe12/MTR/tools/cfgs/waymo/jepa_wt_batch_time_enc.yaml --batch_size=20 --epochs=400 --extra_tag=Overfit_Transf_1_1_0 --tcp_port=$PORT --workers=8 --max_ckpt_save_num=10 --not_eval_with_train --scenario_id 103718491a34269 3c4c59a7d820467 cd7fdd75bfeddbe7 f13e51dfe26339b3 d2b602df5a0b1a54 8189fa99f7c98b1b 6ad806abcf93c8f8 66e1a1918dca9607 c384f3dfc60ec63a 9f42997ad8749585 307349a330fe74f6 d030f6dd4505e09c 13b07777432ecf75 824d951e9769df42 feacabfe2cb0aec6 eb86882e0873801b 123bb8fd87b7d9e2 8539e5a2686c2f91 b0454cdf7b34bdab 16c9ce803f090eb6 --set DATA_CONFIG.DATA_ROOT $TMPDIR MODEL.CONTEXT_ENCODER.mse_coeff 1.0 MODEL.CONTEXT_ENCODER.std_coeff 1.0 MODEL.CONTEXT_ENCODER.cov_coeff 0.0 MODEL.CONTEXT_ENCODER.TIME_ENCODER.TYPE transformer

# Deactivate the virtual environment at the end
deactivate
