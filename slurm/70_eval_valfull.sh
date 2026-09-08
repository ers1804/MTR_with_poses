#!/bin/bash -l
#SBATCH --job-name=mtrpose_evalfull
#SBATCH --output=/hnvme/workspace/v103fe12-mtr_smpl/logs/evalfull_%A_%a.txt
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE
# Re-evaluate EXISTING 25k checkpoints on the FULL clean validation split (27,156 scenes,
# ~5.25x the 5,171 used during training) to raise agent counts in the rare standing-then-moves
# subpopulation. Inference only — no training.
#
# SCOPE: only epoch-30 checkpoints survived, so this is a LAST-epoch operating point. Verified
# 2026-09-08 that the pose-vs-shuffle (content) test is stable across epoch selection, while the
# no-pose-vs-pose (cross-architecture) test is NOT — so use these outputs for the content test only.
unset SLURM_EXPORT_ENV
set -eu
WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}

if ! type -t module >/dev/null 2>&1; then
  set +u
  for f in /etc/profile.d/zz-rrze-local.sh /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    [ -r "$f" ] && . "$f" && break
  done
  set -u
fi
module add gpu-env/2025 python/3.12-conda cuda/12.6.2

CELLS=(no_pose_with_map cross_attn_pe_with_map cross_attn_pe_with_map_shufflepose)
SEEDS=(101 202 303)
i=$SLURM_ARRAY_TASK_ID
cell=${CELLS[$(( i / 3 ))]}
seed=${SEEDS[$(( i % 3 ))]}
tag="HELMA_25k_${cell}_lr25_s${seed}"
run="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${cell}_scale5k_lr5/$tag"
ckpt="$run/ckpt/checkpoint_epoch_30.pth"
[ -f "$ckpt" ] || { echo "missing $ckpt"; exit 1; }
echo "task $i -> $tag"

mkdir -p "$TMPDIR/env"
tar xzf "$WS/env/mtr_smpl.tar.gz" -C "$TMPDIR/env"
set +u; source "$TMPDIR/env/bin/activate"; conda-unpack; set -u
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "staging full validation split ..."
tar xf "$WS/data/valfull.tar" -C "$TMPDIR"      # -> poses/validation, scenarios/processed_scenarios_validation
mkdir -p "$TMPDIR/poses/training"               # split dir must exist; unused in eval
echo "val scenes staged: $(ls "$TMPDIR/poses/validation" | wc -l)"

cd "$WS/MTR_with_poses/tools"
python test.py \
  --cfg_file "cfgs/waymo/mtr+pose_data_${cell}_scale5k_lr5.yaml" \
  --ckpt "$ckpt" --extra_tag "VALFULL_${tag}" --batch_size 8 --workers 8 \
  --set DATA_CONFIG.DATA_ROOT "$TMPDIR/poses" \
        DATA_CONFIG.PROCESSED_SCENARIOS_PATH "$TMPDIR/scenarios" \
        MODEL.MOTION_DECODER.SMPL_MODEL_DIR "$WS/MTR_with_poses/models"
echo "DONE $tag"
