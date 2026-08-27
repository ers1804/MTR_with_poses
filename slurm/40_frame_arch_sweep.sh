#!/bin/bash -l
#SBATCH --job-name=mtr_framearch
#SBATCH --output=/hnvme/workspace/v103fe12-mtr_smpl/logs/fa_%A_%a.txt
#SBATCH --time=24:00:00
# 25k-rung runs need ~16.5 h (12,500 iters/epoch x 30 epochs at ~0.16 s/iter). A 12 h cap
# killed all 18 tasks at epoch ~21/30 on 2026-08-26; 24 h is the h100 partition maximum.
#SBATCH --nodes=1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE
# Follow-up to the 2026-08-26 review, at the 25k rung.
#   wave1 = coordinate-frame fix (AGENT_CENTRIC_POSE_ROT) + its wrong-pose control.
#   wave2 = frame-correct DERIVED pose features fused early into the agent encoder + control.
# Task list is explicit (cfg|lr|seed) rather than index arithmetic, so the grid is auditable.
unset SLURM_EXPORT_ENV
set -eu

WS=${WS:-/hnvme/workspace/v103fe12-mtr_smpl}
RUNG=${RUNG:-25k}
WAVE=${1:-wave1}

if ! type -t module >/dev/null 2>&1; then
  set +u
  for f in /etc/profile.d/zz-rrze-local.sh /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    [ -r "$f" ] && . "$f" && break
  done
  set -u
fi
module add gpu-env/2025
module add python/3.12-conda
module add cuda/12.6.2

echo "unpacking env to $TMPDIR/env ..."
mkdir -p "$TMPDIR/env"
tar xzf "$WS/env/mtr_smpl.tar.gz" -C "$TMPDIR/env"
set +u
source "$TMPDIR/env/bin/activate"
conda-unpack
set -u
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRAIN_MAX_CENTERS=24

echo "staging data to $TMPDIR ..."
tar xf "$WS/data/poses_${RUNG}.tar"     -C "$TMPDIR"
tar xf "$WS/data/scenarios_${RUNG}.tar" -C "$TMPDIR"

case "$WAVE" in
  wave1) TASKS=(
    "cross_attn_pe_with_map_agentrot25k|0.000025|101"
    "cross_attn_pe_with_map_agentrot25k|0.000025|202"
    "cross_attn_pe_with_map_agentrot25k|0.000025|303"
    "cross_attn_pe_with_map_agentrot25k|0.00005|101"
    "cross_attn_pe_with_map_agentrot25k|0.00005|202"
    "cross_attn_pe_with_map_agentrot25k|0.00005|303"
    "cross_attn_pe_with_map_shufflepose_agentrot25k|0.000025|101"
    "cross_attn_pe_with_map_shufflepose_agentrot25k|0.000025|202"
    "cross_attn_pe_with_map_shufflepose_agentrot25k|0.000025|303"
  );;
  wave2) TASKS=(
    "cross_attn_pe_with_map_derived25k|0.000025|101"
    "cross_attn_pe_with_map_derived25k|0.000025|202"
    "cross_attn_pe_with_map_derived25k|0.000025|303"
    "cross_attn_pe_with_map_derived25k|0.00005|101"
    "cross_attn_pe_with_map_derived25k|0.00005|202"
    "cross_attn_pe_with_map_derived25k|0.00005|303"
    "cross_attn_pe_with_map_shufflepose_derived25k|0.000025|101"
    "cross_attn_pe_with_map_shufflepose_derived25k|0.000025|202"
    "cross_attn_pe_with_map_shufflepose_derived25k|0.000025|303"
  );;
  *) echo "unknown wave: $WAVE" >&2; exit 2;;
esac

spec=${TASKS[$SLURM_ARRAY_TASK_ID]}
cfg=${spec%%|*}; rest=${spec#*|}; lr=${rest%%|*}; seed=${rest##*|}
lrtag=$(echo "$lr" | sed 's/0\.0*//')
tag="FA_${RUNG}_${cfg}_lr${lrtag}_s${seed}"
echo "task $SLURM_ARRAY_TASK_ID -> cfg=$cfg lr=$lr seed=$seed tag=$tag"

STATUS="$WS/logs/status_fa.txt"
note() { flock "$STATUS.lock" -c "echo \"$(date +%H:%M:%S) $tag $*\" >> '$STATUS'"; }
note "START"
trap 'note "EXIT rc=$?"' EXIT

cd "$WS/MTR_with_poses/tools"
python train.py \
  --cfg_file "cfgs/waymo/mtr+pose_data_${cfg}.yaml" \
  --extra_tag "$tag" --random_seed "$seed" --workers 8 --max_ckpt_save_num 1 \
  --set DATA_CONFIG.DATA_ROOT "$TMPDIR/poses" \
        DATA_CONFIG.PROCESSED_SCENARIOS_PATH "$TMPDIR/scenarios" \
        MODEL.MOTION_DECODER.SMPL_MODEL_DIR "$WS/MTR_with_poses/models" \
        OPTIMIZATION.LR "$lr"

out="$WS/MTR_with_poses/output/waymo/mtr+pose_data_${cfg}/$tag"
rm -f "$out/ckpt/best_model.pth" "$out/eval/eval_with_train/result.pkl" 2>/dev/null || true
touch "$out/DONE"
note "DONE best=$(grep -hE '^minADE' "$out"/log_train_*.txt 2>/dev/null | awk '{print $2}' | sort -g | head -1)"
echo "DONE $tag"
