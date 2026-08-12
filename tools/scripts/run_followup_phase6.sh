#!/bin/bash
# Phase 6 (2026-08-12) — professor follow-ups #1 and #4, on the map x xattn+PE cell
# (the only configuration with a seed-significant pose benefit).
#   map_shufflepose : each pedestrian gets ANOTHER pedestrian's pose sequence, placed on
#                     the recipient's own observation grid => validity masks bit-identical,
#                     only pose CONTENT is wrong. If -1.6% survives, it is not information.
#   map_joints3d    : encoder input = 72-d root-relative 3D SMPL joints (skeleton keypoints)
#                     instead of 144-d 6D rotations. Targets unchanged => isolates representation.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
MASTER_LOG=/tmp/multiseed_phase6.log
run_one () {
  local cfg="$1"; local tag="$2"; shift 2
  local outdir="../output/waymo/${cfg}/${tag}"
  [ -f "${outdir}/DONE" ] && { echo "skip ${tag}" >> "$MASTER_LOG"; return; }
  echo "--- $(date '+%H:%M:%S') start ${tag}" >> "$MASTER_LOG"
  WANDB_MODE=offline "$PY" train.py --cfg_file "cfgs/waymo/${cfg}.yaml" \
    --extra_tag "${tag}" --workers 8 --max_ckpt_save_num 1 "$@" > "/tmp/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then
    touch "${outdir}/DONE"
    rm -f "${outdir}/ckpt/best_model.pth" "${outdir}/eval/eval_with_train/result.pkl"
    local best=$(grep -E "^minADE" "${outdir}"/log_train_*.txt | awk '{print $2}' | sort -g | head -1)
    echo "OK  ${tag} best_minADE=${best} $(date '+%H:%M:%S')" >> "$MASTER_LOG"
  else
    echo "FAIL ${tag} (see /tmp/${tag}.log)" >> "$MASTER_LOG"
  fi
}
echo "=== phase6 started $(date) ===" >> "$MASTER_LOG"
for seed in 101 202 303 404 505; do
  run_one "mtr+pose_data_cross_attn_pe_with_map_shufflepose" "MS_map_shufflepose_s${seed}" --random_seed "${seed}"
  run_one "mtr+pose_data_cross_attn_pe_with_map_joints3d"    "MS_map_joints3d_s${seed}"    --random_seed "${seed}"
done
echo "=== phase6 finished $(date) ===" >> "$MASTER_LOG"
