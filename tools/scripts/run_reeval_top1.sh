#!/bin/bash
# R4: re-evaluate saved (last-epoch) checkpoints to record top-1 ADE and mode ranking.
# minADE@6 rewards mode coverage; pose may sharpen the most-likely mode instead.
# NOTE: last-epoch checkpoints (best-epoch ckpts were not retained), so these numbers
# are a DIAGNOSTIC and are not directly comparable to the best-epoch tables.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/reeval_top1.log
declare -A C=(
  [map_nopose]=mtr+pose_data_no_pose_with_map
  [map_xattn_pe]=mtr+pose_data_cross_attn_pe_with_map
  [map_shufflepose]=mtr+pose_data_cross_attn_pe_with_map_shufflepose
  [map_constpose]=mtr+pose_data_cross_attn_pe_with_map_constpose
)
echo "=== reeval started $(date) ===" >> "$LOG"
for cell in map_nopose map_xattn_pe map_shufflepose map_constpose; do
  cfg=${C[$cell]}
  for seed in 101 202 303 404 505; do
    tag="MS_${cell}_s${seed}"
    ck=$(ls -t ../output/waymo/${cfg}/${tag}/ckpt/checkpoint_epoch_*.pth 2>/dev/null | head -1)
    [ -z "$ck" ] && { echo "no ckpt ${tag}" >> "$LOG"; continue; }
    WANDB_MODE=offline "$PY" test.py --cfg_file "cfgs/waymo/${cfg}.yaml" --ckpt "$ck" \
      --extra_tag "${tag}" --eval_tag top1b --workers 4 --batch_size 10 > "/tmp/reeval_${tag}.log" 2>&1 \
      && echo "OK ${tag}" >> "$LOG" || echo "FAIL ${tag}" >> "$LOG"
  done
done
echo "=== reeval finished $(date) ===" >> "$LOG"
