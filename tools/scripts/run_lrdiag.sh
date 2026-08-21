#!/bin/bash
# Diagnostic: is the 5k pose-branch degradation an LR/schedule artifact?
# Both arms at LR 5e-5, 8 epochs. If pose tracks nopose here, the 1e-4 result is a tuning
# artifact, not evidence that pose conditioning hurts at scale.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/lrdiag.log
echo "=== lrdiag started $(date) ===" >> "$LOG"
for cell in no_pose cross_attn_pe; do
  cfg="mtr+pose_data_${cell}_with_map_scale5k_lrdiag"
  tag="LRDIAG_${cell}"
  out="../output/waymo/${cfg}/${tag}"
  [ -f "${out}/DONE" ] && continue
  while true; do
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 20
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break; sleep 240
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 "$PY" train.py \
    --cfg_file "cfgs/waymo/${cfg}.yaml" --extra_tag "${tag}" \
    --random_seed 101 --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
    && touch "${out}/DONE" && echo "OK ${tag}" >> "$LOG" || echo "FAIL ${tag}" >> "$LOG"
done
echo "=== lrdiag finished $(date) ===" >> "$LOG"
