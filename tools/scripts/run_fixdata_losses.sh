#!/bin/bash
# Phase 2 of clean-data re-runs: does the aux pose-prediction loss help with CORRECT poses?
# (The corrupt-era null was measured against scrambled targets.) Waits for the core queue.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/multiseed_fixdata.log
until grep -q "fixdata queue finished" "$LOG" 2>/dev/null; do sleep 300; done
echo "=== fixdata losses queue started $(date) ===" >> "$LOG"
for seed in 101 202 303; do
  tag="MS_map_wta01_fix_s${seed}"; out="../output/waymo/mtr+pose_data_geo_only_with_map_fixdata/${tag}"
  [ -f "${out}/DONE" ] && continue
  while true; do
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break; sleep 270
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 "$PY" train.py \
    --cfg_file "cfgs/waymo/mtr+pose_data_geo_only_with_map_fixdata.yaml" --extra_tag "${tag}" \
    --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
    && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
    && echo "OK ${tag}" >> "$LOG" || echo "FAIL ${tag}" >> "$LOG"
done
echo "=== fixdata losses queue finished $(date) ===" >> "$LOG"
