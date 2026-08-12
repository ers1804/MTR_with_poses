#!/bin/bash
# Phase 8: capacity control — constant (zero-information) pose for every pedestrian.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/multiseed_phase8.log
t=0; until grep -q "phase7 finished" /tmp/multiseed_phase7.log 2>/dev/null; do
  sleep 120; t=$((t+1)); [ $t -ge 300 ] && { echo "gave up waiting" >> "$LOG"; exit 1; }; done
echo "=== phase8 started $(date) ===" >> "$LOG"
for seed in 101 202 303 404 505; do
  tag="MS_map_constpose_s${seed}"; out="../output/waymo/mtr+pose_data_cross_attn_pe_with_map_constpose/${tag}"
  [ -f "${out}/DONE" ] && continue
  echo "--- $(date '+%H:%M:%S') start ${tag}" >> "$LOG"
  WANDB_MODE=offline "$PY" train.py --cfg_file cfgs/waymo/mtr+pose_data_cross_attn_pe_with_map_constpose.yaml \
    --extra_tag "${tag}" --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
    && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
    && echo "OK ${tag} best=$(grep -E '^minADE' ${out}/log_train_*.txt|awk '{print $2}'|sort -g|head -1)" >> "$LOG" \
    || echo "FAIL ${tag}" >> "$LOG"
done
echo "=== phase8 finished $(date) ===" >> "$LOG"
