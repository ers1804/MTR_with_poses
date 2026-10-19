#!/bin/bash
# Two experiments agreed 2026-10-19, run back to back on one GPU (~22 h total).
#
# A. HEADING ABLATION (9 runs). The paper claims pose is redundant *because* the
#    tracker's box heading is already an agent input. That claim currently rests on
#    correlational evidence (onset-direction error, stratification) and has never
#    been tested inside the forecaster. Zeroing the heading sin/cos channels removes
#    the only facing cue a stationary pedestrian has, so the mechanism predicts the
#    content test finally goes NEGATIVE here. If it does not, the mechanism is wrong.
#
# B. DETECTION FLOOR (18 runs). Having shown the branch is live, degrade the oracle:
#    direction-only, corrupted by a fixed per-agent angular error. Finds how accurately
#    a side channel must predict future heading before this pathway converts it into a
#    minADE gain -- a threshold to compare against SMPL facing (44.2 deg) and the box
#    heading (25.2 deg).
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/mechanism_queue.log
echo "=== mechanism queue started $(date) ===" >> "$LOG"

run_one() {   # $1 = cell tag, $2 = config basename
  local tag="MS_$1"; local cfg="$2"; local out="../output/waymo/${cfg}/${tag}"
  [ -f "${out}/DONE" ] && { echo "SKIP ${tag}" >> "$LOG"; return; }
  while true; do
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break
    sleep 270
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 \
    "$PY" train.py --cfg_file "cfgs/waymo/${cfg}.yaml" --extra_tag "${tag}" \
    --random_seed "${3}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
    && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
    && echo "OK ${tag} $(date)" >> "$LOG" || echo "FAIL ${tag} $(date)" >> "$LOG"
}

# A. heading ablation
for seed in 101 202 303; do
  run_one "nohead_nopose_s${seed}" mtr+pose_data_no_pose_with_map_nohead_fixdata           "$seed"
  run_one "nohead_pose_s${seed}"   mtr+pose_data_cross_attn_pe_with_map_nohead_fixdata     "$seed"
  run_one "nohead_shuf_s${seed}"   mtr+pose_data_cross_attn_pe_with_map_shufflepose_nohead_fixdata "$seed"
done
echo "=== heading ablation done $(date) ===" >> "$LOG"

# B. detection floor
for seed in 101 202 303; do
  for m in 00 15 25 35 45 60; do
    run_one "oracledir${m}_s${seed}" "mtr+pose_data_cross_attn_pe_with_map_oracledir${m}_fixdata" "$seed"
  done
done
echo "=== mechanism queue finished $(date) ===" >> "$LOG"
