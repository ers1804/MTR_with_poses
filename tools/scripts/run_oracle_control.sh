#!/bin/bash
# POSITIVE CONTROL for the pose pathway (reviewer-anticipated, 2026-10-19).
#
# The shuffled arm shows the branch cannot distinguish own-body from stranger-body.
# It does NOT show the branch can transmit anything: the fusion residual is
# zero-initialised, so "dead branch" and "live branch carrying no information" are
# observationally identical in every number the paper reports.
#
# This arm replaces the SMPL features with a 6-channel encoding of the agent's OWN
# future displacement (+3 s and +8 s, centre-object frame), holding the branch, its
# parameter count, its input dimensionality and its validity masks fixed. If minADE
# drops sharply against no-pose, the pathway is live and the null is a statement
# about the information in body pose rather than about the plumbing that carries it.
#
# Same 579-scene scratch rung as Table I/II, LR 1e-4, 30 epochs, n=3 seeds.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/oracle_control.log
CFG=mtr+pose_data_cross_attn_pe_with_map_oracle_fixdata
echo "=== oracle positive control started $(date) ===" >> "$LOG"
for seed in 101 202 303; do
  tag="MS_map_oracle_fix_s${seed}"; out="../output/waymo/${CFG}/${tag}"
  [ -f "${out}/DONE" ] && { echo "SKIP ${tag} (done)" >> "$LOG"; continue; }
  # wait for a free GPU (two consecutive checks; foreign jobs respected)
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    sleep 30
    used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 4000 ] && [ "$used2" -lt 4000 ] && break
    sleep 270
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 \
    "$PY" train.py --cfg_file "cfgs/waymo/${CFG}.yaml" --extra_tag "${tag}" \
    --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
    && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
    && echo "OK ${tag} $(date)" >> "$LOG" \
    || echo "FAIL ${tag} $(date)" >> "$LOG"
done
echo "=== oracle positive control finished $(date) ===" >> "$LOG"
