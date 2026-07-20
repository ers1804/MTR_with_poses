#!/bin/bash
# Phase 5d (2026-07-20, review 2.4): map_xattn_pe with agent-centric pose rotation,
# 3 seeds. Comparator map_xattn_pe is all post-fix code, so current code is correct
# here. Waits for the phase5->5b->5c review queue to finish (marker), cap ~9h.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
MASTER_LOG=/tmp/multiseed_phase5d.log
t=0
until grep -q "review queue COMPLETE" /tmp/review_queue.log 2>/dev/null; do
  sleep 60; t=$((t+1)); [ $t -ge 540 ] && { echo "gave up waiting" >> "$MASTER_LOG"; exit 1; }
done
echo "=== phase5d started $(date) ===" >> "$MASTER_LOG"
for seed in 101 202 303; do
  tag="MS_map_agentrot_s${seed}"; outdir="../output/waymo/mtr+pose_data_cross_attn_pe_with_map_agentrot/${tag}"
  [ -f "${outdir}/DONE" ] && { echo "skip ${tag}" >> "$MASTER_LOG"; continue; }
  echo "--- $(date '+%H:%M:%S') start ${tag}" >> "$MASTER_LOG"
  WANDB_MODE=offline "$PY" train.py --cfg_file "cfgs/waymo/mtr+pose_data_cross_attn_pe_with_map_agentrot.yaml" \
    --extra_tag "${tag}" --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then
    touch "${outdir}/DONE"
    rm -f "${outdir}/ckpt/best_model.pth" "${outdir}/eval/eval_with_train/result.pkl"
    best=$(grep -E "^minADE" "${outdir}"/log_train_*.txt | awk '{print $2}' | sort -g | head -1)
    echo "OK  ${tag} best_minADE=${best} $(date '+%H:%M:%S')" >> "$MASTER_LOG"
  else
    echo "FAIL ${tag} (see /tmp/${tag}.log)" >> "$MASTER_LOG"
  fi
done
echo "=== phase5d finished $(date) ===" >> "$MASTER_LOG"
