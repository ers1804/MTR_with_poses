#!/bin/bash
# Phase 7 (2026-08-12) — professor follow-up #2: does a 2 s horizon help?
# Retrains both arms of the map cell with NUM_FUTURE_FRAMES=20 and intention points
# RE-CLUSTERED from 2 s endpoints (the shipped 8 s goals span 1.7-49.2 m vs 0.02-7.48 m
# for 2 s, so reusing them would mis-specify the goal head and confound the comparison).
# Waits for phase6 to finish.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
MASTER_LOG=/tmp/multiseed_phase7.log
t=0
until grep -q "phase6 finished" /tmp/multiseed_phase6.log 2>/dev/null; do
  sleep 60; t=$((t+1)); [ $t -ge 420 ] && { echo "gave up waiting for phase6" >> "$MASTER_LOG"; exit 1; }
done
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
echo "=== phase7 started $(date) ===" >> "$MASTER_LOG"
for seed in 101 202 303 404 505; do
  run_one "mtr+pose_data_no_pose_with_map_2s"        "MS_map_nopose_2s_s${seed}"   --random_seed "${seed}"
  run_one "mtr+pose_data_cross_attn_pe_with_map_2s"  "MS_map_xattn_pe_2s_s${seed}" --random_seed "${seed}"
done
echo "=== phase7 finished $(date) ===" >> "$MASTER_LOG"
