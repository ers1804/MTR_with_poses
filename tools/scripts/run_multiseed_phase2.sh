#!/bin/bash
# Phase 2: extra seeds (404, 505) for the headline cells, to characterize
# the heavy-tailed training variance observed in phase 1 (e.g. wta01 s202 = 0.7848).
set -u
cd "$(dirname "$0")/.."

PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
SEEDS="404 505"
MASTER_LOG=/tmp/multiseed_matrix.log

CELLS=(
  "baseline:mtr+pose_data_no_pose"
  "wta01:mtr+pose_data_geo_only"
  "geo_pure:mtr+pose_data_geo_pure"
)

echo "=== phase2 started $(date) ===" >> "$MASTER_LOG"
for cell in "${CELLS[@]}"; do
  name="${cell%%:*}"
  cfg="${cell#*:}"
  for seed in $SEEDS; do
    tag="MS_${name}_s${seed}"
    outdir="../output/waymo/${cfg}/${tag}"
    if [ -f "${outdir}/DONE" ]; then
      echo "skip ${tag} (already done)" >> "$MASTER_LOG"
      continue
    fi
    echo "--- $(date '+%H:%M:%S') start ${tag}" >> "$MASTER_LOG"
    WANDB_MODE=offline "$PY" train.py \
      --cfg_file "cfgs/waymo/${cfg}.yaml" \
      --extra_tag "${tag}" \
      --random_seed "${seed}" \
      --workers 8 \
      --max_ckpt_save_num 1 \
      > "/tmp/${tag}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "${outdir}/DONE"
      rm -f "${outdir}/ckpt/best_model.pth" "${outdir}/eval/eval_with_train/result.pkl"
      best=$(grep -E "^minADE" "${outdir}"/log_train_*.txt | awk '{print $2}' | sort -g | head -1)
      echo "OK  ${tag} rc=0 best_minADE=${best} $(date '+%H:%M:%S')" >> "$MASTER_LOG"
    else
      echo "FAIL ${tag} rc=${rc} (see /tmp/${tag}.log)" >> "$MASTER_LOG"
    fi
  done
done
echo "=== phase2 finished $(date) ===" >> "$MASTER_LOG"
