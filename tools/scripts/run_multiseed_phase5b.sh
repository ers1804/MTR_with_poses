#!/bin/bash
# Phase 5b (2026-07-20, review 2.2 extended): current-code cells to n=5.
# Only cells whose existing seeds were ALREADY trained with the post-fix (masked)
# code: pose30fps + the five masked audit cells. (Pre-fix cells -> phase5c worktree.)
set -u
cd "$(dirname "$0")/.."   # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
MASTER_LOG=/tmp/multiseed_phase5b.log

run_one () {
  local cfg="$1"; local tag="$2"; shift 2
  local outdir="../output/waymo/${cfg}/${tag}"
  [ -f "${outdir}/DONE" ] && { echo "skip ${tag} (done)" >> "$MASTER_LOG"; return; }
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

echo "=== phase5b started $(date) ===" >> "$MASTER_LOG"
for seed in 404 505; do
  run_one "mtr+pose_data_30fps"    "MS_pose30fps_s${seed}"       --random_seed "${seed}"
  run_one "mtr+pose_data_no_pose"  "MS_baseline_masked_s${seed}" --random_seed "${seed}"
  run_one "mtr+pose_data_geo_only" "MS_wta01_masked_s${seed}"    --random_seed "${seed}"
  run_one "mtr+pose_data_mpjpe_only" "MS_mpjpe_masked_s${seed}"  --random_seed "${seed}"
  run_one "mtr+pose_data"          "MS_full_masked_s${seed}"     --random_seed "${seed}"
  run_one "mtr+pose_data_geo_pure" "MS_geo_pure_masked_s${seed}" --random_seed "${seed}"
done
echo "=== phase5b finished $(date) ===" >> "$MASTER_LOG"
