#!/bin/bash
# Phase 5 (2026-07-20, review items 2.1 + 2.2 minimal):
#  2.1 map_norootorient x3   — root-orientation ablation IN THE MAP CONDITION (the
#      mechanistic headline was extrapolated from no-map; this tests it where it's used).
#  2.2 norootorient s404/505 — the no-map root-orient cell to n=5 (sigma 0.0031 at n=3
#      is the exact tight-cluster signature that fooled us twice).
#  2.2 map_wta01 s404/505    — "GRU+map gives nothing" to n=5.
set -u
cd "$(dirname "$0")/.."   # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
MASTER_LOG=/tmp/multiseed_phase5.log

run_one () {  # cfg tag [extra args...]
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

echo "=== phase5 started $(date) ===" >> "$MASTER_LOG"
# 2.1 first (highest value)
for seed in 101 202 303; do
  run_one "mtr+pose_data_cross_attn_pe_with_map_norootorient" "MS_map_norootorient_s${seed}" --random_seed "${seed}"
done
# 2.2 minimal
for seed in 404 505; do
  run_one "mtr+pose_data_cross_attn_pe_norootorient" "MS_norootorient_s${seed}" --random_seed "${seed}"
  run_one "mtr+pose_data_geo_only_with_map"          "MS_map_wta01_s${seed}"    --random_seed "${seed}"
done
echo "=== phase5 finished $(date) ===" >> "$MASTER_LOG"
