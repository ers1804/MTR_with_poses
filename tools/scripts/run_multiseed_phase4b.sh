#!/bin/bash
# Phase 4b (gap-closing, 2026-07-13):
#  (1) Re-run the four auxiliary-loss audit cells with the P1.1/P1.2 mask fix active,
#      under NEW tags (MS_<cell>_masked_s<seed>) so the historical UNMASKED runs are
#      preserved for comparison. Tests whether masking the (inert-on-10fps) pose
#      losses removes the WTA-L1 heavy-tailed failure mode.
#  (2) Add seeds 404/505 for xattn_pe so its -3.5% headline no longer rests on n=3.
# Same protocol as run_multiseed_matrix.sh (30 epochs, batch 10, per-agent metrics).
set -u
cd "$(dirname "$0")/.."   # tools/

PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
MASTER_LOG=/tmp/multiseed_phase4b.log
SEEDS="101 202 303"

run_one () {  # cfg tag [extra_args...]
  local cfg="$1"; local tag="$2"; shift 2
  local outdir="../output/waymo/${cfg}/${tag}"
  if [ -f "${outdir}/DONE" ]; then echo "skip ${tag} (done)" >> "$MASTER_LOG"; return; fi
  echo "--- $(date '+%H:%M:%S') start ${tag}" >> "$MASTER_LOG"
  WANDB_MODE=offline "$PY" train.py --cfg_file "cfgs/waymo/${cfg}.yaml" \
    --extra_tag "${tag}" --workers 8 --max_ckpt_save_num 1 "$@" \
    > "/tmp/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "${outdir}/DONE"
    rm -f "${outdir}/ckpt/best_model.pth" "${outdir}/eval/eval_with_train/result.pkl"
    local best=$(grep -E "^minADE" "${outdir}"/log_train_*.txt | awk '{print $2}' | sort -g | head -1)
    echo "OK  ${tag} best_minADE=${best} $(date '+%H:%M:%S')" >> "$MASTER_LOG"
  else
    echo "FAIL ${tag} rc=${rc} (see /tmp/${tag}.log)" >> "$MASTER_LOG"
  fi
}

echo "=== phase4b started $(date) ===" >> "$MASTER_LOG"

# (1) masked audit cells + a masked no-pose baseline (anchor: isolates the pose-loss
#     effect from the rest of the fix bundle, since P1.7 etc. also touch the baseline).
declare -A AUDIT=(
  [baseline_masked]=mtr+pose_data_no_pose
  [wta01_masked]=mtr+pose_data_geo_only
  [mpjpe_masked]=mtr+pose_data_mpjpe_only
  [full_masked]=mtr+pose_data
  [geo_pure_masked]=mtr+pose_data_geo_pure
)
for cell in baseline_masked wta01_masked mpjpe_masked full_masked geo_pure_masked; do
  for seed in $SEEDS; do
    run_one "${AUDIT[$cell]}" "MS_${cell}_s${seed}" --random_seed "${seed}"
  done
done

# (2) xattn_pe extra seeds 404/505 (same config/tag scheme as the existing 101/202/303)
for seed in 404 505; do
  run_one "mtr+pose_data_cross_attn_pe" "MS_xattn_pe_s${seed}" --random_seed "${seed}"
done

echo "=== phase4b finished $(date) ===" >> "$MASTER_LOG"
