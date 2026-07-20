#!/bin/bash
# Phase 5c (2026-07-20): extend PRE-FIX aux-active cells to n=5 with PRE-FIX code.
# Runs inside the git worktree pinned to the Phase-0 commit (70c7a5d, before the
# P1.1 mask fix), so seeds 404/505 come from the SAME training distribution as the
# cells' original 101/202/303 (2026-06-12 matrix). Finished runs are moved into the
# main repo's output tree so analyze_multiseed.py sees complete cells.
# NOTE: this also replaces the previously mis-extended xattn_pe 404/505 (those
# masked-code runs were renamed MS_xattn_pe_maskedcode_s{404,505}).
set -u
WT=/home/erik/ssd2/gitprojects/MTR_prefix_worktree
MAIN=/home/erik/ssd2/gitprojects/MTR_with_poses
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
H7_CKPT=${MAIN}/output/waymo/mtr+full_ped_pretrain/H7_full_pretrain/ckpt/checkpoint_epoch_30.pth
MASTER_LOG=/tmp/multiseed_phase5c.log
cd "$WT/tools"

run_one () {
  local cfg="$1"; local tag="$2"; shift 2
  local outdir="$WT/output/waymo/${cfg}/${tag}"
  local final="$MAIN/output/waymo/${cfg}/${tag}"
  [ -f "${final}/DONE" ] && { echo "skip ${tag} (done in main)" >> "$MASTER_LOG"; return; }
  echo "--- $(date '+%H:%M:%S') start ${tag} [pre-fix worktree]" >> "$MASTER_LOG"
  WANDB_MODE=offline "$PY" train.py --cfg_file "cfgs/waymo/${cfg}.yaml" \
    --extra_tag "${tag}" --workers 8 --max_ckpt_save_num 1 "$@" > "/tmp/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then
    rm -f "${outdir}/ckpt/best_model.pth" "${outdir}/eval/eval_with_train/result.pkl"
    mkdir -p "$(dirname "$final")"
    mv "$outdir" "$final" && touch "${final}/DONE"
    local best=$(grep -E "^minADE" "${final}"/log_train_*.txt | awk '{print $2}' | sort -g | head -1)
    echo "OK  ${tag} best_minADE=${best} (moved to main) $(date '+%H:%M:%S')" >> "$MASTER_LOG"
  else
    echo "FAIL ${tag} (see /tmp/${tag}.log)" >> "$MASTER_LOG"
  fi
}

echo "=== phase5c started $(date) [worktree @ $(cd $WT && git rev-parse --short HEAD)] ===" >> "$MASTER_LOG"
for seed in 404 505; do
  run_one "mtr+pose_data_cross_attn_pe" "MS_xattn_pe_s${seed}"  --random_seed "${seed}"
  run_one "mtr+pose_data_cross_attn"    "MS_xattn_s${seed}"     --random_seed "${seed}"
  run_one "mtr+pose_data_gmm_only"      "MS_gmm_only_s${seed}"  --random_seed "${seed}"
  run_one "mtr+pose_data_mpjpe_only"    "MS_mpjpe_s${seed}"     --random_seed "${seed}"
  run_one "mtr+pose_data"               "MS_full_s${seed}"      --random_seed "${seed}"
  run_one "mtr+pose_data_geo_only_with_map" "MS_map_wta01_s${seed}" --random_seed "${seed}"
  run_one "mtr+full_ped_finetune_geo"   "MS_ft_wta01_s${seed}"  --random_seed "${seed}" --pretrained_model "${H7_CKPT}"
done
echo "=== phase5c finished $(date) ===" >> "$MASTER_LOG"
