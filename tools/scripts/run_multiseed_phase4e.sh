#!/bin/bash
# Phase 4e (2026-07-13): map_nopose & ft_nopose seeds 404/505, so the map/pretrain
# pose PAIRS have 5 common seeds and their hierarchical bootstrap is airtight at n=5
# (the pose cells map_xattn_pe/ft_xattn_pe are already n=5 from Phase 4d).
set -u
cd "$(dirname "$0")/.."   # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
H7_CKPT=../output/waymo/mtr+full_ped_pretrain/H7_full_pretrain/ckpt/checkpoint_epoch_30.pth
MASTER_LOG=/tmp/multiseed_phase4e.log

run_one () {  # cfg tag [extra_args...]
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

echo "=== phase4e started $(date) ===" >> "$MASTER_LOG"
for seed in 404 505; do
  run_one "mtr+pose_data_no_pose_with_map" "MS_map_nopose_s${seed}" --random_seed "${seed}"
  run_one "mtr+full_ped_finetune_no_pose"  "MS_ft_nopose_s${seed}"  --random_seed "${seed}" --pretrained_model "${H7_CKPT}"
done
echo "=== phase4e finished $(date) ===" >> "$MASTER_LOG"
