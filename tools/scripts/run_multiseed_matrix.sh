#!/bin/bash
# Multi-seed replication matrix (2026-06-12).
# 12 cells x 3 seeds, identical protocol to the original H-series runs
# (30 epochs, batch 10, eval-with-train every 2 epochs + last 10 epochs).
# Per-agent metrics are saved per eval epoch (metrics_epoch_N.pkl) for bootstrap CIs.

set -u
cd "$(dirname "$0")/.."   # tools/

PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
H7_CKPT=../output/waymo/mtr+full_ped_pretrain/H7_full_pretrain/ckpt/checkpoint_epoch_30.pth
SEEDS="101 202 303"
MASTER_LOG=/tmp/multiseed_matrix.log

# cell_name:cfg_file:pretrained(0/1)
CELLS=(
  "baseline:mtr+pose_data_no_pose:0"
  "wta01:mtr+pose_data_geo_only:0"
  "geo_pure:mtr+pose_data_geo_pure:0"
  "map_nopose:mtr+pose_data_no_pose_with_map:0"
  "map_wta01:mtr+pose_data_geo_only_with_map:0"
  "ft_nopose:mtr+full_ped_finetune_no_pose:1"
  "ft_wta01:mtr+full_ped_finetune_geo:1"
  "xattn:mtr+pose_data_cross_attn:0"
  "xattn_pe:mtr+pose_data_cross_attn_pe:0"
  "gmm_only:mtr+pose_data_gmm_only:0"
  "mpjpe:mtr+pose_data_mpjpe_only:0"
  "full:mtr+pose_data:0"
)

echo "=== matrix started $(date) ===" >> "$MASTER_LOG"
for cell in "${CELLS[@]}"; do
  name="${cell%%:*}"
  rest="${cell#*:}"
  cfg="${rest%%:*}"
  pretrained="${rest##*:}"
  for seed in $SEEDS; do
    tag="MS_${name}_s${seed}"
    outdir="../output/waymo/${cfg}/${tag}"
    if [ -f "${outdir}/DONE" ]; then
      echo "skip ${tag} (already done)" >> "$MASTER_LOG"
      continue
    fi
    echo "--- $(date '+%H:%M:%S') start ${tag}" >> "$MASTER_LOG"
    extra_args=""
    if [ "$pretrained" = "1" ]; then
      extra_args="--pretrained_model ${H7_CKPT}"
    fi
    WANDB_MODE=offline "$PY" train.py \
      --cfg_file "cfgs/waymo/${cfg}.yaml" \
      --extra_tag "${tag}" \
      --random_seed "${seed}" \
      --workers 8 \
      --max_ckpt_save_num 1 \
      ${extra_args} \
      > "/tmp/${tag}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "${outdir}/DONE"
      # save disk: drop the (broken, epoch-1) best_model and bulky result.pkl
      rm -f "${outdir}/ckpt/best_model.pth" "${outdir}/eval/eval_with_train/result.pkl"
      best=$(grep -E "^minADE" "${outdir}"/log_train_*.txt | awk '{print $2}' | sort -g | head -1)
      echo "OK  ${tag} rc=0 best_minADE=${best} $(date '+%H:%M:%S')" >> "$MASTER_LOG"
    else
      echo "FAIL ${tag} rc=${rc} (see /tmp/${tag}.log)" >> "$MASTER_LOG"
    fi
  done
done
echo "=== matrix finished $(date) ===" >> "$MASTER_LOG"
