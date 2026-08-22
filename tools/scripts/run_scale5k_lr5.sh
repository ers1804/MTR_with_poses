#!/bin/bash
# Rung 1 at LR 5e-5, full 30 epochs: fair per-arm comparison after the 1e-4 pose collapse
# was shown to be largely an optimization artifact (LR diagnostic 2026-08-21).
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/multiseed_scale.log
declare -A C=(
  [scale5k_lr5_nopose]=mtr+pose_data_no_pose_with_map_scale5k_lr5
  [scale5k_lr5_xattn_pe]=mtr+pose_data_cross_attn_pe_with_map_scale5k_lr5
  [scale5k_lr5_shufflepose]=mtr+pose_data_cross_attn_pe_with_map_shufflepose_scale5k_lr5
)
echo "=== scale5k LR5e-5 queue started $(date) ===" >> "$LOG"
for seed in 202 303; do
  for cell in scale5k_lr5_nopose scale5k_lr5_xattn_pe scale5k_lr5_shufflepose; do
    tag="MS_${cell}_s${seed}"; out="../output/waymo/${C[$cell]}/${tag}"
    [ -f "${out}/DONE" ] && continue
    while true; do
      u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
      u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break; sleep 270
    done
    WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 "$PY" train.py \
      --cfg_file "cfgs/waymo/${C[$cell]}.yaml" --extra_tag "${tag}" \
      --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
      && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
      && echo "OK ${tag}" >> "$LOG" || echo "FAIL ${tag}" >> "$LOG"
  done
done
echo "=== scale5k LR5e-5 queue finished $(date) ===" >> "$LOG"
