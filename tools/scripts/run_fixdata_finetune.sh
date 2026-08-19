#!/bin/bash
# Phase 3 of clean-data re-runs: the triad in the FINETUNE regime (strongest model).
# The H7 pretrain checkpoint is trajectory-only -> unaffected by the export bug.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/multiseed_fixdata.log
H7_CKPT=../output/waymo/mtr+full_ped_pretrain/H7_full_pretrain/ckpt/checkpoint_epoch_30.pth
until grep -q "fixdata extend queue finished" "$LOG" 2>/dev/null; do sleep 300; done
echo "=== fixdata finetune queue started $(date) ===" >> "$LOG"
declare -A C=(
  [ft_nopose_fix]=mtr+full_ped_finetune_no_pose_fixdata
  [ft_xattn_pe_fix]=mtr+full_ped_finetune_xattn_pe_fixdata
  [ft_shufflepose_fix]=mtr+full_ped_finetune_shufflepose_fixdata
)
for seed in 101 202 303; do
  for cell in ft_nopose_fix ft_xattn_pe_fix ft_shufflepose_fix; do
    tag="MS_${cell}_s${seed}"; out="../output/waymo/${C[$cell]}/${tag}"
    [ -f "${out}/DONE" ] && continue
    while true; do
      u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
      u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break; sleep 270
    done
    WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 "$PY" train.py \
      --cfg_file "cfgs/waymo/${C[$cell]}.yaml" --extra_tag "${tag}" --pretrained_model "${H7_CKPT}" \
      --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
      && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
      && echo "OK ${tag}" >> "$LOG" || echo "FAIL ${tag}" >> "$LOG"
  done
done
echo "=== fixdata finetune queue finished $(date) ===" >> "$LOG"
