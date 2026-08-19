#!/bin/bash
# n=5 extension of the clean-data core triad (pre-registered: extend if signal at n=3).
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/multiseed_fixdata.log
until grep -q "fixdata losses queue finished" "$LOG" 2>/dev/null; do sleep 300; done
echo "=== fixdata extend queue started $(date) ===" >> "$LOG"
declare -A C=(
  [map_nopose_fix]=mtr+pose_data_no_pose_with_map_fixdata
  [map_xattn_pe_fix]=mtr+pose_data_cross_attn_pe_with_map_fixdata
  [map_shufflepose_fix]=mtr+pose_data_cross_attn_pe_with_map_shufflepose_fixdata
)
for seed in 404 505; do
  for cell in map_nopose_fix map_xattn_pe_fix map_shufflepose_fix; do
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
echo "=== fixdata extend queue finished $(date) ===" >> "$LOG"
