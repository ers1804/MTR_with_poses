#!/bin/bash
# Scale rung 1: 5,000 scenes (31,302 train peds, 4.9x the 579-scene subset).
# Validation identical to all clean-data runs (5,171 scenes). 10 epochs, decay [3,7,8].
# Waits for a free GPU — a foreign navsim_eval job shares this card.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/multiseed_scale.log
declare -A C=(
  [scale5k_nopose]=mtr+pose_data_no_pose_with_map_scale5k
  [scale5k_xattn_pe]=mtr+pose_data_cross_attn_pe_with_map_scale5k
  [scale5k_shufflepose]=mtr+pose_data_cross_attn_pe_with_map_shufflepose_scale5k
)
echo "=== scale5k queue started $(date) ===" >> "$LOG"
for seed in 101 202 303; do
  for cell in scale5k_nopose scale5k_xattn_pe scale5k_shufflepose; do
    tag="MS_${cell}_s${seed}"; out="../output/waymo/${C[$cell]}/${tag}"
    [ -f "${out}/DONE" ] && continue
    while true; do
      u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
      u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break
      sleep 270
    done
    start=$(date +%s)
    WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 "$PY" train.py \
      --cfg_file "cfgs/waymo/${C[$cell]}.yaml" --extra_tag "${tag}" \
      --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
      && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
      && echo "OK ${tag} ($(( ($(date +%s)-start)/60 )) min)" >> "$LOG" \
      || echo "FAIL ${tag}" >> "$LOG"
  done
done
echo "=== scale5k queue finished $(date) ===" >> "$LOG"
