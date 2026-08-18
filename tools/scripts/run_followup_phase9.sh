#!/bin/bash
# Part 7 (pre-registered): temporal-denoising discriminator, 2 cells x 5 seeds.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/multiseed_phase9.log
declare -A C=(
  [map_denoise]=mtr+pose_data_cross_attn_pe_with_map_denoise
  [map_denoise_shuffle]=mtr+pose_data_cross_attn_pe_with_map_denoise_shuffle
)
echo "=== phase9 started $(date) ===" >> "$LOG"
for cell in map_denoise map_denoise_shuffle; do
  for seed in 101 202 303 404 505; do
    tag="MS_${cell}_s${seed}"; out="../output/waymo/${C[$cell]}/${tag}"
    [ -f "${out}/DONE" ] && continue
    WANDB_MODE=offline "$PY" train.py --cfg_file "cfgs/waymo/${C[$cell]}.yaml" --extra_tag "${tag}" \
      --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
      && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
      && echo "OK ${tag} best=$(grep -E '^minADE' ${out}/log_train_*.txt|awk '{print $2}'|sort -g|head -1)" >> "$LOG" \
      || echo "FAIL ${tag}" >> "$LOG"
  done
done
echo "=== phase9 finished $(date) ===" >> "$LOG"
