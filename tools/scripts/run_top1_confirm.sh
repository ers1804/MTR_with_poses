#!/bin/bash
# Confirmation run: the top-1 finding currently rests on LAST-epoch checkpoints, where the
# mode-classification head may simply be undertrained. Retrain the 4 map cells so that
# top1_ADE / best_mode_rank / mode_spread are recorded at EVERY eval epoch, giving
# best-epoch numbers on the paper's protocol plus stability across training.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/top1_confirm.log
declare -A C=(
  [t1_nopose]=mtr+pose_data_no_pose_with_map
  [t1_xattnpe]=mtr+pose_data_cross_attn_pe_with_map
  [t1_shuffle]=mtr+pose_data_cross_attn_pe_with_map_shufflepose
  [t1_const]=mtr+pose_data_cross_attn_pe_with_map_constpose
)
echo "=== top1-confirm started $(date) ===" >> "$LOG"
for cell in t1_nopose t1_xattnpe t1_shuffle t1_const; do
  for seed in 101 202 303 404 505; do
    tag="MS_${cell}_s${seed}"; out="../output/waymo/${C[$cell]}/${tag}"
    [ -f "${out}/DONE" ] && continue
    WANDB_MODE=offline "$PY" train.py --cfg_file "cfgs/waymo/${C[$cell]}.yaml" --extra_tag "${tag}" \
      --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
      && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
      && echo "OK ${tag}" >> "$LOG" || echo "FAIL ${tag}" >> "$LOG"
  done
done
echo "=== top1-confirm finished $(date) ===" >> "$LOG"
