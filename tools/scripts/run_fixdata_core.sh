#!/bin/bash
# Clean-data re-run of the three core cells (post export-bug fix), n=3 first.
# Same 579 training / 5,171 validation scenes as the paper, but poses re-exported
# per-scene (export_final_data_fixed.py) -> 6,348 train peds (was 1,348 corrupt).
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/multiseed_fixdata.log
declare -A C=(
  [map_nopose_fix]=mtr+pose_data_no_pose_with_map_fixdata
  [map_xattn_pe_fix]=mtr+pose_data_cross_attn_pe_with_map_fixdata
  [map_shufflepose_fix]=mtr+pose_data_cross_attn_pe_with_map_shufflepose_fixdata
)
echo "=== fixdata queue started $(date) ===" >> "$LOG"
for seed in 101 202 303; do
  for cell in map_nopose_fix map_xattn_pe_fix map_shufflepose_fix; do
    tag="MS_${cell}_s${seed}"; out="../output/waymo/${C[$cell]}/${tag}"
    [ -f "${out}/DONE" ] && continue
    # wait for a free GPU (2 consecutive checks, foreign jobs respected)
    while true; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      sleep 30
      used2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$used" -lt 4000 ] && [ "$used2" -lt 4000 ] && break
      sleep 270
    done
    WANDB_MODE=offline "$PY" train.py --cfg_file "cfgs/waymo/${C[$cell]}.yaml" --extra_tag "${tag}" \
      --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
      && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
      && echo "OK ${tag}" >> "$LOG" \
      || echo "FAIL ${tag}" >> "$LOG"
  done
done
echo "=== fixdata queue finished $(date) ===" >> "$LOG"
