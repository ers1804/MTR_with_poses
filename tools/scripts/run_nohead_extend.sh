#!/bin/bash
# Extend the heading-ablation difference-in-differences to n=8 (2026-10-19).
#
# At n=3 the DiD in the stationary subpopulation is -4.10pp with all three seeds agreeing
# in sign, p=0.098. That is the direction and the location the mechanism predicts, and it
# is also exactly the regime where this project's xattn+PE headline (-3.5% at n=3 -> n.s.
# at n=5) and AVO candidate (-1.18% -> -0.04% at n=8) both dissolved. Power: the DiD sd is
# 2.41pp, so n=5 detects 4.0pp at 80% power against an observed 4.1pp -- right at the edge,
# where a null would be uninterpretable. n=8 detects 2.8pp and settles it either way.
#
# The DiD pairs a seed ACROSS conditions, so both triads must exist at the same seeds.
# With-heading already has 101-505; no-heading has 101-303. Seed-major order below means
# the paired n grows by one usable seed at a time (n=5 after 505, n=8 after 808).
#
# 24 runs, ~53 min each, ~21 h. Waits for the detection-floor ladder to finish first.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/nohead_extend.log

echo "waiting for the mechanism queue to finish $(date)" >> "$LOG"
while ! grep -q 'mechanism queue finished' /tmp/mechanism_queue.log 2>/dev/null; do sleep 120; done
echo "=== nohead n=8 extension started $(date) ===" >> "$LOG"

run_one() {   # $1 = run tag, $2 = config basename, $3 = seed
  local tag="$1"; local cfg="$2"; local out="../output/waymo/${cfg}/${tag}"
  [ -f "${out}/DONE" ] && { echo "SKIP ${tag}" >> "$LOG"; return; }
  while true; do
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break
    sleep 270
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 \
    "$PY" train.py --cfg_file "cfgs/waymo/${cfg}.yaml" --extra_tag "${tag}" \
    --random_seed "${3}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1 \
    && touch "${out}/DONE" && rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl" \
    && echo "OK ${tag} $(date)" >> "$LOG" || echo "FAIL ${tag} $(date)" >> "$LOG"
  # NOTE: metrics_epoch_*.pkl are deliberately kept -- they carry the per-agent records the
  # subpopulation analysis re-scores at the best epoch without any further inference.
}

for seed in 404 505 606 707 808; do
  run_one "MS_nohead_nopose_s${seed}" mtr+pose_data_no_pose_with_map_nohead_fixdata            "$seed"
  run_one "MS_nohead_pose_s${seed}"   mtr+pose_data_cross_attn_pe_with_map_nohead_fixdata      "$seed"
  run_one "MS_nohead_shuf_s${seed}"   mtr+pose_data_cross_attn_pe_with_map_shufflepose_nohead_fixdata "$seed"
  case "$seed" in
    606|707|808)   # the with-heading reference only exists up to seed 505
      run_one "MS_map_nopose_fix_s${seed}"     mtr+pose_data_no_pose_with_map_fixdata                    "$seed"
      run_one "MS_map_xattn_pe_fix_s${seed}"   mtr+pose_data_cross_attn_pe_with_map_fixdata              "$seed"
      run_one "MS_map_shufflepose_fix_s${seed}" mtr+pose_data_cross_attn_pe_with_map_shufflepose_fixdata "$seed"
      ;;
  esac
  echo "--- paired seeds complete through s${seed} $(date) ---" >> "$LOG"
done
echo "=== nohead n=8 extension finished $(date) ===" >> "$LOG"
