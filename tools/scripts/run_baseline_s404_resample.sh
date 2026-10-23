#!/bin/bash
# A second sample of the seed-404 no-pose baseline.
#
# WHY. map_nopose_fix_s404 = 0.4954 is a ~2 SD outlier for its cell (others
# 0.4428-0.4584) while seed 404 averages BETTER than its cell mean across 37
# other cells (within-cell z = -0.37), and the sibling cells map_xattn_pe_fix
# and map_shufflepose_fix put the same seed at 0.4570 and 0.4547. That makes it
# a bad run rather than a slow seed. It is also the paired denominator for the
# two rungs Part A is meant to sharpen, and it dominates them: at rung 10.2 its
# pair reads -15.45% against -2.38/-2.75/-6.00 for the other three seeds, which
# lifts the rung's MDE from 6.2 to 12.7 pp. With it, resolving that rung needs
# n=9; without it, n=5 suffices.
#
# WHY THIS IS NOT CHERRY-PICKING. The outlier criterion -- a >2 SD within-cell
# baseline from a seed that is not systematically slow -- was committed in
# fb8eee7, hours BEFORE the s404 oracle run at rung 10.2 finished. The ordering
# is in git. This run therefore tests a pre-specified suspicion rather than
# chasing a number.
#
# WHAT IS DONE WITH IT. Both samples are kept; the original is NOT overwritten.
# detection_floor.py reports the affected rungs under both denominators, and if
# the bracket edge differs between them the edge is reported as unresolved
# rather than picked. If this re-run also lands near 0.495, then 0.495 is simply
# that seed's value and the original stands.
#
# Waits for the main queue to finish so the two never race for the GPU.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."                       # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
QLOG=/tmp/floor_extend_early.log
LOG=/tmp/baseline_s404_resample.log
CFG=mtr+pose_data_no_pose_with_map_fixdata
TAG=MS_map_nopose_fix_s404_resample

echo "=== waiting for the main queue $(date) ===" >> "$LOG"
while ! grep -q '=== queue finished' "$QLOG" 2>/dev/null; do sleep 120; done
echo "=== main queue done, waiting for a quiet card $(date) ===" >> "$LOG"
while true; do
  u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 20
  u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break
  sleep 240
done

echo "=== training $TAG $(date) ===" >> "$LOG"
WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 \
  "$PY" train.py --cfg_file "cfgs/waymo/${CFG}.yaml" --extra_tag "$TAG" \
  --random_seed 404 --workers 8 --max_ckpt_save_num 1 > "/tmp/${TAG}.log" 2>&1
rc=$?
out="../output/waymo/${CFG}/${TAG}"
if [ $rc -eq 0 ]; then
  touch "${out}/DONE"; rm -f "${out}/ckpt/best_model.pth"
  echo "OK $TAG $(date)" >> "$LOG"
else
  echo "FAIL $TAG rc=$rc $(date)" >> "$LOG"
fi
