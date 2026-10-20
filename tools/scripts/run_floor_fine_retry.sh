#!/bin/bash
# Resume the fine detection-floor rungs after a foreign-job collision (2026-10-20).
#
# MS_oracledir10_s101 died with CUDA OOM at 14:48 when an unrelated job (train_lost.py,
# not ours) took ~7.3 GB of the 24 GB card while our run was mid-training. Our runs peak
# near 18 GB, so the two do not fit. The original queue logged FAIL and moved on, which
# would have left a hole in the ladder, and it has no retry.
#
# This version: re-lists every rung (run_one skips anything already marked DONE, so the
# completed 05_s101 is not repeated), and retries a run once if it fails, since the
# failure mode is transient contention rather than a bug. Foreign jobs are never touched;
# we simply wait for the card.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/floor_fine.log
echo "=== fine-rung queue resumed with retry $(date) ===" >> "$LOG"

launch() {    # $1 tag, $2 cfg, $3 seed -> 0 on success
  local tag="$1" cfg="$2" seed="$3"
  local out="../output/waymo/${cfg}/${tag}"
  while true; do
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break
    sleep 270
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 \
    "$PY" train.py --cfg_file "cfgs/waymo/${cfg}.yaml" --extra_tag "${tag}" \
    --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1
}

run_one() {   # $1 tag, $2 cfg, $3 seed
  local tag="$1" cfg="$2" seed="$3"
  local out="../output/waymo/${cfg}/${tag}"
  [ -f "${out}/DONE" ] && { echo "SKIP ${tag} (done)" >> "$LOG"; return; }
  for attempt in 1 2; do
    if launch "$tag" "$cfg" "$seed"; then
      touch "${out}/DONE"
      rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl"
      echo "OK ${tag} $(date)" >> "$LOG"; return
    fi
    if grep -q 'OutOfMemoryError' "/tmp/${tag}.log" 2>/dev/null; then
      echo "OOM ${tag} attempt ${attempt} -- foreign job on the card, waiting 20 min $(date)" >> "$LOG"
      sleep 1200
    else
      echo "FAIL ${tag} attempt ${attempt} (not OOM) $(date)" >> "$LOG"; break
    fi
  done
  [ -f "${out}/DONE" ] || echo "FAIL ${tag} (gave up) $(date)" >> "$LOG"
}

for seed in 101 202 303; do
  for m in 05 10; do
    run_one "MS_oracledir${m}_s${seed}" "mtr+pose_data_cross_attn_pe_with_map_oracledir${m}_fixdata" "$seed"
  done
done
echo "=== fine rungs done $(date) ===" >> "$LOG"

nohup bash tools/scripts/run_nohead_extend.sh >> /tmp/nohead_extend_outer.log 2>&1 &
echo "=== extension relaunched $(date) ===" >> "$LOG"
