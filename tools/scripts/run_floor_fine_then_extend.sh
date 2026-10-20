#!/bin/bash
# Reorder the queue (2026-10-20): the Sec. IV reframe will quote the detection-floor
# threshold, so localising it moves onto the critical path ahead of the n=8 heading
# extension (which tests redundancy -- now the secondary argument).
#
# The coarse ladder put the floor in (0, 14.6] deg, a factor-of-three bracket. Two fine
# rungs at a measured 5.5 and 10.2 deg median bearing error close it. 6 runs, ~5.3 h.
#
# Stops the extension BETWEEN runs rather than killing it mid-run: the in-flight run is
# 38 min into ~53, and its DONE marker is written by the extension script itself, so
# killing now would discard the whole run. Afterwards the extension is relaunched and
# skips everything already marked DONE.
set -u
cd "$(dirname "$0")/.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/floor_fine.log
echo "=== fine-rung reorder started $(date) ===" >> "$LOG"

# 1. let the in-flight extension run finish and be marked, then stop the extension loop
echo "waiting for the in-flight extension run to be marked OK" >> "$LOG"
for i in $(seq 1 480); do
  grep -q 'OK MS_nohead_nopose_s404' /tmp/nohead_extend.log 2>/dev/null && break
  sleep 15
done
pkill -f '[r]un_nohead_extend.sh' && echo "extension loop stopped $(date)" >> "$LOG"
sleep 5

run_one() {   # $1 = tag, $2 = config, $3 = seed
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
}

# 2. the fine rungs
for seed in 101 202 303; do
  for m in 05 10; do
    run_one "MS_oracledir${m}_s${seed}" "mtr+pose_data_cross_attn_pe_with_map_oracledir${m}_fixdata" "$seed"
  done
done
echo "=== fine rungs done $(date) ===" >> "$LOG"

# 3. resume the heading extension (skips every run already marked DONE)
nohup bash tools/scripts/run_nohead_extend.sh >> /tmp/nohead_extend_outer.log 2>&1 &
echo "=== extension relaunched $(date) ===" >> "$LOG"
