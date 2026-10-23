#!/bin/bash
# Two queued jobs, sequential because there is one GPU.
#
# PART A (4 runs) -- tighten the floor bracket. The ladder is n=3 per rung and its
# MDE beyond 10 deg is ~6 pp, which is why Sec. IV reports a bracket rather than a
# crossing point. Seeds 404/505 at the two rungs that define the bracket (10.2 and
# 14.6 deg) take those to n=5. Baselines (map_nopose_fix s404/s505) already exist.
#
# PART B (9 runs) -- the floor of the EARLY-fusion pathway. Sec. III-B shows early
# fusion removes the branch-capacity tax entirely, so the paper recommends a pathway
# whose floor was never measured; Sec. IV-B admits this. Oracle rungs 0/5.5/10.2 deg
# carried in the agent channels instead of a side branch. Compared against the same
# map_nopose_fix baselines.
#
# Path note: resolve everything from this script's own directory. A previous queue
# cd'd to tools/ and then used a repo-relative path, which silently left the GPU idle.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."                       # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/floor_extend_early.log
echo "=== queue started $(date) ===" >> "$LOG"

launch() {    # tag cfg seed
  local tag="$1" cfg="$2" seed="$3"
  # wait for a quiet card: foreign jobs are never touched, we just wait
  while true; do
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 20
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u1" -lt 4000 ] && [ "$u2" -lt 4000 ] && break
    sleep 240
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 \
    "$PY" train.py --cfg_file "cfgs/waymo/${cfg}.yaml" --extra_tag "${tag}" \
    --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1
}

run_one() {   # tag cfg seed
  local tag="$1" cfg="$2" seed="$3"
  local out="../output/waymo/${cfg}/${tag}"
  [ -f "${out}/DONE" ] && { echo "SKIP ${tag} (done)" >> "$LOG"; return; }
  for attempt in 1 2; do
    if launch "$tag" "$cfg" "$seed"; then
      touch "${out}/DONE"
      rm -f "${out}/ckpt/best_model.pth"
      echo "OK ${tag} $(date)" >> "$LOG"; return
    fi
    if grep -q 'OutOfMemoryError' "/tmp/${tag}.log" 2>/dev/null; then
      echo "OOM ${tag} attempt ${attempt} -- waiting 20 min $(date)" >> "$LOG"; sleep 1200
    else
      echo "FAIL ${tag} attempt ${attempt} (not OOM) $(date)" >> "$LOG"; break
    fi
  done
  [ -f "${out}/DONE" ] || echo "FAIL ${tag} (gave up) $(date)" >> "$LOG"
}

echo "--- PART A: bracket rungs to n=5 $(date) ---" >> "$LOG"
for seed in 404 505; do
  for m in 10 15; do
    run_one "MS_oracledir${m}_s${seed}" \
            "mtr+pose_data_cross_attn_pe_with_map_oracledir${m}_fixdata" "$seed"
  done
done

echo "--- PART B: early-fusion floor $(date) ---" >> "$LOG"
for seed in 101 202 303; do
  for m in 00 05 10; do
    run_one "MS_earlyoracle${m}_s${seed}" \
            "mtr+pose_data_early_oracledir${m}_fixdata" "$seed"
  done
done

echo "=== queue finished $(date) ===" >> "$LOG"
