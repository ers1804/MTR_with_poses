#!/bin/bash
# The two HIGH rungs of the late-fusion oracle ladder at 25k: 24.9 and 44.4 deg.
#
# WHY. Eq. (1) decomposes the ladder into a precision term with a 3.0 deg
# half-life and a term the 0-15.1 deg rungs do not resolve. The paper used to call
# that second term corruption-invariant, which is more than four rungs over a
# 15 deg span can support: a 3-parameter fit to 4 points has one degree of
# freedom, and its asymptote is an extrapolation, not a measurement. The 579
# ladder does reach 24.9-60.1 deg and reads -0.1 to +2.7% there, with intervals
# that exclude the ~6% a persistent term scaled from its own theta=0 amplitude
# would predict -- so the term decays somewhere between 15 and 25 deg, as it must,
# since a 60 deg median error is close to uniform on the circle. But those 579
# rungs carry MDEs of 5.5-9.1 pp, which is the same blindness that manufactured
# the detection floor, so they cannot settle it either.
#
# These two rungs settle it at 25k, where the ladder's MDE was 1.4-2.9 pp at
# n=3. Either the persistent term is still there at 24.9 deg, and Eq. (1) holds
# over a span wide enough to mean something, or it is gone, and the honest claim
# is a slow decay with a measured end. The paper is written for the second
# reading, so this queue can only strengthen it or sharpen it.
#
# RUNG 25 IS THE GATE: it is the first rung beyond the fitted span and the one
# the 579 ladder puts at zero. Informative after 3 runs, complete after 6.
#
# Configs are the rung-15 25k config with ONE value changed, the per-agent
# corruption sigma (22.2 -> 37.1 / 66.7 deg), taken from the 579 configs so both
# ladders' rungs carry the same measured-error labels. Everything else -- data
# root, scenario store, LR 2.5e-5, the branch -- is bit-for-bit the rung-15 setup,
# and the paired denominator is the same MS_scale25k_nopose (n=5) the rest of the
# ladder uses, so no baseline runs are needed.
#
# WAITS FOR THE EARLY-FUSION QUEUE. One card, so this gates on run_floor_ef25k.sh
# leaving the process table before it stages anything. Foreign jobs are never
# touched; we only ever wait.
#
# LAUNCH IT WITH setsid:
#     setsid nohup bash tools/scripts/run_floor_hi25k.sh > /tmp/floor_hi25k_wrapper.log 2>&1 &
# nohup alone is NOT enough -- on 2026-10-28 a session crash took the process
# group with it and idled the GPU for eight hours.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."                       # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/floor_hi25k.log
STAGE=/home/erik/big1/waymo_agent_batch_25k
WANT=30171                          # 25,000 training + 5,171 validation scenario pkls
RUNGS="25 45"
CFG_FOR() { echo "mtr+pose_data_cross_attn_pe_with_map_oracledir${1}_scale25k"; }

echo "=== hi queue started $(date) ===" >> "$LOG"

# --- wait out the early-fusion ladder ----------------------------------------
# Bracket the pattern so this grep cannot match its own command line.
while pgrep -f '[r]un_floor_ef25k' > /dev/null; do
  echo "waiting: ef queue still running $(date)" >> "$LOG"
  sleep 1800
done
echo "ef queue clear $(date)" >> "$LOG"

have=$(find "$STAGE" -name '*.pkl' | wc -l)
if [ "$have" -lt "$WANT" ]; then
  echo "ABORT staging incomplete: $have/$WANT pkls $(date)" >> "$LOG"; exit 1
fi
echo "staging ok: $have pkls $(date)" >> "$LOG"

free=0
for m in $RUNGS; do
  d="../output/waymo/$(CFG_FOR "$m")"
  if [ ! -L "$d" ]; then
    echo "ABORT $d is not a symlink; output would land on a full ssd2 $(date)" >> "$LOG"
    exit 1
  fi
  free=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
  if [ "${free:-0}" -lt 50 ]; then
    echo "ABORT only ${free}G free behind $d, need 50G $(date)" >> "$LOG"; exit 1
  fi
done
echo "output ok: symlinked, ${free}G free $(date)" >> "$LOG"

launch() {    # tag cfg seed
  local tag="$1" cfg="$2" seed="$3" u1=0 u2=0
  while true; do
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 20
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${u1:-99999}" -lt 4000 ] && [ "${u2:-99999}" -lt 4000 ] && break
    sleep 240
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 \
    "$PY" train.py --cfg_file "cfgs/waymo/${cfg}.yaml" --extra_tag "${tag}" \
    --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1
}

run_one() {   # tag cfg seed
  local tag="$1" cfg="$2" seed="$3"
  local out="../output/waymo/${cfg}/${tag}"
  [ -f "${out}/DONE" ] && { echo "SKIP ${tag} (done)" >> "$LOG"; return 0; }
  local attempt
  for attempt in 1 2; do
    if launch "$tag" "$cfg" "$seed"; then
      touch "${out}/DONE"
      # keep metrics_epoch_*.pkl: the per-agent horizon and mode columns in them
      # are what horizon_decomposition.py and top1_content_25k.py read.
      rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl"
      echo "OK ${tag} $(date)" >> "$LOG"; return 0
    fi
    if grep -qE 'No space left|failed writing file|unexpected pos' \
            "/tmp/${tag}.log" 2>/dev/null; then
      echo "ABORT ${tag}: out of disk $(date)" >> "$LOG"; exit 1
    elif grep -q 'OutOfMemoryError' "/tmp/${tag}.log" 2>/dev/null; then
      echo "OOM ${tag} attempt ${attempt} -- waiting 20 min $(date)" >> "$LOG"; sleep 1200
    else
      echo "FAIL ${tag} attempt ${attempt} (not OOM) $(date)" >> "$LOG"; break
    fi
  done
  if [ ! -f "${out}/DONE" ]; then
    echo "FAIL ${tag} (gave up) $(date)" >> "$LOG"
  fi
  return 0
}

for m in $RUNGS; do
  echo "--- hi rung ${m} $(date) ---" >> "$LOG"
  for seed in 101 202 303; do
    run_one "MS_oracledir${m}_25k_s${seed}" "$(CFG_FOR "$m")" "$seed"
  done
done

echo "=== hi queue finished $(date) ===" >> "$LOG"
