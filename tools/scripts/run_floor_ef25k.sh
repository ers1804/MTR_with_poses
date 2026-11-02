#!/bin/bash
# The EARLY-FUSION oracle ladder at the 25k-scene rung.
#
# WHY. Sec. V-C says the cheaper pathway "has no floor to measure": at 579 scenes
# no early-fusion rung is significant, including the uncorrupted one (-4.5% at
# theta=0, p=0.44, MDE 25.5 pp, n=3). That reading does not survive what the late
# ladder just taught us. At 579 the late ladder also read a clean floor at 15.1
# deg; at 25k the same rung is -13.6%, because the 579 rungs could only resolve
# 5.5-12.3 pp. An MDE of 25.5 pp cannot distinguish "the pathway ignores the
# channel" from "the pathway converts it and we cannot see it" -- and the first is
# what the paper currently concludes. The same power argument that retired the
# bracket applies here, so the honest fix is to re-run this ladder at 25k, where
# the late ladder's MDE was 1.4-2.9 pp at the same n=3.
#
# RUNG ORDER IS A GATE. Rung 00 (an uncorrupted oracle) runs first, all three
# seeds, before anything else. It alone settles the question the paper gets wrong:
# does early fusion convert a perfect direction channel at 25k? If it does, the
# current withdrawal of the early-fusion suggestion has to be withdrawn in turn.
# If it does not -- at an MDE near 1.5 pp rather than 25.5 -- then "the channel
# reaches the input and is not used" becomes a measurement instead of a guess.
# Informative after 3 runs, complete after 9.
#
# Configs are the 579 early-fusion configs with exactly three values changed
# (DATA_ROOT, PROCESSED_SCENARIOS_PATH, LR 1e-4 -> 2.5e-5), so they differ from
# the LATE 25k configs only by POSE_EARLY_FUSION, NUM_INPUT_ATTR_AGENT 29->35 and
# the disabled pose encoder / cross-attention. At a given rung the six early
# channels equal the branch's six inputs bit for bit, so the rungs keep their
# measured-error labels and the two ladders are directly comparable.
#
# LR. 2.5e-5, the per-arm optimum for all three 25k cells in the HELMA sweep, and
# the same LR the late 25k ladder and the paired no-pose baseline used. Inheriting
# the 579 ladder's 1e-4 at 43x the data is what manufactured a fake +37% effect in
# August.
#
# Denominator comes free: MS_scale25k_nopose (n=5, 0.3683-0.3752) is the same
# paired baseline the late ladder used, so no baseline runs are needed here.
#
# Output lives on big1, not ssd2: ssd2 is full, and a run needs ~2.6 GiB transient
# to write a 1.3 GiB checkpoint while the previous one still exists. The pre-flight
# below refuses to start unless each output dir is a symlink with 50 GiB behind it,
# and a write failure aborts the whole queue rather than burning the other runs.
#
# LAUNCH IT WITH setsid:
#     setsid nohup bash tools/scripts/run_floor_ef25k.sh > /tmp/floor_ef25k_wrapper.log 2>&1 &
# nohup alone is NOT enough -- on 2026-10-28 a session crash took the process group
# with it and idled the GPU for eight hours. setsid makes this script a session
# leader. Recovery is cheap either way (train.py reloads the newest checkpoint, so
# a relaunch resumes), but only if something relaunches it.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."                       # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/floor_ef25k.log
STAGE=/home/erik/big1/waymo_agent_batch_25k
WANT=30171                          # 25,000 training + 5,171 validation scenario pkls
RUNGS="00 05 10"
CFG_FOR() { echo "mtr+pose_data_early_oracledir${1}_scale25k"; }

echo "=== ef queue started $(date) ===" >> "$LOG"

# --- the scenario store is already staged; verify rather than wait ------------
# A missing pkl is silent: _load_scenario returns (None, None) and the scene loses
# its map, so a partial store would leave every arm map-degraded by a different
# random amount. The late ladder staged this same directory, so this is a check,
# not a copy.
have=$(find "$STAGE" -name '*.pkl' | wc -l)
if [ "$have" -lt "$WANT" ]; then
  echo "ABORT staging incomplete: $have/$WANT pkls $(date)" >> "$LOG"; exit 1
fi
echo "staging ok: $have pkls $(date)" >> "$LOG"

# --- refuse to start without somewhere to put a 1.3 GiB checkpoint ------------
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
  # wait for a quiet card: foreign jobs are never touched, we just wait
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
      # keep metrics_epoch_*.pkl: the per-agent minFDE / minADE_{2,4,8}s / mode
      # columns in them are what the horizon decomposition reads, and result.pkl
      # alone is 2.6 GiB of the same predictions in a form nothing reads twice.
      rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl"
      echo "OK ${tag} $(date)" >> "$LOG"; return 0
    fi
    if grep -qE 'No space left|failed writing file|unexpected pos' \
            "/tmp/${tag}.log" 2>/dev/null; then
      # Every remaining run would fail the same way at the end of its first epoch.
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
  echo "--- ef rung ${m} $(date) ---" >> "$LOG"
  for seed in 101 202 303; do
    run_one "MS_earlyoracle${m}_25k_s${seed}" "$(CFG_FOR "$m")" "$seed"
  done
done

echo "=== ef queue finished $(date) ===" >> "$LOG"
