#!/bin/bash
# Local copy of the pre-fit rerun (slurm/60_heading_noise.sh, mode prefit), 30 cm rung.
#
# WHY. The Helma array (job 1300030) sits behind the account's GPU cap and may not
# start in time. This runs the same pairs on the local card: same configs, same
# tags, same recipe (TRAIN_MAX_CENTERS=24, LR 2.5e-5 from the yaml, 30 epochs).
#
# ORDER. Helma's array goes s101 -> s505, so this goes s505 -> s101. Whatever Helma
# starts first is then not what this has already done. Each seed runs pose then
# shuffled, so a finished seed is a complete pair from one machine.
#
# WAITS FOR THE FLOOR QUEUE (run_floor_hi25k.sh) to leave the process table, and
# before every launch for the card to be empty. Foreign jobs are never touched.
#
# 2026-11-11 09:25: Helma took all ten 30 cm tasks after all, so the 30 cm run here
# was stopped after two epochs and the card went to the 10 cm rung, which Helma does
# not run. NOISE picks the rung in cm. There is no Helma order to avoid at 10 cm, so
# launch that with SEEDS in ascending order.
#
# LAUNCH IT WITH setsid:
#     NOISE=10 SEEDS="101 202 303 404 505" setsid nohup bash tools/scripts/run_prefit_local.sh \
#         > /tmp/prefit_local_wrapper.log 2>&1 &
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."                       # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/prefit_local.log
NOISE="${NOISE:-30}"
SEEDS="${SEEDS:-505 404 303 202 101}"
ARMS="cross_attn_pe_with_map_prefit_posnoise${NOISE}_scale25k cross_attn_pe_with_map_shufflepose_prefit_posnoise${NOISE}_scale25k"
for arm in $ARMS; do
  [ -f "cfgs/waymo/mtr+pose_data_${arm}.yaml" ] || { echo "ABORT no config for ${arm} $(date)" >> "$LOG"; exit 1; }
done

echo "=== prefit local queue started, ${NOISE} cm, seeds ${SEEDS} $(date) ===" >> "$LOG"
while pgrep -f '[r]un_floor_hi25k' > /dev/null; do sleep 120; done
echo "floor queue clear $(date)" >> "$LOG"

for arm in $ARMS; do
  d="../output/waymo/mtr+pose_data_${arm}"
  if [ ! -L "$d" ]; then
    echo "ABORT $d is not a symlink; output would land on a full ssd2 $(date)" >> "$LOG"; exit 1
  fi
  free=$(df -BG --output=avail "$d/" | tail -1 | tr -dc '0-9')
  if [ "${free:-0}" -lt 50 ]; then
    echo "ABORT only ${free}G free behind $d, need 50G $(date)" >> "$LOG"; exit 1
  fi
done

launch() {    # tag arm seed
  local tag="$1" arm="$2" seed="$3" u1=0 u2=0
  while true; do
    u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 20
    u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${u1:-99999}" -lt 4000 ] && [ "${u2:-99999}" -lt 4000 ] && break
    sleep 240
  done
  WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRAIN_MAX_CENTERS=24 \
    "$PY" train.py --cfg_file "cfgs/waymo/mtr+pose_data_${arm}.yaml" --extra_tag "${tag}" \
    --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 > "/tmp/${tag}.log" 2>&1
}

run_one() {   # arm seed
  local arm="$1" seed="$2"
  local tag="PF_25k_${arm}_lr25_s${seed}"
  local out="../output/waymo/mtr+pose_data_${arm}/${tag}"
  [ -f "${out}/DONE" ] && { echo "SKIP ${tag} (done)" >> "$LOG"; return 0; }
  echo "START ${tag} $(date)" >> "$LOG"
  local attempt
  for attempt in 1 2; do
    if launch "$tag" "$arm" "$seed"; then
      touch "${out}/DONE"
      rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl"
      echo "OK ${tag} $(date)" >> "$LOG"; return 0
    fi
    if grep -qE 'No space left|failed writing file|unexpected pos' "/tmp/${tag}.log" 2>/dev/null; then
      echo "ABORT ${tag}: out of disk $(date)" >> "$LOG"; exit 1
    elif grep -q 'OutOfMemoryError' "/tmp/${tag}.log" 2>/dev/null; then
      echo "OOM ${tag} attempt ${attempt} -- waiting 20 min $(date)" >> "$LOG"; sleep 1200
    else
      echo "FAIL ${tag} attempt ${attempt} (not OOM) $(date)" >> "$LOG"; break
    fi
  done
  [ -f "${out}/DONE" ] || echo "FAIL ${tag} (gave up) $(date)" >> "$LOG"
  return 0
}

for seed in $SEEDS; do
  for arm in $ARMS; do
    run_one "$arm" "$seed"
  done
done
echo "=== prefit local queue finished $(date) ===" >> "$LOG"
