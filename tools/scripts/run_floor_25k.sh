#!/bin/bash
# The detection-floor ladder at the 25k-scene rung.
#
# WHY. The published floor bracket is measured on the 579-scene subset while the
# branch-capacity tax it is weighed against is measured at 25k (disclosed in
# 5cfb4d5). Two numbers at two scales is the weakest joint in Sec. IV, and 579 is
# a cheap iteration rung that exists only because a run there takes ~52 min -- it
# is not a data limit (the corrected export covers all 299,220 scenes). Re-running
# the ladder at 25k states the floor and the tax at ONE scale.
#
# Power. The 25k no-pose baseline is far tighter than the 579 one: sd 0.0027 over
# five seeds (0.73% of its mean) against a 0.4428-0.4954 spread at 579, where
# baseline scatter -- not the oracle arm -- dominated the ladder's variance. n=3
# here should buy a smaller MDE than n=5 did at 579.
#
# RUNG ORDER IS A GATE, NOT A PREFERENCE. Rung 00 (an uncorrupted oracle) runs
# first and all three seeds of it before anything else. If a clean oracle does not
# convert at 25k, every corrupted rung reads n.s. for a reason that has nothing to
# do with a floor -- which is exactly how Part B's early-fusion ladder failed, and
# there it cost nine runs to find out. Three runs here instead. Then 10 and 15,
# the pair that defines the published bracket edge, then 05 as the monotonicity
# check below it. The queue is therefore informative after 6 runs and complete
# after 12.
#
# Top of the ladder comes free: MS_scale25k_xattn_pe (real SMPL poses, n=5) is the
# rung-infinity reference and is already in hand, as is MS_scale25k_nopose (n=5),
# the paired denominator.
#
# LR. 2.5e-5, which the HELMA sweep (slurm/10_scale_sweep.sh, 4 LRs x 3 cells x 3
# seeds) found to be the per-arm optimum for ALL THREE 25k cells -- no_pose
# 0.3711, xattn_pe 0.3773, shufflepose 0.3770, each the best of its row. So the
# common LR and the per-arm best coincide here and one LR is legitimate. This is
# NOT the 579 ladder's 1e-4; inheriting that LR at 4.9x data is what manufactured
# a +37% fake "pose hurts" effect in August.
#
# Output lives on big1, not ssd2. ssd2 is 100% full with 2.2 GiB free and 234 GiB of
# it is finished experiments' checkpoints; a run needs ~2.6 GiB transient to write a
# 1.3 GiB checkpoint while the previous one still exists. The first attempt at this
# queue died exactly there -- "PytorchStreamWriter failed writing file data/385" at the
# end of epoch 1 -- and then burned 28 min failing the other two seeds the same way,
# because a non-OOM failure only skips to the next run. So output/waymo/<cfg> for these
# four configs are symlinks into /home/erik/big1/mtr_output, the pre-flight below
# refuses to start without them, and a write failure now aborts the whole queue.
#
# Data. Scenario pkls are staged to local disk first: 22 GiB of them, read for 30
# epochs x 12 runs. The 579 ladder read them off the NFS mount, which is fine for
# a 52 min run and a 667 MiB working set, but a mount hiccup eight days into this
# queue is not worth risking. HELMA did the same thing with node-local NVMe.
#
# LAUNCH IT WITH setsid:
#     setsid nohup bash tools/scripts/run_floor_25k.sh > /tmp/floor_25k_wrapper.log 2>&1 &
# nohup alone is NOT enough. On 2026-10-28 the controlling Claude Code session crashed
# and took the whole process group with it, killing a run 8 h 26 m into epoch 18 and
# leaving the GPU idle for eight hours before anyone noticed. setsid makes this script a
# session leader, so it outlives whatever started it. Recovery is cheap either way --
# train.py globs ckpt/, skips best_model.pth and reloads the newest checkpoint, so a
# relaunched run resumes instead of restarting -- but only if something relaunches it.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."                       # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/floor_25k.log
STAGE=/home/erik/big1/waymo_agent_batch_25k
WANT=30171                          # 25,000 training + 5,171 validation scenario pkls

echo "=== queue started $(date) ===" >> "$LOG"

# --- wait for staging, and refuse to train on a partial copy ------------------
# A missing pkl is silent: _load_scenario returns (None, None) and the scene
# simply loses its map. Trained on a half-staged store, every arm would be
# quietly map-degraded by a different random amount.
# Poll the count rather than the copier's process name: the staging job has already
# been restarted once (rsync -> parallel cp) and the queue must not care which tool
# finished it. Give up after 4 h rather than wait on a copy that has died.
deadline=$(( $(date +%s) + 14400 ))
while true; do
  have=$(find "$STAGE" -name '*.pkl' | wc -l)
  [ "$have" -ge "$WANT" ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "ABORT staging stalled at $have/$WANT pkls $(date)" >> "$LOG"; exit 1
  fi
  sleep 60
done
echo "staging ok: $have pkls $(date)" >> "$LOG"

# --- refuse to start without somewhere to put a 1.3 GiB checkpoint ------------
for m in 00 05 10 15; do
  d="../output/waymo/mtr+pose_data_cross_attn_pe_with_map_oracledir${m}_scale25k"
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
      rm -f "${out}/ckpt/best_model.pth" "${out}/eval/eval_with_train/result.pkl"
      echo "OK ${tag} $(date)" >> "$LOG"; return
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
  [ -f "${out}/DONE" ] || echo "FAIL ${tag} (gave up) $(date)" >> "$LOG"
}

for m in 00 10 15 05; do
  echo "--- rung ${m} $(date) ---" >> "$LOG"
  for seed in 101 202 303; do
    run_one "MS_oracledir${m}_25k_s${seed}" \
            "mtr+pose_data_cross_attn_pe_with_map_oracledir${m}_scale25k" "$seed"
  done
done

echo "=== queue finished $(date) ===" >> "$LOG"
