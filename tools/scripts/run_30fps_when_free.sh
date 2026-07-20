#!/bin/bash
# Scheduled (2026-07-15): seed-replicate the 30fps future-pose-supervision ablation.
# The GPU is currently occupied by another user's long job, so this poller waits until
# the GPU is free (no non-display compute process, low memory) for two consecutive
# checks, then runs 3 seeds of mtr+pose_data_30fps (full pose loss set — the ONE
# condition where the geodesic/MPJPE terms get real gradients), then runs the analysis.
# Detach with:  setsid nohup bash tools/scripts/run_30fps_when_free.sh >/tmp/run_30fps_when_free.out 2>&1 &
set -u
cd "$(dirname "$0")/.."   # tools/
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/run_30fps_when_free.log
SEEDS="101 202 303"
CFG=mtr+pose_data_30fps
MEM_THRESH=2500          # MiB; below this (and no compute apps) => free
POLL=300                 # seconds between checks

log(){ echo "$(date '+%F %T') $*" >> "$LOG"; }

gpu_free(){
  # non-display compute processes?
  local apps mem
  apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
         | grep -viE 'gnome-remote|xorg|/usr/lib/xorg|Xwayland' | grep -v '^[[:space:]]*$')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ -z "$apps" ] && [ -n "$mem" ] && [ "$mem" -lt "$MEM_THRESH" ]
}

log "=== scheduler started; waiting for a free GPU (thresh ${MEM_THRESH}MiB) ==="
free_streak=0
while true; do
  if gpu_free; then
    free_streak=$((free_streak+1))
    log "gpu appears free (streak=$free_streak/2)"
    [ "$free_streak" -ge 2 ] && break
  else
    free_streak=0
  fi
  sleep "$POLL"
done
log "=== GPU free — launching 30fps seed replication ==="

for seed in $SEEDS; do
  tag="MS_pose30fps_s${seed}"; outdir="../output/waymo/${CFG}/${tag}"
  [ -f "${outdir}/DONE" ] && { log "skip ${tag} (done)"; continue; }
  # bail out if the GPU got taken again right before we start this run
  tries=0; until gpu_free || [ $tries -ge 288 ]; do sleep "$POLL"; tries=$((tries+1)); done
  log "start ${tag}"
  WANDB_MODE=offline "$PY" train.py --cfg_file "cfgs/waymo/${CFG}.yaml" \
    --extra_tag "${tag}" --random_seed "${seed}" --workers 8 --max_ckpt_save_num 1 \
    > "/tmp/${tag}.log" 2>&1
  if [ $? -eq 0 ]; then
    touch "${outdir}/DONE"
    rm -f "${outdir}/ckpt/best_model.pth" "${outdir}/eval/eval_with_train/result.pkl"
    best=$(grep -E "^minADE" "${outdir}"/log_train_*.txt | awk '{print $2}' | sort -g | head -1)
    log "OK ${tag} best_minADE=${best}"
  else
    log "FAIL ${tag} (see /tmp/${tag}.log)"
  fi
done

log "=== training done; running analysis ==="
"$PY" scripts/analyze_multiseed.py >> "$LOG" 2>&1 || log "analyze failed"
log "=== 30fps seed replication COMPLETE ==="
