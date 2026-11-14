#!/bin/bash
# Held-out replication (heldout_replication.py): score the fifteen epoch-30 checkpoints of
# the 30 cm runs (pre-fit own pose, pre-fit stranger, no pose, five seeds) on the validation
# scenes that contain a pedestrian with at least 200 LiDAR returns. Inference only.
#
# It shares one GPU with the training queue (run_prefit_local.sh) and is written to stay out
# of its way:
#   - GPU memory is capped (eval_scene_subset.py), so a shortage fails here, not in training
#   - the queue launches a run only below 4 GB of GPU memory. The cap keeps an evaluation
#     under that, and should the card still read 3.9 GB or more while no training runs,
#     the evaluation is given up and resumed once the next run has started
#   - inputs are read from the SSD, because the training reads its data from the HDD
#
# Usage: bash tools/scripts/run_heldout_eval.sh     (logs to $LOGS, skips finished runs)
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="${STAGE:-/home/erik/ssd2/heldout_stage}"
CK="${CK:-/home/erik/big1/heldout_eval/ckpt}"
LOGS="${LOGS:-/home/erik/big1/heldout_eval/logs}"
LIST="$REPO/experiments/heldout_wellresolved_scenes.txt"
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
FRACTION="${FRACTION:-0.09}"          # of the GPU's memory, about 2.2 GB of 24
export CHUNK="${CHUNK:-8}"            # pedestrians per forward pass
cd "$REPO/tools"
export OMP_NUM_THREADS=4 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

queue_alive() { pgrep -f '[r]un_prefit_local.sh' > /dev/null; }
training()    { pgrep -f '[t]rain.py --cfg_file cfgs/waymo' > /dev/null; }
blocking()    {   # the queue is waiting to launch and this evaluation holds it up
  queue_alive && ! training &&
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 3900 ]
}
wait_for_training() {   # between two runs of the queue: stay off the GPU until the next has started
  local waited=0
  while queue_alive && ! training; do sleep 30; waited=1; done
  [ "$waited" -eq 1 ] && queue_alive && sleep 300     # let the new run reach its memory level
  return 0
}

run_eval() {    # tag arm batch_size -> 0 done, 2 gave way to the queue, else failed
  local tag="$1" a="$2" bs="$3" pid
  nice -n 10 "$PY" scripts/eval_scene_subset.py "$LIST" "$FRACTION" \
    --cfg_file "cfgs/waymo/mtr+pose_data_${a}.yaml" \
    --ckpt "$CK/$tag/checkpoint_epoch_30.pth" --extra_tag "HELDOUT_${tag}" \
    --batch_size "$bs" --workers 4 \
    --set DATA_CONFIG.DATA_ROOT "$STAGE/poses" DATA_CONFIG.PROCESSED_SCENARIOS_PATH "$STAGE/scenarios" \
    > "$LOGS/eval_${tag}.log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    if blocking; then
      sleep 5
      if blocking && kill -0 "$pid" 2>/dev/null; then
        kill "$pid"; wait "$pid" 2>/dev/null
        return 2
      fi
    fi
  done
  wait "$pid"
}

for seed in 101 202 303 404 505; do
  for cfg in cross_attn_pe_with_map_prefit_posnoise30_scale25k:PF cross_attn_pe_with_map_shufflepose_prefit_posnoise30_scale25k:PF no_pose_with_map_posnoise30_scale25k:PN; do
    a=${cfg%%:*}; p=${cfg##*:}
    tag=${p}_25k_${a}_lr25_s${seed}
    dir="$REPO/output/waymo/mtr+pose_data_${a}/HELDOUT_${tag}/eval/epoch_30/default"
    [ -f "$dir/metrics_epoch_30.pkl" ] && { echo "skip $tag (done)"; continue; }
    bs=2
    while true; do
      wait_for_training
      echo "start $tag batch $bs $(date +%H:%M:%S)"
      run_eval "$tag" "$a" "$bs"; rc=$?
      if [ "$rc" -eq 0 ] && [ -f "$dir/metrics_epoch_30.pkl" ]; then
        # the raw predictions are several GB. One file is kept: it holds the tracked
        # histories that tracker_state_error.py reads, the same in every arm and seed.
        [ "$tag" = "PF_25k_cross_attn_pe_with_map_prefit_posnoise30_scale25k_lr25_s101" ] || rm -f "$dir/result.pkl"
        echo "ok $tag $(date +%H:%M:%S)"; break
      elif [ "$rc" -eq 2 ]; then
        echo "gave way to the training queue during $tag $(date +%H:%M:%S)"
      elif grep -q 'OutOfMemoryError' "$LOGS/eval_${tag}.log" && [ "$bs" -gt 1 ]; then
        bs=1; echo "out of GPU memory in $tag, retrying with batch 1 $(date +%H:%M:%S)"
      else
        echo "FAILED $tag (exit $rc) $(date +%H:%M:%S)"; break
      fi
    done
  done
done
echo "EVAL DONE $(date)"
