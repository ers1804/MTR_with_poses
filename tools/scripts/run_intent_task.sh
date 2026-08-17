#!/bin/bash
# R2 runner: waits for intent labels, then 3 variants x 5 seeds of the intent classifier.
set -u
cd "$(dirname "$0")/../.."
PY=/home/erik/anaconda3/envs/mtr_smpl/bin/python
LOG=/tmp/intent_task.log
t=0
until [ -s experiments/intent_labels.json ]; do
  sleep 60; t=$((t+1)); [ $t -ge 120 ] && { echo "labels never appeared" >> "$LOG"; exit 1; }
done
echo "=== intent task started $(date) ===" >> "$LOG"
for variant in nopose pose wrongpose; do
  for seed in 101 202 303 404 505; do
    "$PY" tools/scripts/intent_classifier.py --variant "$variant" --seed "$seed" >> "$LOG" 2>&1 \
      && echo "OK ${variant}_s${seed}" >> "$LOG" || echo "FAIL ${variant}_s${seed}" >> "$LOG"
  done
done
echo "=== intent task finished $(date) ===" >> "$LOG"
