#!/bin/bash
# Waits for a free GPU (no foreign compute process), then runs the top1 confirmation.
set -u
cd "$(dirname "$0")"
LOG=/tmp/top1_scheduler.log
free_gpu(){
  local apps mem
  apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
         | grep -viE 'gnome-remote|xorg|Xwayland' | grep -v '^[[:space:]]*$')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ -z "$apps" ] && [ -n "$mem" ] && [ "$mem" -lt 2500 ]
}
echo "$(date '+%F %T') waiting for free GPU" >> "$LOG"
streak=0
while true; do
  if free_gpu; then streak=$((streak+1)); [ $streak -ge 2 ] && break; else streak=0; fi
  sleep 300
done
echo "$(date '+%F %T') GPU free, starting top1 confirmation" >> "$LOG"
bash ./run_top1_confirm.sh
echo "$(date '+%F %T') top1 confirmation done" >> "$LOG"
