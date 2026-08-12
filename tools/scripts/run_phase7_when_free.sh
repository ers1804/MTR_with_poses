#!/bin/bash
# Waits until no foreign process holds the GPU, then runs phase7 (2s horizon).
set -u
cd "$(dirname "$0")"
LOG=/tmp/phase7_scheduler.log
MEM_THRESH=2500
free_gpu(){
  local apps mem
  apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
         | grep -viE 'gnome-remote|xorg|Xwayland' | grep -v '^[[:space:]]*$')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ -z "$apps" ] && [ -n "$mem" ] && [ "$mem" -lt "$MEM_THRESH" ]
}
echo "$(date '+%F %T') waiting for free GPU" >> "$LOG"
streak=0
while true; do
  if free_gpu; then streak=$((streak+1)); [ $streak -ge 2 ] && break
  else streak=0; fi
  sleep 300
done
echo "$(date '+%F %T') GPU free -> launching phase7" >> "$LOG"
bash ./run_followup_phase7.sh
echo "$(date '+%F %T') phase7 done" >> "$LOG"
