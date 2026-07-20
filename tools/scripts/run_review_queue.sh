#!/bin/bash
# Serial driver for the 2026-07-20 review experiment queue: 5 -> 5b -> 5c.
D="$(dirname "$0")"
bash "$D/run_multiseed_phase5.sh"
bash "$D/run_multiseed_phase5b.sh"
bash "$D/run_multiseed_phase5c.sh"
echo "=== review queue COMPLETE $(date) ===" >> /tmp/review_queue.log
