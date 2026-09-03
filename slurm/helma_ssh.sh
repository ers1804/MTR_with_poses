#!/bin/bash
# The ONLY way automation should talk to Helma.
#
# Guarantees:
#  1. Connection multiplexing — one master connection is reused for 8 h, so repeated status
#     checks cost no new TCP/auth handshakes on Helma OR on the csnhr jump host.
#  2. Rate limiting — a NEW master connection is attempted at most once per MIN_INTERVAL
#     seconds, with a lock so parallel watchers cannot stampede.
#  3. Backoff memory — after a refusal, the next attempt waits progressively longer.
#
# Background: on 2026-08-24 several watchers each ran `ssh helma` every 45-60 s. Because
# `Host helma` uses ProxyJump, that was 2 connections per poll, and the account was
# rate-limited into "kex_exchange_identification: Connection closed by remote host".
# Running jobs were unaffected, but all visibility was lost for ~an hour.
#
# Usage:  slurm/helma_ssh.sh 'squeue -u $USER'
#         slurm/helma_ssh.sh --status        # cheap cached status, safe to call often
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$HERE/ssh_config"
STATE="${HOME}/.ssh/cm"
STAMP="$STATE/.helma_last_attempt"
FAILS="$STATE/.helma_fail_count"
LOCK="$STATE/.helma_lock"
MIN_INTERVAL=${HELMA_MIN_INTERVAL:-120}
mkdir -p "$STATE"; chmod 700 "$STATE"

master_alive() {
  # -O check can HANG on a stale master (socket file outlives the connection after
  # network changes). Bound it, and tear down the corpse so reconnection can work.
  if timeout 10 ssh -F "$CFG" -O check helma >/dev/null 2>&1; then return 0; fi
  timeout 10 ssh -F "$CFG" -O exit helma >/dev/null 2>&1
  timeout 10 ssh -F "$CFG" -O exit csnhr.nhr.fau.de >/dev/null 2>&1   # the JUMP master hangs too
  rm -f "$STATE"/v103fe12@* 2>/dev/null
  return 1
}

ensure_master() {
  master_alive && return 0
  exec 9>"$LOCK"; flock 9                     # one connector at a time
  master_alive && return 0                    # another watcher may have just opened it
  local now last fails wait
  now=$(date +%s); last=$(cat "$STAMP" 2>/dev/null || echo 0)
  fails=$(cat "$FAILS" 2>/dev/null || echo 0)
  wait=$(( MIN_INTERVAL * (fails + 1) - (now - last) ))
  [ "$wait" -gt 0 ] && sleep "$wait"
  date +%s > "$STAMP"
  if ssh -F "$CFG" -MNf -o ConnectTimeout=30 helma 2>/dev/null; then
    echo 0 > "$FAILS"; return 0
  fi
  echo $(( fails < 8 ? fails + 1 : 8 )) > "$FAILS"
  echo "helma_ssh: master connection refused (backoff level $(cat "$FAILS"))" >&2
  return 1
}

if [ "${1:-}" = "--status" ]; then
  ensure_master || exit 1
  ssh -F "$CFG" helma 'squeue -u $USER -h -o "%.12i %.16j %.8T %.8M" | head -20; echo "--- jobs above ---"'
  exit $?
fi

ensure_master || exit 1
exec ssh -F "$CFG" helma "$@"
