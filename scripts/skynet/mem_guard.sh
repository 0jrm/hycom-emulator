#!/bin/bash
# Run a command next to live training: kill its whole session if host memory in use (MemTotal - MemAvailable)
# passes HOST_GB (default 520) or the session's summed RSS passes RSS_GB (default 40). Logs the peak RSS at exit.
# usage: mem_guard.sh <log> <command> [args...]
# Detached: setsid nohup mem_guard.sh <log> <command> ... < /dev/null > <log>.guard 2>&1 &
set -u
LOG=$1; shift
HOST_GB=${HOST_GB:-520}; RSS_GB=${RSS_GB:-40}
setsid "$@" >>"$LOG" 2>&1 < /dev/null &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
  used=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print int((t-a)/1048576)}' /proc/meminfo)
  rss=$(ps -o rss= --sid "$pid" | awk '{s+=$1} END{print int(s/1024)}')
  [ "$rss" -gt "$peak" ] && peak=$rss
  if [ "$used" -gt "$HOST_GB" ] || [ "$rss" -gt $((RSS_GB * 1024)) ]; then
    kill -- -"$pid"
    echo "$(date -Is) mem_guard killed $* at host $used GB, rss $rss MB" >> "$LOG"
    exit 1
  fi
  sleep 5
done
wait "$pid"; code=$?
echo "$(date -Is) mem_guard exit $code, peak rss $peak MB" >> "$LOG"
exit $code
