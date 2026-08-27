#!/usr/bin/env bash
# Launch `mlx_lm.server` under monitor.py so memory is tracked for its whole
# lifetime, with a chart saved when the server stops.
#
# Usage: pass mlx_lm.server's own flags straight through, e.g.:
#   ./serve.sh --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --port 8080
#
# Override the monitor's sampling interval or memory ceiling with env vars:
#   MONITOR_INTERVAL=0.5 MONITOR_LIMIT_GB=32 ./serve.sh --model ...
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INTERVAL="${MONITOR_INTERVAL:-1.0}"
LIMIT_GB="${MONITOR_LIMIT_GB:-32}"

exec python3 "$DIR/monitor.py" --interval "$INTERVAL" --limit-gb "$LIMIT_GB" --plot -- \
    python3 -m mlx_lm.server "$@"
