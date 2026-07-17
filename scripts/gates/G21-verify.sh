#!/usr/bin/env bash
set -euo pipefail

# G21: Context Budget — advisory only (non-blocking)
# Reads .scale/context-budget.json (if present) and reports usage.

BUDGET_FILE=".scale/context-budget.json"
USAGE_FILE=".agent/state/context-usage.json"

if [ ! -f "$BUDGET_FILE" ]; then
  echo "[SKIP] G21: no .scale/context-budget.json — using default 2400 token budget"
  BUDGET=2400
else
  BUDGET=$(grep -oE '"budget"[[:space:]]*:[[:space:]]*[0-9]+' "$BUDGET_FILE" | head -1 | grep -oE '[0-9]+$' || echo 2400)
fi

USED=0
if [ -f "$USAGE_FILE" ]; then
  USED=$(grep -oE '"used"[[:space:]]*:[[:space:]]*[0-9]+' "$USAGE_FILE" | head -1 | grep -oE '[0-9]+$' || echo 0)
fi

PCT=0
[ "$BUDGET" -gt 0 ] && PCT=$(( (USED * 100) / BUDGET ))

echo "[INFO] G21: context budget $USED/$BUDGET tokens (${PCT}%)"
if [ "$PCT" -ge 100 ]; then
  echo "[WARN] G21: context budget exceeded — consider checkpointing and starting fresh session"
elif [ "$PCT" -ge 80 ]; then
  echo "[WARN] G21: approaching context budget limit — plan to wrap up soon"
fi

# Advisory — always exit 0
exit 0
