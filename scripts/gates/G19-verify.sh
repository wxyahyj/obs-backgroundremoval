#!/usr/bin/env bash
set -euo pipefail

# G19: Code Review — aligned with scale-engine docs/workflow/GATES_AND_SCORE.md
# Blocking for L/CRITICAL tasks; advisory for S/M.
# Evidence: .agent/state/review-*.json OR review.md alongside plan.md

TASK_LEVEL=""
if [ -f ".scale/task-level" ]; then
  TASK_LEVEL=$(cat .scale/task-level | tr '[:lower:]' '[:upper:]' | tr -d '[:space:]')
fi

# Locate review.md
REVIEW_FILE=""
for candidate in \
  ".scale/review.md" \
  "$(ls -t docs/worklog/tasks/*/review.md 2>/dev/null | head -1)" \
  "review.md"; do
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    REVIEW_FILE="$candidate"
    break
  fi
done

# Locate structured review record (scale-engine style)
REVIEW_JSON=$(ls -t .agent/state/review-*.json 2>/dev/null | head -1 || true)

# Advisory mode for S/M (when no task level or S/M)
if [ "$TASK_LEVEL" != "L" ] && [ "$TASK_LEVEL" != "CRITICAL" ]; then
  if [ -n "$REVIEW_FILE" ] || [ -n "$REVIEW_JSON" ]; then
    echo "[PASS] G19: review evidence present (advisory for $TASK_LEVEL/S/M)"
  else
    echo "[INFO] G19: no review evidence (advisory for $TASK_LEVEL/S/M — skip)"
  fi
  exit 0
fi

# Blocking mode for L/CRITICAL
if [ -z "$REVIEW_FILE" ] && [ -z "$REVIEW_JSON" ]; then
  echo "[FAIL] G19: $TASK_LEVEL-level task requires code review evidence."
  echo "  Create review.md (template: docs/workflow/templates/review.md)"
  echo "  Or run: scale review --task <TASK_ID>"
  exit 1
fi

# If review.md exists, check it has non-TBD Findings + Residual Risk
if [ -n "$REVIEW_FILE" ]; then
  echo "[INFO] G19: Checking $REVIEW_FILE"
  FINDINGS=$(awk '/^## Findings/{flag=1; next} /^## /{flag=0} flag' "$REVIEW_FILE" 2>/dev/null || true)
  FINDINGS_BODY=$(echo "$FINDINGS" \
    | grep -v '^[[:space:]]*$' \
    | grep -v '^<' \
    | grep -v -i '^[[:space:]]*-?[[:space:]]*TBD' \
    | wc -l)
  if [ "$FINDINGS_BODY" -lt 1 ]; then
    echo "[FAIL] G19: '## Findings' section in $REVIEW_FILE is empty or TBD"
    exit 1
  fi

  RESIDUAL=$(awk '/^## Residual Risk/{flag=1; next} /^## /{flag=0} flag' "$REVIEW_FILE" 2>/dev/null || true)
  RESIDUAL_BODY=$(echo "$RESIDUAL" \
    | grep -v '^[[:space:]]*$' \
    | grep -v '^<' \
    | wc -l)
  if [ "$RESIDUAL_BODY" -lt 1 ]; then
    echo "[FAIL] G19: '## Residual Risk' section in $REVIEW_FILE is empty"
    exit 1
  fi
  echo "[PASS] G19: review.md verified — Findings + Residual Risk populated"
fi

# If structured JSON exists, check all findings are resolved
if [ -n "$REVIEW_JSON" ]; then
  OPEN_FINDINGS=$(grep -oE '"status"[[:space:]]*:[[:space:]]*"open"' "$REVIEW_JSON" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$OPEN_FINDINGS" -gt 0 ]; then
    echo "[FAIL] G19: $OPEN_FINDINGS unresolved finding(s) in $REVIEW_JSON"
    exit 1
  fi
  echo "[PASS] G19: all review findings resolved in $REVIEW_JSON"
fi

exit 0
