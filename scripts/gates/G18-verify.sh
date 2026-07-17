#!/usr/bin/env bash
set -euo pipefail

# G18: Runtime Evidence — aligned with scale-engine docs/workflow/GATES_AND_SCORE.md
# Task must have at least one runtime evidence record with exit code 0.
# Evidence store: .agent/state/runtime-evidence.json
# Fallback: verification.md must list ≥1 passed command.

EVIDENCE_FILE=".agent/state/runtime-evidence.json"
VERIFICATION_FILE=""
for candidate in \
  ".scale/verification.md" \
  "$(ls -t docs/worklog/tasks/*/verification.md 2>/dev/null | head -1)" \
  "verification.md"; do
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    VERIFICATION_FILE="$candidate"
    break
  fi
done

# ── Path 1: structured runtime evidence store ──────────────────────────
if [ -f "$EVIDENCE_FILE" ]; then
  PASS_COUNT=$(grep -oE '"exitCode"[[:space:]]*:[[:space:]]*0' "$EVIDENCE_FILE" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$PASS_COUNT" -gt 0 ]; then
    echo "[PASS] G18: $PASS_COUNT runtime evidence record(s) with exit code 0"
    exit 0
  else
    echo "[FAIL] G18: $EVIDENCE_FILE exists but no exit-code-0 records"
    exit 1
  fi
fi

# ── Path 2: fallback to verification.md ────────────────────────────────
if [ -z "$VERIFICATION_FILE" ]; then
  echo "[FAIL] G18: No runtime evidence found."
  echo "  Expected one of:"
  echo "    - $EVIDENCE_FILE (structured store, populated by 'scale runtime record')"
  echo "    - verification.md with at least one command marked Passed"
  echo "  Template: docs/workflow/templates/verification.md"
  exit 1
fi

# Look for "## Passed" section with non-TBD content
PASSED_SECTION=$(awk '/^## Passed/{flag=1; next} /^## /{flag=0} flag' "$VERIFICATION_FILE" 2>/dev/null || true)
PASSED_BODY=$(echo "$PASSED_SECTION" \
  | grep -v '^[[:space:]]*$' \
  | grep -v '^<' \
  | grep -v -i '^[[:space:]]*-?[[:space:]]*TBD' \
  | wc -l)

if [ "$PASSED_BODY" -lt 1 ]; then
  echo "[FAIL] G18: $VERIFICATION_FILE has empty '## Passed' section"
  echo "  Record at least one command that actually ran with exit code 0."
  exit 1
fi

echo "[PASS] G18: verification.md has $PASSED_BODY passed command(s) recorded"
exit 0
