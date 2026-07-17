#!/usr/bin/env bash
set -euo pipefail

# G2: Planning gate — aligned with scale-engine docs/workflow/TEMPLATE_GUIDE.md §4
# Enforces 5 keyword groups + 3 companion artifacts (reality-check / runtime / resource-cleanup).
# Reference: scale-engine/scripts/gates/G2-verify.sh
#
# Keyword requirements (case-insensitive):
#   1. Scope        : scope | boundary | boundaries | limit | non-goal
#   2. Exception    : exception | error | fail | failure | rollback  (≥3 occurrences total)
#   3. Rollback     : rollback | recovery | disable | fallback
#   4. Acceptance   : acceptance | success criteria | definition of done
#   5. L/CRITICAL   : human confirmation | review before execution
#                     (only enforced when .scale/task-level is L or CRITICAL)

# Locate plan.md
PLAN_FILE=""
for candidate in \
  ".scale/plan.md" \
  "$(ls -t docs/worklog/tasks/*/plan.md 2>/dev/null | head -1)" \
  "plan.md"; do
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    PLAN_FILE="$candidate"
    break
  fi
done

if [ -z "$PLAN_FILE" ]; then
  echo "[FAIL] G2: No plan.md found."
  echo "  Looked in: .scale/plan.md, docs/worklog/tasks/*/plan.md, ./plan.md"
  echo "  Create one using docs/workflow/templates/plan.md as the starting point."
  exit 1
fi

echo "[INFO] G2: Checking $PLAN_FILE"
failures=0
missing=()

# ── Check 1: Scope / Boundary keyword ─────────────────────────────────
if ! grep -qiE 'scope|boundary|boundaries|limit|non-goal' "$PLAN_FILE"; then
  missing+=("scope / boundary / limit / non-goal")
fi

# ── Check 2: Exception coverage (≥3 total occurrences) ────────────────
EXC_COUNT=$(grep -ioE 'exception|error|fail|failure|rollback' "$PLAN_FILE" | wc -l)
if [ "$EXC_COUNT" -lt 3 ]; then
  missing+=("exception/error/fail/failure/rollback (need ≥3, found $EXC_COUNT)")
fi

# ── Check 3: Rollback strategy keyword ────────────────────────────────
if ! grep -qiE 'rollback|recovery|disable|fallback' "$PLAN_FILE"; then
  missing+=("rollback / recovery / disable / fallback")
fi

# ── Check 4: Acceptance criteria keyword ──────────────────────────────
if ! grep -qiE 'acceptance|success criteria|definition of done' "$PLAN_FILE"; then
  missing+=("acceptance / success criteria / definition of done")
fi

# ── Check 5: L/CRITICAL human confirmation ────────────────────────────
TASK_LEVEL=""
if [ -f ".scale/task-level" ]; then
  TASK_LEVEL=$(cat .scale/task-level | tr '[:lower:]' '[:upper:]' | tr -d '[:space:]')
fi
if [ "$TASK_LEVEL" = "L" ] || [ "$TASK_LEVEL" = "CRITICAL" ]; then
  if ! grep -qiE 'human confirmation|review before execution' "$PLAN_FILE"; then
    missing+=("human confirmation / review before execution (required for $TASK_LEVEL)")
  fi
fi

# ── Check 6: Companion artifacts (TEMPLATE_GUIDE.md §4 G2 requirements) ──
# reality-check.md, runtime.md, resource-cleanup.md must exist in the same dir as plan.md
PLAN_DIR=$(dirname "$PLAN_FILE")
for companion in reality-check.md runtime.md resource-cleanup.md; do
  if [ ! -f "$PLAN_DIR/$companion" ]; then
    missing+=("$companion (expected alongside plan.md in $PLAN_DIR)")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  echo "[FAIL] G2: Plan exists but missing required content:"
  for item in "${missing[@]}"; do
    echo "  - $item"
  done
  echo ""
  echo "  Reference: docs/workflow/TEMPLATE_GUIDE.md §4 G2 keyword list"
  echo "  Template:  docs/workflow/templates/plan.md"
  exit 1
fi

echo "[PASS] G2: Plan verified — scope / exception($EXC_COUNT) / rollback / acceptance"
[ -n "$TASK_LEVEL" ] && echo "[PASS] G2: $TASK_LEVEL-level human confirmation present"
echo "[PASS] G2: Companion artifacts (reality-check / runtime / resource-cleanup) present"
exit 0
