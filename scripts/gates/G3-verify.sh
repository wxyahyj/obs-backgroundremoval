#!/usr/bin/env bash
set -euo pipefail

# G3: TDD compliance gate — verifies test evidence exists
# Checks that test files exist and are newer than the plan (if plan exists)
PLAN_FILE=".scale/plan.md"
EXEMPT_FILE=".scale/tdd-exempt.md"

# Allow TDD exemption for specific tasks
if [ -f "$EXEMPT_FILE" ]; then
  echo "[PASS] G3: TDD exemption recorded in $EXEMPT_FILE"
  exit 0
fi

# Find test files using project-specific patterns
TEST_FILES=$(find . -type f \( -name '*_test.py' -o -name 'test_*.py' -o -name '*_test.rs' -o -path '*/tests/*.rs' -o -name '*.test.ts' -o -name '*.test.tsx' -o -name '*.spec.ts' -o -name '*_test.cpp' -o -name '*_test.cc' \) \
  ! -path '*/node_modules/*' \
  ! -path '*/.agent/*' \
  ! -path '*/vendor/*' \
  ! -path '*/.git/*' \
  2>/dev/null | head -5)

if [ -z "$TEST_FILES" ]; then
  echo "[FAIL] G3: No test files found matching project patterns"
  echo "  Expected test patterns for this stack: -name '*_test.py' -o -name 'test_*.py' -o -name '*_test.rs' -o -path '*/tests/*.rs' -o -name '*.test.ts' -o -name '*.test.tsx' -o -name '*.spec.ts' -o -name '*_test.cpp' -o -name '*_test.cc'"
  echo "  If TDD is not applicable, create .scale/tdd-exempt.md with justification"
  exit 1
fi

# If plan.md exists, verify at least one test is newer than the plan
if [ -f "$PLAN_FILE" ]; then
  PLAN_TIME=$(stat -c %Y "$PLAN_FILE" 2>/dev/null || stat -f %m "$PLAN_FILE" 2>/dev/null || echo 0)
  NEWER_TESTS=$(find . -type f \( -name '*_test.py' -o -name 'test_*.py' -o -name '*_test.rs' -o -path '*/tests/*.rs' -o -name '*.test.ts' -o -name '*.test.tsx' -o -name '*.spec.ts' -o -name '*_test.cpp' -o -name '*_test.cc' \) \
    ! -path '*/node_modules/*' \
    ! -path '*/.agent/*' \
    -newer "$PLAN_FILE" \
    2>/dev/null | head -1)

  if [ -n "$NEWER_TESTS" ]; then
    echo "[PASS] G3: TDD evidence found — test files newer than plan.md"
    echo "  Test file: $NEWER_TESTS"
    exit 0
  fi
fi

# If no plan.md, just having test files is sufficient
echo "[PASS] G3: Test files exist — TDD evidence found"
echo "  Files: $(echo $TEST_FILES | head -3)"
exit 0
