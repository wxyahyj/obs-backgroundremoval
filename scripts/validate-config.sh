#!/usr/bin/env bash
set -euo pipefail

# Enhanced config validator — checks existence, JSON validity, and consistency
failures=0
warnings=0

echo "=== SCALE OS Configuration Validator ==="
echo ""

# 1. Check required files exist
echo "[CHECK] File existence..."
required=(".agent/project.json" ".agent/report.json" ".scale/workflow.json" ".scale/quality-contract.json" ".scale/skills-registry.json")

for file in "${required[@]}"; do
  if [ ! -f "$file" ]; then
    echo "[FAIL] missing $file"
    failures=$((failures + 1))
  else
    echo "[OK] $file exists"
  fi
done

# Optional files (warn only)
optional_files=("SCALE-REPORT.md" "scripts/gates/all.sh" "scripts/workflow/verify.sh" "scripts/tests/run.sh")
for file in "${optional_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "[WARN] missing optional $file"
    warnings=$((warnings + 1))
  fi
done

echo ""

# 2. Validate JSON syntax
echo "[CHECK] JSON validity..."
for file in .agent/project.json .agent/report.json .scale/workflow.json .scale/quality-contract.json .scale/skills-registry.json; do
  if [ -f "$file" ]; then
    if jq empty "$file" 2>/dev/null; then
      echo "[OK] $file is valid JSON"
    else
      echo "[FAIL] $file has invalid JSON syntax"
      failures=$((failures + 1))
    fi
  fi
done

echo ""

# 3. Stack consistency check
echo "[CHECK] Stack consistency..."
if [ -f ".agent/project.json" ]; then
  STACK=$(jq -r '.stack // "unknown"' ".agent/project.json" 2>/dev/null || echo "unknown")
  DEV_CMD=$(jq -r '.commands.dev // ""' ".agent/project.json" 2>/dev/null || echo "")

  # Check for known inconsistencies
  if [ "$STACK" = "go" ] && echo "$DEV_CMD" | grep -qi "python\|uvicorn\|pytest"; then
    echo "[FAIL] Stack is 'go' but dev command uses Python: $DEV_CMD"
    failures=$((failures + 1))
  elif [ "$STACK" = "python" ] && echo "$DEV_CMD" | grep -qi "pnpm\|npm\|node"; then
    echo "[FAIL] Stack is 'python' but dev command uses Node: $DEV_CMD"
    failures=$((failures + 1))
  elif [ "$STACK" = "rust" ] && echo "$DEV_CMD" | grep -qi "python\|uvicorn\|pnpm"; then
    echo "[FAIL] Stack is 'rust' but dev command uses non-Rust tools: $DEV_CMD"
    failures=$((failures + 1))
  elif [ "$STACK" = "java" ] && echo "$DEV_CMD" | grep -qi "pnpm\|npm\|cargo\|go run"; then
    echo "[FAIL] Stack is 'java' but dev command uses non-Java tools: $DEV_CMD"
    failures=$((failures + 1))
  elif [ "$STACK" = "cpp" ] && echo "$DEV_CMD" | grep -qi "python\|uvicorn\|pnpm\|npm\|cargo\|go run"; then
    echo "[FAIL] Stack is 'cpp' but dev command uses non-C++ tools: $DEV_CMD"
    failures=$((failures + 1))
  else
    echo "[OK] Stack '$STACK' is consistent with dev command"
  fi

  # Check lint command matches stack
  LINT_CMD=$(jq -r '.commands.lint // ""' ".agent/project.json" 2>/dev/null || echo "")
  if [ -n "$LINT_CMD" ] && [ "$LINT_CMD" != "echo"* ]; then
    echo "[OK] Lint command configured: $LINT_CMD"
  fi
fi

echo ""

# 4. Summary
echo "=== Validation Summary ==="
if [ "$failures" -gt 0 ]; then
  echo "[FAIL] $failures failure(s), $warnings warning(s)"
  exit 1
fi

echo "[OK] Configuration valid ($warnings warning(s))"
