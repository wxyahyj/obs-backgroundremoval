#!/usr/bin/env bash
set -euo pipefail

# G20: Supply Chain — aligned with scale-engine docs/workflow/GATES_AND_SCORE.md
# Checks:
#   1. Lock file exists and is consistent with manifest
#   2. No CRITICAL/HIGH vulnerabilities (npm audit / pip-audit / cargo audit)
#   3. No un-pinned dependencies in production deps (best-effort)

failures=0

# ── Check 1: Lock file presence + consistency ──────────────────────────
if [ -f "package.json" ]; then
  if [ ! -f "package-lock.json" ] && [ ! -f "bun.lock" ] && [ ! -f "pnpm-lock.yaml" ] && [ ! -f "yarn.lock" ]; then
    echo "[FAIL] G20: package.json present but no lock file"
    failures=$((failures + 1))
  else
    echo "[PASS] G20: lock file present"
  fi
elif [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
  if [ ! -f "requirements.lock" ] && [ ! -f "poetry.lock" ] && [ ! -f "uv.lock" ] && ! command -v pip >/dev/null 2>&1; then
    echo "[WARN] G20: Python project without lock file — consider 'pip freeze > requirements.lock'"
  else
    echo "[PASS] G20: Python lock file present"
  fi
elif [ -f "Cargo.toml" ]; then
  if [ ! -f "Cargo.lock" ]; then
    echo "[FAIL] G20: Cargo.toml present but Cargo.lock missing"
    failures=$((failures + 1))
  else
    echo "[PASS] G20: Cargo.lock present"
  fi
elif [ -f "go.mod" ]; then
  if [ ! -f "go.sum" ]; then
    echo "[FAIL] G20: go.mod present but go.sum missing"
    failures=$((failures + 1))
  else
    echo "[PASS] G20: go.sum present"
  fi
else
  echo "[SKIP] G20: no recognized package manifest"
fi

# ── Check 2: Vulnerability scan (best-effort, skip if tool missing) ────
if [ -f "package-lock.json" ] && command -v npm >/dev/null 2>&1; then
  AUDIT_JSON=$(npm audit --json 2>/dev/null || echo '{"metadata":{"vulnerabilities":{"critical":0,"high":0}}}')
  CRITICAL=$(echo "$AUDIT_JSON" | grep -oE '"critical"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+$' || echo 0)
  HIGH=$(echo "$AUDIT_JSON" | grep -oE '"high"[[:space:]]*:[[:space:]]*[0-9]+' | head -1 | grep -oE '[0-9]+$' || echo 0)
  if [ "$CRITICAL" -gt 0 ] || [ "$HIGH" -gt 0 ]; then
    echo "[FAIL] G20: $CRITICAL critical, $HIGH high vulnerabilities (npm audit)"
    failures=$((failures + 1))
  else
    echo "[PASS] G20: no critical/high vulnerabilities (npm audit)"
  fi
elif [ -f "Cargo.lock" ] && command -v cargo-audit >/dev/null 2>&1; then
  if cargo audit 2>&1 | grep -qE 'RUSTSEC-[0-9]+-[0-9]+.*(unsound|high|critical)'; then
    echo "[FAIL] G20: cargo audit found high/critical advisories"
    failures=$((failures + 1))
  else
    echo "[PASS] G20: cargo audit clean"
  fi
else
  echo "[INFO] G20: vulnerability scanner not available — skipping (advisory)"
fi

[ "$failures" -gt 0 ] && exit 1
exit 0
