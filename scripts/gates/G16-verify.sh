#!/usr/bin/env bash
set -euo pipefail

# G16: Commit Discipline — aligned with scale-engine docs/workflow/GATES_AND_SCORE.md
# Blocking when: >25 uncommitted files OR >180min since last commit
#                  OR staged file >1MB OR whitespace errors in diff

if ! command -v git >/dev/null 2>&1; then
  echo "[SKIP] G16: git not installed"
  exit 0
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[SKIP] G16: not a git work tree"
  exit 0
fi

failures=0

# ── Check 1: Uncommitted file count ────────────────────────────────────
UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
if [ "$UNCOMMITTED" -gt 25 ]; then
  echo "[FAIL] G16: $UNCOMMITTED uncommitted files (block threshold: 25)"
  echo "  Commit or stash before proceeding. Large uncommitted state risks lost work."
  failures=$((failures + 1))
elif [ "$UNCOMMITTED" -gt 10 ]; then
  echo "[WARN] G16: $UNCOMMITTED uncommitted files (warn threshold: 10)"
else
  echo "[PASS] G16: $UNCOMMITTED uncommitted files"
fi

# ── Check 2: Time since last commit ────────────────────────────────────
LAST_COMMIT_EPOCH=$(git log -1 --format=%ct 2>/dev/null || echo 0)
if [ "$LAST_COMMIT_EPOCH" -gt 0 ]; then
  NOW_EPOCH=$(date +%s)
  MINUTES_SINCE=$(( (NOW_EPOCH - LAST_COMMIT_EPOCH) / 60 ))
  if [ "$MINUTES_SINCE" -gt 180 ]; then
    echo "[FAIL] G16: ${MINUTES_SINCE}min since last commit (block threshold: 180min)"
    failures=$((failures + 1))
  elif [ "$MINUTES_SINCE" -gt 60 ]; then
    echo "[WARN] G16: ${MINUTES_SINCE}min since last commit (warn threshold: 60min)"
  else
    echo "[PASS] G16: ${MINUTES_SINCE}min since last commit"
  fi
else
  echo "[WARN] G16: no commits yet — skipping time-since-commit check"
fi

# ── Check 3: Staged files >1MB ─────────────────────────────────────────
LARGE_STAGED=$(git diff --cached --name-only 2>/dev/null | while read -r f; do
  [ -f "$f" ] || continue
  size=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f" 2>/dev/null || echo 0)
  if [ "$size" -gt 1048576 ]; then echo "$f ($size bytes)"; fi
done | wc -l | tr -d ' ')

if [ "$LARGE_STAGED" -gt 0 ]; then
  echo "[FAIL] G16: $LARGE_STAGED staged file(s) >1MB — use git-lfs or exclude"
  failures=$((failures + 1))
else
  echo "[PASS] G16: no staged files >1MB"
fi

# ── Check 4: Whitespace errors ─────────────────────────────────────────
if git diff --cached --check 2>/dev/null | grep -q .; then
  echo "[FAIL] G16: whitespace errors in staged diff:"
  git diff --cached --check 2>/dev/null | head -5
  failures=$((failures + 1))
else
  echo "[PASS] G16: no whitespace errors"
fi

[ "$failures" -gt 0 ] && exit 1
exit 0
