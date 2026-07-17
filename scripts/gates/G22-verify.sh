#!/usr/bin/env bash
set -euo pipefail

# G22: Session Health — advisory only (non-blocking)
# Checks for stale worktrees, abandoned session state, leaked temp files.

failures=0

# ── Check 1: stale worktrees (git worktree list with stale entries) ───
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  STALE=$(git worktree list 2>/dev/null | grep -c 'prunable' || echo 0)
  if [ "$STALE" -gt 0 ]; then
    echo "[WARN] G22: $STALE prunable git worktree(s) — run 'git worktree prune'"
  else
    echo "[PASS] G22: no prunable worktrees"
  fi
fi

# ── Check 2: leaked .agent/logs/ temp files older than 7 days ──────────
if [ -d ".agent/logs" ]; then
  OLD_LOGS=$(find .agent/logs -type f -mtime +7 2>/dev/null | wc -l | tr -d ' ')
  if [ "$OLD_LOGS" -gt 50 ]; then
    echo "[WARN] G22: $OLD_LOGS log files older than 7 days in .agent/logs — consider cleanup"
  else
    echo "[PASS] G22: .agent/logs cleanup healthy ($OLD_LOGS old files)"
  fi
fi

# ── Check 3: orphaned session state ────────────────────────────────────
SESSION_STATE=".agent/state/session.json"
if [ -f "$SESSION_STATE" ]; then
  STATE_AGE_MIN=$(( ($(date +%s) - $(stat -c %Y "$SESSION_STATE" 2>/dev/null || stat -f %m "$SESSION_STATE" 2>/dev/null || echo 0)) / 60 ))
  if [ "$STATE_AGE_MIN" -gt 1440 ]; then
    echo "[WARN] G22: session.json last updated ${STATE_AGE_MIN}min ago (>24h) — may be stale"
  else
    echo "[PASS] G22: session.json fresh (${STATE_AGE_MIN}min old)"
  fi
fi

# Advisory — always exit 0
exit 0
