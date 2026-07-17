#!/usr/bin/env bash
set -euo pipefail

# G17: Documentation Hygiene — aligned with scale-engine docs/workflow/GATES_AND_SCORE.md
# Validates internal markdown links in changed *.md files.
# Reference: scale-engine/scripts/workflow/docs-health.mjs

if ! command -v git >/dev/null 2>&1; then
  echo "[SKIP] G17: git not installed"
  exit 0
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[SKIP] G17: not a git work tree"
  exit 0
fi

# Gather changed markdown files (staged + unstaged + untracked)
CHANGED_MD=$(git diff --name-only --diff-filter=AM HEAD 2>/dev/null | grep -E '\.md$' || true)
CHANGED_MD="$CHANGED_MD$(git ls-files --others --exclude-standard 2>/dev/null | grep -E '\.md$' | sed 's/^/\n/')"
CHANGED_MD=$(echo "$CHANGED_MD" | grep -v '^[[:space:]]*$' | sort -u || true)

if [ -z "$CHANGED_MD" ]; then
  echo "[PASS] G17: no changed markdown files"
  exit 0
fi

failures=0
checked=0
for mdfile in $CHANGED_MD; do
  [ -f "$mdfile" ] || continue
  checked=$((checked + 1))
  dir=$(dirname "$mdfile")

  # Extract markdown links [text](relative-path) — skip http(s) URLs and anchors
  while IFS= read -r link; do
    target="$link"
    # Strip optional #anchor
    target="${target%%#*}"
    [ -z "$target" ] && continue
    # Skip absolute URLs
    case "$target" in
      http://*|https://*|mailto:*) continue ;;
    esac
    # Resolve relative to the markdown file's directory
    if [ ! -e "$dir/$target" ]; then
      echo "[FAIL] G17: broken link in $mdfile → $target"
      failures=$((failures + 1))
    fi
  done < <(grep -oE '\]\([^)]+\)' "$mdfile" 2>/dev/null | sed 's/^](//; s/)$//')
done

if [ "$failures" -gt 0 ]; then
  echo "[FAIL] G17: $failures broken internal link(s) across $checked changed markdown file(s)"
  exit 1
fi

echo "[PASS] G17: $checked changed markdown file(s) — all internal links valid"
exit 0
