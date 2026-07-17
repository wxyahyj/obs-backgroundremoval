#!/usr/bin/env bash
set -euo pipefail

# G1: Exploration gate — aligned with scale-engine docs/workflow/TEMPLATE_GUIDE.md §4
# Requires: explore.md with ≥3 files read AND a Main Contradiction section.
# Reference: scale-engine/scripts/gates/G1-verify.sh
#
# Lookup order (first match wins):
#   1. .scale/explore.md
#   2. docs/worklog/tasks/*/explore.md  (most recent)
#   3. explore.md in current directory
#   4. .scale/g1-verified  (manual override — record reason inside)

# Manual override path
if [ -f ".scale/g1-verified" ]; then
  echo "[PASS] G1: Exploration manually verified (.scale/g1-verified present)"
  echo "  Override reason:"
  sed 's/^/    /' .scale/g1-verified 2>/dev/null || true
  exit 0
fi

# Locate explore.md
EXPLORE_FILE=""
for candidate in \
  ".scale/explore.md" \
  "$(ls -t docs/worklog/tasks/*/explore.md 2>/dev/null | head -1)" \
  "explore.md"; do
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    EXPLORE_FILE="$candidate"
    break
  fi
done

if [ -z "$EXPLORE_FILE" ]; then
  echo "[FAIL] G1: No explore.md found."
  echo "  Looked in: .scale/explore.md, docs/worklog/tasks/*/explore.md, ./explore.md"
  echo "  Create one using docs/workflow/templates/explore.md as the starting point."
  echo "  Or run: echo 'manual override reason' > .scale/g1-verified"
  exit 1
fi

echo "[INFO] G1: Checking $EXPLORE_FILE"

# ── Check 1: ≥3 files read ────────────────────────────────────────────
# Count non-comment, non-empty lines that look like file paths under
# ## Files Read section. Acceptable forms: 'src/foo.ts', './src/foo.ts',
# '/abs/path/foo.ts', '- src/foo.ts'.
FILES_SECTION=$(awk '/^## Files Read/{flag=1; next} /^## /{flag=0} flag' "$EXPLORE_FILE" 2>/dev/null || true)

if [ -z "$FILES_SECTION" ]; then
  echo "[FAIL] G1: Missing '## Files Read' section in $EXPLORE_FILE"
  echo "  TEMPLATE_GUIDE.md §4 requires explore.md to list ≥3 files read."
  exit 1
fi

# Strip markdown list markers, comments, empty lines; keep plausible paths
FILE_COUNT=$(echo "$FILES_SECTION" \
  | grep -v '^<' \
  | grep -v '^#' \
  | grep -v '^[[:space:]]*$' \
  | sed 's/^[[:space:]]*[-*][[:space:]]*//' \
  | grep -E '(^|[/.])(src|lib|app|tests|test|docs|scripts|packages)/|[.][tj]sx?$|[.]py$|[.]go$|[.]rs$|[.]java$|[.]cs$|[.]cpp$|[.]cc$|[.]md$' \
  | wc -l)

if [ "$FILE_COUNT" -lt 3 ]; then
  echo "[FAIL] G1: Only $FILE_COUNT file(s) listed in '## Files Read' (need ≥3)."
  echo "  TEMPLATE_GUIDE.md §4: 'explore at least 3 files and record main contradiction'"
  echo "  Hint: list the actual source files you read, e.g.:"
  echo "    - src/lib/config-generator.ts"
  echo "    - src/lib/config/workflow-adapters.ts"
  echo "    - docs/workflow/TEMPLATE_GUIDE.md"
  exit 1
fi

echo "[PASS] G1: $FILE_COUNT files recorded in '## Files Read'"

# ── Check 2: Main Contradiction section is non-trivial ────────────────
CONTRADICTION=$(awk '/^## Main Contradiction/{flag=1; next} /^## /{flag=0} flag' "$EXPLORE_FILE" 2>/dev/null || true)

if [ -z "$CONTRADICTION" ]; then
  echo "[FAIL] G1: Missing '## Main Contradiction' section in $EXPLORE_FILE"
  echo "  Record the main contradiction / tension / open question you found while reading."
  exit 1
fi

# Strip TBD / placeholders / comments / blank lines
CONTRADICTION_BODY=$(echo "$CONTRADICTION" \
  | grep -v '^[[:space:]]*$' \
  | grep -v '^<' \
  | grep -v -i '^[[:space:]]*-?[[:space:]]*TBD' \
  | wc -l)

if [ "$CONTRADICTION_BODY" -lt 1 ]; then
  echo "[FAIL] G1: '## Main Contradiction' section is empty or still TBD."
  echo "  Replace the TBD placeholder with the actual contradiction you discovered."
  exit 1
fi

echo "[PASS] G1: Main Contradiction section populated"
exit 0
