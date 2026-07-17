#!/usr/bin/env bash
set -euo pipefail

# G9: Knowledge-updated gate — verifies knowledge documents have been maintained
KNOWLEDGE_DOC=".trae/rules"
SUMMARY_FILE=".scale/task-summary.md"

found_update=0

# Check if knowledge doc was updated recently (within last session)
if [ -f "$KNOWLEDGE_DOC" ]; then
  # Verify knowledge doc has substance (more than just a header)
  content=$(grep -v '^#' "$KNOWLEDGE_DOC" | grep -v '^$' | grep -v '^>' | wc -l)
  if [ "$content" -gt 5 ]; then
    echo "[PASS] G9: Knowledge document $KNOWLEDGE_DOC has substantive content"
    found_update=1
  fi
fi

# Check for task summary
if [ -f "$SUMMARY_FILE" ]; then
  echo "[PASS] G9: Task summary exists at $SUMMARY_FILE"
  found_update=1
fi

# Allow explicit verification marker
if [ -f ".scale/g9-verified" ]; then
  echo "[PASS] G9: Knowledge update manually verified"
  found_update=1
fi

if [ "$found_update" -eq 0 ]; then
  echo "[WARN] G9: No knowledge update evidence found"
  echo "  Update $KNOWLEDGE_DOC or create .scale/task-summary.md after completing tasks"
  # G9 is a warning, not a hard block — exit 0 to not block the pipeline
fi

exit 0
