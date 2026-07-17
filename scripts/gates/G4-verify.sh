#!/usr/bin/env bash
set -euo pipefail

source "scripts/lib/project-config.sh"
cmd="$(command_for_gate "lint")"

if [ -z "$cmd" ] || [ "$cmd" = "N/A" ]; then
  echo "[SKIP] G4 Lint verification: no configured command"
  exit 0
fi

# Skip echo placeholder commands (e.g. 'echo "[TYPECHECK] 不适用"')
if [[ "$cmd" == echo*"["*"]"* ]]; then
  echo "[SKIP] G4 Lint verification: placeholder command (not applicable for this stack)"
  exit 0
fi

echo "[RUN] G4 Lint verification: $cmd"
bash -lc "$cmd"
