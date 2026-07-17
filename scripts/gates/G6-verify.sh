#!/usr/bin/env bash
set -euo pipefail

source "scripts/lib/project-config.sh"
cmd="$(command_for_gate "typecheck")"

if [ -z "$cmd" ] || [ "$cmd" = "N/A" ]; then
  echo "[SKIP] G6 Type verification: no configured command"
  exit 0
fi

# Skip echo placeholder commands (e.g. 'echo "[TYPECHECK] 不适用"')
if [[ "$cmd" == echo*"["*"]"* ]]; then
  echo "[SKIP] G6 Type verification: placeholder command (not applicable for this stack)"
  exit 0
fi

echo "[RUN] G6 Type verification: $cmd"
bash -lc "$cmd"
