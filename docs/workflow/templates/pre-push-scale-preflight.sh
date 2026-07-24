#!/usr/bin/env sh
set -eu

if command -v scale >/dev/null 2>&1; then
  scale preflight --service all
else
  npx @hongmaple0820/scale-engine@latest preflight --service all
fi
