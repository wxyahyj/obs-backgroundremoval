#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════
# SCALE Engine Gate Runner — supports G0–G22 + 4 profiles
# Reference: scale-engine/docs/workflow/GATES_AND_SCORE.md
# ═══════════════════════════════════════════════════════════
#
# This project's default profile: standard
# (selected in SCALE OS configurator → 偏好设置 → 质量门禁 Profile)
#
# Usage:
#   bash scripts/gates/all.sh                            # standard (project default)
#   bash scripts/gates/all.sh --profile fast-lane        # S-level: typo / comment / config
#   bash scripts/gates/all.sh --profile standard         # M-level: 2-5 files behavior change
#   bash scripts/gates/all.sh --profile full             # L-level: cross-module / refactor
#   bash scripts/gates/all.sh --profile comprehensive    # CRITICAL: auth / money / migration
#   bash scripts/gates/all.sh --dry-run                  # check which scripts exist, don't run
#   bash scripts/gates/all.sh --gate G2                  # run a single gate
#   bash scripts/gates/all.sh --gates G1,G2,G5           # run a subset of gates

DRY_RUN=0
PROFILE="standard"
SINGLE_GATE=""
SUBSET=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)     DRY_RUN=1; shift ;;
    --profile)     PROFILE="$2"; shift 2 ;;
    --gate)        SINGLE_GATE="$2"; shift 2 ;;
    --gates)       SUBSET="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]
  --profile <name>   fast-lane | standard | full | comprehensive (default: comprehensive)
  --gate <id>        run a single gate (e.g. G2)
  --gates <ids>      comma-separated subset (e.g. G1,G2,G5)
  --dry-run          check scripts exist, don't run
  -h, --help         show this help
EOF
      exit 0 ;;
    *) echo "[FAIL] unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ── Resolve gates to run ────────────────────────────────────────────────
if [ -n "$SINGLE_GATE" ]; then
  GATES=("$SINGLE_GATE")
elif [ -n "$SUBSET" ]; then
  IFS=',' read -ra GATES <<< "$SUBSET"
else
  case "$PROFILE" in
    fast-lane)       GATES=(G0 G4 G5) ;;
    standard)        GATES=(G2 G4 G5 G6 G16) ;;
    full)            GATES=(G1 G2 G3 G4 G5 G6 G16 G17 G18) ;;
    comprehensive)   GATES=(G1 G2 G3 G4 G5 G6 G7 G16 G17 G18 G19 G20) ;;
    *) echo "[FAIL] unknown profile: $PROFILE" >&2; exit 2 ;;
  esac
fi

echo "=== SCALE Engine Gate Runner (profile=$PROFILE, gates=${GATES[*]}) ==="
echo ""

failures=0
skipped=0
passed=0

for gate in "${GATES[@]}"; do
  script="scripts/gates/${gate}-verify.sh"
  if [ ! -f "$script" ]; then
    echo "[SKIP] $gate: missing $script"
    skipped=$((skipped + 1))
    continue
  fi
  if [ ! -x "$script" ]; then
    # Some generated scripts may not be chmod +x'd on Windows; bash doesn't actually
    # require +x for explicit 'bash script.sh' invocation, so this is a soft check.
    chmod +x "$script" 2>/dev/null || true
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] $gate: $script schedulable"
    continue
  fi
  echo "── $gate ────────────────────────────────────────"
  if bash "$script"; then
    passed=$((passed + 1))
  else
    echo "[FAIL] $gate returned non-zero"
    failures=$((failures + 1))
  fi
  echo ""
done

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[OK] dry-run complete — ${#GATES[@]} gate(s) checked"
  exit 0
fi

echo "=== Summary ==="
echo "  passed:  $passed"
echo "  skipped: $skipped (missing scripts)"
echo "  failed:  $failures"
echo ""

if [ "$failures" -gt 0 ]; then
  echo "[FAIL] $failures gate(s) failed"
  exit 1
fi

echo "[OK] all gates passed (profile=$PROFILE)"
