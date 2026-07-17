#!/usr/bin/env bash
set -euo pipefail

profile="${1:-default}"
include_optional=0
if [ "${2:-}" = "--include-optional" ]; then
  include_optional=1
fi

config=".agent/project.json"
if [ ! -f "$config" ]; then
  echo "[FAIL] missing $config" >&2
  exit 1
fi

# 使用 jq 验证 verification profile 是否存在
if ! jq -e ".verification_profiles.\"$profile\"" "$config" > /dev/null 2>&1; then
  echo "[FAIL] unknown verification profile: $profile"
  exit 2
fi

failures=0

# 使用 jq 提取 profile 的 required 和 optional 检查项
required_checks=$(jq -r ".verification_profiles.\"$profile\".required // [] | .[]" "$config" 2>/dev/null)
if [ "$include_optional" -eq 1 ]; then
  optional_checks=$(jq -r ".verification_profiles.\"$profile\".optional // [] | .[]" "$config" 2>/dev/null)
else
  optional_checks=""
fi

# 提取 service_matrix
service_count=$(jq -r '.service_matrix | length // 0' "$config" 2>/dev/null || echo "0")

if [ "$service_count" -eq 0 ]; then
  # 无 service_matrix，直接使用全局 commands
  for check in $required_checks; do
    cmd=$(jq -r ".commands.\"$check\" // \"\"" "$config" 2>/dev/null || echo "")
    if [ -z "$cmd" ] || [[ "$cmd" == echo* ]]; then
      echo "[WARN] required global/$check has no executable command"
      failures=$((failures + 1))
      continue
    fi
    echo "[RUN] required global/$check: $cmd"
    bash -lc "$cmd" || failures=$((failures + 1))
  done
  for check in $optional_checks; do
    cmd=$(jq -r ".commands.\"$check\" // \"\"" "$config" 2>/dev/null || echo "")
    if [ -z "$cmd" ] || [[ "$cmd" == echo* ]]; then
      echo "[WARN] optional global/$check has no executable command"
      continue
    fi
    echo "[RUN] optional global/$check: $cmd"
    bash -lc "$cmd" || failures=$((failures + 1))
  done
else
  # 有 service_matrix，遍历服务
  service_ids=$(jq -r ".verification_profiles.\"$profile\".services // [.service_matrix[] | select(.default) | .id] | .[]" "$config" 2>/dev/null)
  for service_id in $service_ids; do
    service_dir=$(jq -r ".service_matrix[] | select(.id == \"$service_id\") | .directory // \".\"" "$config" 2>/dev/null || echo ".")

    for check in $required_checks; do
      cmd=$(jq -r "((.service_matrix[] | select(.id == \"$service_id\") | .commands.\"$check\") // .commands.\"$check\" // \"\")" "$config" 2>/dev/null || echo "")
      if [ -z "$cmd" ] || [[ "$cmd" == echo* ]]; then
        echo "[WARN] required $service_id/$check has no executable command"
        failures=$((failures + 1))
        continue
      fi
      if [ ! -d "$service_dir" ]; then
        echo "[FAIL] service directory missing: $service_id -> $service_dir"
        failures=$((failures + 1))
        continue
      fi
      echo "[RUN] required $service_id/$check: $cmd"
      (cd "$service_dir" && bash -lc "$cmd") || failures=$((failures + 1))
    done

    for check in $optional_checks; do
      cmd=$(jq -r "((.service_matrix[] | select(.id == \"$service_id\") | .commands.\"$check\") // .commands.\"$check\" // \"\")" "$config" 2>/dev/null || echo "")
      if [ -z "$cmd" ] || [[ "$cmd" == echo* ]]; then
        echo "[WARN] optional $service_id/$check has no executable command"
        continue
      fi
      if [ ! -d "$service_dir" ]; then
        echo "[FAIL] service directory missing: $service_id -> $service_dir"
        failures=$((failures + 1))
        continue
      fi
      echo "[RUN] optional $service_id/$check: $cmd"
      (cd "$service_dir" && bash -lc "$cmd") || failures=$((failures + 1))
    done
  done
fi

if [ "$failures" -gt 0 ]; then
  echo "[FAIL] verification profile '$profile' failed: $failures failure(s)"
  exit 1
fi

echo "[OK] verification profile '$profile' passed"
