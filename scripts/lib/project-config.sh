#!/usr/bin/env bash
set -euo pipefail

PROJECT_CONFIG=".agent/project.json"

if [ ! -f "$PROJECT_CONFIG" ]; then
  echo "[FAIL] missing $PROJECT_CONFIG" >&2
  exit 1
fi

json_get() {
  # 将点分隔路径转换为 jq 路径表达式，例如 "commands.build" -> ".commands.build"
  local jq_path
  jq_path=".$(echo "$1" | sed 's/\././g')"
  local value
  value=$(jq -r "$jq_path // empty" "$PROJECT_CONFIG" 2>/dev/null) || return 2
  if [ -z "$value" ] || [ "$value" = "null" ]; then
    return 2
  fi
  echo "$value"
}

command_for_gate() {
  local key="$1"
  local value
  value="$(json_get "commands.$key" 2>/dev/null || true)"
  if [ -n "$value" ]; then
    echo "$value"
    return 0
  fi

  # 使用 jq 从 stacks.<stack>.commands 中读取命令
  local stack value
  stack=$(jq -r '.stack // "generic"' "$PROJECT_CONFIG" 2>/dev/null) || return 0
  value=$(jq -r ".stacks.\"${stack}\".commands.\"${key}\" // empty" "$PROJECT_CONFIG" 2>/dev/null) || return 0
  if [ -z "$value" ] || [ "$value" = "N/A" ] || [ "$value" = "null" ]; then
    return 0
  fi
  echo "$value"
}
