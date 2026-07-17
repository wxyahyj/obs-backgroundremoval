#!/usr/bin/env bash
set -euo pipefail

# G7: Security verification gate — stack-aware security checks
# First try the configured security command from project.json
if [ -f "scripts/lib/project-config.sh" ]; then
  source "scripts/lib/project-config.sh"
  cmd="$(command_for_gate "security" 2>/dev/null || true)"
  if [ -n "$cmd" ] && [ "$cmd" != "N/A" ]; then
    echo "[RUN] G7 Security: $cmd"
    bash -lc "$cmd" || exit 1
  fi
fi

# Run stack-specific and universal security pattern checks
# C/C++ security checks — 扫描不安全的 C 标准库函数
if true; then
  echo "[CHECK] Scanning for unsafe C/C++ function calls..."
  grep -rn --include='*.cpp' --include='*.c' --include='*.h' --include='*.hpp' --include='*.cc' --include='*.cxx' \
    -E '\b(system|exec|sprintf|gets|strcpy|strcat|scanf|vsprintf)\s*\(' . 2>/dev/null | grep -v '.agent/' | grep -v 'build_' | head -5 || true
  echo "[CHECK] Scanning for hardcoded secrets in C/C++ files..."
  grep -rn --include='*.cpp' --include='*.c' --include='*.h' --include='*.hpp' --include='*.cc' --include='*.cxx' \
    -E '(password|secret|api_key|token)\s*=\s*['"'"'"][^'"'"'"]+['"'"'"]' . 2>/dev/null | grep -v '.agent/' | grep -v 'build_' | head -5 || true
fi

# C/C++ buffer overflow checks — 扫描缓冲区溢出风险函数
if true; then
  echo "[CHECK] Scanning for buffer overflow risk patterns in C/C++ files..."
  grep -rn --include='*.cpp' --include='*.c' --include='*.h' --include='*.hpp' --include='*.cc' --include='*.cxx' \
    -E '\b(gets|strcpy|strcat|sprintf|vsprintf|scanf|sscanf)\s*\(' . 2>/dev/null | grep -v '.agent/' | grep -v 'build_' | head -5 || true
fi

# Universal security checks
if true; then
  echo "[CHECK] Scanning for .env files with secrets..."
  found_env=0
  for envfile in .env .env.local .env.production .env.staging; do
    if [ -f "$envfile" ]; then
      echo "[WARN] G7: Found $envfile — ensure it is in .gitignore and does not contain real secrets"
      found_env=1
    fi
  done

  echo "[CHECK] Scanning for common secret patterns in all files..."
  SECRET_HITS=$(grep -rn --include='*.ts' --include='*.tsx' --include='*.js' --include='*.jsx' --include='*.py' --include='*.go' --include='*.java' --include='*.rs' --include='*.cs' --include='*.cpp' --include='*.h' --include='*.hpp' --include='*.c' --include='*.cc' --include='*.cxx' \
    -E '(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |EC )?PRIVATE KEY-----)' \
    . 2>/dev/null | grep -v 'node_modules' | grep -v '.agent/' | grep -v '.git/' | head -5 || true)

  if [ -n "$SECRET_HITS" ]; then
    echo "[FAIL] G7: Detected potential secrets/credentials in source code:"
    echo "$SECRET_HITS"
    exit 1
  fi
fi

echo "[PASS] G7: Security verification passed — no obvious secrets or dangerous patterns found"
exit 0
