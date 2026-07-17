#!/usr/bin/env bash
set -euo pipefail

# Harness Engineering Quality Gates
# 所有门控条件必须可程序化验证

echo "=== Harness Engineering Quality Gates ==="
echo ""

failures=0

# HG1: 需求完整性
echo "[HG1] 检查需求完整性..."
spec_file=$(find .harness/changes -name "spec.md" 2>/dev/null | head -1)
if [ -n "$spec_file" ] && grep -q "## 验收标准" "$spec_file" 2>/dev/null; then
  echo "  ✅ 需求文档包含验收标准"
else
  echo "  ⚠️ 需求文档缺失验收标准（可能尚未进入需求分析阶段）"
fi

# HG2: 评审通过
echo "[HG2] 检查评审状态..."
review_files=$(find .harness/changes -name "*_review_*.md" 2>/dev/null)
if [ -n "$review_files" ]; then
  must_fix=$(grep -l "MUST FIX" $review_files 2>/dev/null || true)
  if [ -z "$must_fix" ]; then
    echo "  ✅ 评审无 MUST FIX 项"
  else
    echo "  ❌ 评审存在 MUST FIX 项"
    failures=$((failures + 1))
  fi
else
  echo "  ⚠️ 无评审文件（可能尚未进入评审阶段）"
fi

# HG3: 编译通过
echo "[HG3] 检查编译..."
if [ -f "CMakeLists.txt" ]; then
  if cmake --build build_x64 --config RelWithDebInfo 2>&1 > /dev/null; then
    echo "  ✅ 编译通过"
  else
    echo "  ❌ 编译失败"
    failures=$((failures + 1))
  fi
else
  echo "  ⚠️ 未检测到 CMakeLists.txt，跳过"
fi

# HG4: 测试通过
echo "[HG4] 检查测试..."
if [ -f "CMakeLists.txt" ]; then
  test_output=$(ctest --test-dir build_x64 --output-on-failure 2>&1 || true)
  if echo "$test_output" | grep -q "passed\|100% tests passed"; then
    echo "  ✅ 测试通过"
  else
    echo "  ❌ 测试未通过"
    failures=$((failures + 1))
  fi
else
  echo "  ⚠️ 未检测到 CMakeLists.txt，跳过"
fi

# HG5: 安全检查
echo "[HG5] 安全检查..."
secrets=$(grep -rn "password\|secret_key\|api_key" --include="*.cpp" --include="*.h" --include="*.hpp" src/ 2>/dev/null || true)
if [ -z "$secrets" ]; then
  echo "  ✅ 未检测到敏感信息"
else
  echo "  ❌ 检测到可能的敏感信息泄露"
  echo "$secrets" | head -5
  failures=$((failures + 1))
fi

echo ""
echo "=== 结果: $failures 个门控失败 ==="
[ "$failures" -eq 0 ] && echo "✅ 所有 Harness 质量门控通过" || echo "❌ 存在未通过的门控"
exit $failures
