# Spec - {{TASK_ID}}

Date: {{DATE}}
Level: {{LEVEL}}

> P0 六要素契约（借鉴 Codex Goals 完成契约模型）。`Outcome` 复用 `What`，其余为可选补强字段；
> 留空即视为未声明，不会破坏旧流程。CLI 对应参数见 `docs/workflow/TEMPLATE_GUIDE.md`。

## What

<!-- Outcome: 期望的最终现实状态，而非任务步骤描述。 -->

## Why


## Verification Surface

<!-- 具体证据来源：测试名 / 基准命令 / 产物路径。verify/review/ship 的 evidence 应能映射回这里。 -->
<!-- CLI: scale define ... --verification-surface "tests/test_foo.cpp,ctest" -->
-

## Constraints

<!-- 运行期间不能退化的指标（性能 / 安全 / 兼容性）。 -->
<!-- CLI: --constraints "p95 < 200ms,no new prod dependency" -->
-

## Boundaries

<!-- CLI: --boundary-files / --boundary-tools / --boundary-forbidden -->
- Files:
- Tools:
- Forbidden:

## Iteration Strategy

<!-- build 阶段每轮迭代后如何决定下一步。CLI: --iteration-strategy "..." -->

## Blocked Stop Condition

<!-- 无可行路径时报告什么、需要什么才能解锁。CLI: --blocked-stop "..." -->

## Acceptance Criteria

- [ ] 
