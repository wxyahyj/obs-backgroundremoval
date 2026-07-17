# SCALE Quality Contract

- Project: OBS Background Removal Plugin
- Agent: trae
- Stack: cpp
- Scenario: standard

## Source Of Truth
- agentEntry: `AGENTS.md / CLAUDE.md / equivalent agent entry`
- workflowState: `.scale/workflow.json`
- runtimeCommands: `.agent/project.json`
- qualityContract: `.scale/quality-contract.json`
- skillRegistry: `.scale/skills-registry.json`

## Task Levels
| Level | Intent | Required Artifacts | Required Verification |
| --- | --- | --- | --- |
| S | 单点小改、文案、注释、低风险配置。Fast-lane profile，仅 G0/G4/G5。 | verification.md | relevant command only |
| M | 普通功能、bugfix、2-5 个文件的行为变化。Standard profile。 | explore.md, plan.md, reality-check.md, runtime.md, resource-cleanup.md, verification.md, summary.md | cppcheck --enable=all src/, ctest --test-dir build_x64 --output-on-failure, cmake --build build_x64 --config RelWithDebInfo |
| L | 跨模块、架构、数据模型、迁移或高影响重构。Full profile，含 G18 运行时证据。 | mini-prd.md when user-facing, explore.md, spec.md, tasks.md, plan.md, reality-check.md, runtime.md, resource-cleanup.md, review.md, verification.md, summary.md | cmake --build build_x64 --config RelWithDebInfo, cppcheck --enable=all src/, ctest --test-dir build_x64 --output-on-failure, cmake --build build_x64 --config RelWithDebInfo |
| CRITICAL | 认证、权限、资金、生产配置、删除、迁移、外部集成。Comprehensive profile，G16–G20 全套。 | mini-prd.md, spec.md, tasks.md, explore.md, plan.md, reality-check.md, runtime.md, resource-cleanup.md, security-review.md, rollback-plan, review.md, verification.md, summary.md | cmake --build build_x64 --config RelWithDebInfo, cppcheck --enable=all src/, ctest --test-dir build_x64 --output-on-failure, cmake --build build_x64 --config RelWithDebInfo, security profile |

## Verification Profiles
| Profile | Required | Optional | Success Rule |
| --- | --- | --- | --- |
| fast | ctest --test-dir build_x64 --output-on-failure | cppcheck --enable=all src/ | 只用于 S 级或局部验证，不能代表发布可用。 |
| default | cppcheck --enable=all src/, ctest --test-dir build_x64 --output-on-failure, cmake --build build_x64 --config RelWithDebInfo | cmake --build build_x64 --config RelWithDebInfo | M 级默认闭环，失败必须记录原因和修复循环。 |
| release | cmake --build build_x64 --config RelWithDebInfo, cppcheck --enable=all src/, ctest --test-dir build_x64 --output-on-failure, cmake --build build_x64 --config RelWithDebInfo | coverage profile, security profile, e2e profile | 发版前必须全部真实运行并在 verification.md 记录退出码。 |
| productSmoke | cmake --build build_x64 --config RelWithDebInfo, ctest --test-dir build_x64 --output-on-failure | e2e profile, browser automation | status: "skipped" 不算完成证据；必须真实跨过产品边界（路由/认证/存储/异步任务）。 |

## Skill Policy
- Mode: progressive-disclosure
- Max always-visible skills: 24
- Install safety: 只从可信仓库或已审查来源安装。
- Install safety: 安装前检查脚本、二进制、网络下载、postinstall、权限和 license。
- Install safety: 优先固定版本或 commit；禁止无审查执行 curl | bash。
- Usage evidence: 说明为什么选这些 skills、MCP 或 CLI。
- Usage evidence: 记录实际调用的工具和输出证据。
- Usage evidence: 技能失败时记录失败原因和降级方案。

## Resource Governance
| Kind | Git Policy | Retention | Update Trigger | Examples |
| --- | --- | --- | --- | --- |
| canonical-docs | commit | long-lived | 架构、模块关系、规范、命令或用户路径变化时更新。 | README.md, AGENTS.md, docs/architecture/**, docs/standards/** |
| task-artifacts | commit-summary-only | task-lifetime | M/L/CRITICAL 任务执行和交付时维护，完成后保留 summary 与关键证据。 | docs/worklog/tasks/<task>/*.md |
| generated-evidence | ignore-by-default | evidence-window | 只在审计、回归复现或发布证据需要时保留。 | screenshots, videos, trace logs, e2e reports |
| temporary-work-files | ignore-by-default | short-lived | 任务结束前清理或沉淀为正式脚本/文档。 | tmp/**, .agent/logs/**, one-off scripts |
| large-media-assets | external-store | long-lived | 通过外部资产库管理，git 中只保留索引、用途和版本。 | design videos, audio, large datasets |

## Red Lines
- 不得声称未运行的验证通过。
- 不得把临时日志、截图、视频、抓包和一次性脚本默认提交到 git。
- 不得在日志、文档、测试报告中输出 token、密码、手机号、身份证、密钥和连接串。
- 不得绕过项目的 OBS API、Qt UI 框架、日志、错误处理、安全和 UI 规范。
