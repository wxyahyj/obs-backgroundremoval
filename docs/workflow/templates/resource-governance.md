# Resource Governance

资源包括文档、图片、视频、音频、测试报告、临时脚本、生成物、长期规范和任务证据。

| Kind | Git Policy | Retention | Update Trigger | Examples |
| --- | --- | --- | --- | --- |
| canonical-docs | commit | long-lived | 架构、模块关系、规范、命令或用户路径变化时更新。 | README.md, AGENTS.md, docs/architecture/**, docs/standards/** |
| task-artifacts | commit-summary-only | task-lifetime | M/L/CRITICAL 任务执行和交付时维护，完成后保留 summary 与关键证据。 | docs/worklog/tasks/<task>/*.md |
| generated-evidence | ignore-by-default | evidence-window | 只在审计、回归复现或发布证据需要时保留。 | screenshots, videos, trace logs, e2e reports |
| temporary-work-files | ignore-by-default | short-lived | 任务结束前清理或沉淀为正式脚本/文档。 | tmp/**, .agent/logs/**, one-off scripts |
| large-media-assets | external-store | long-lived | 通过外部资产库管理，git 中只保留索引、用途和版本。 | design videos, audio, large datasets |

## Cleanup Rule
- 任务结束前必须检查临时脚本、日志、截图、视频、E2E 报告是否需要删除、移动到 evidence、或转成 summary。
- 长期文档只保留最终事实、架构决策、模块关系和维护规则，不保留过程草稿。
- 模块级产品方案和技术方案必须在需求影响模块边界、数据结构、API、权限或用户路径时同步更新。
