# Engineering Rules

这份文档是长期维护的工程红线，面向人类与 Agent 协同开发。

## Logging
- 默认禁止新增无约束 std::cout、printf、blog(LOG_INFO, ...) 或调试日志。
- 生产日志必须使用项目统一 logger，并带级别、traceId/requestId 和模块名。
- 禁止输出 token、password、secret、cookie、手机号、身份证、邮箱、连接串、文件私密路径。
- 临时日志必须在任务完成、验收前、发版前清理，或者转为受控 debug 日志。

## Data And OBS API
- 数据访问必须遵守项目 OBS API / 模型推理接口 规范，禁止绕过统一事务、租户、权限和审计层。
- SQL 或查询条件必须参数化，禁止字符串拼接用户输入。
- 迁移、DDL、批量删除和数据修复脚本属于 CRITICAL 任务，必须有回滚方案和人工确认。

## Architecture
- 修改前识别模块边界、调用关系、数据流和同类实现。
- 公共接口、API、事件、数据库结构变化必须同步更新架构文档、接口文档和任务 summary。
- 不为单次需求新增全局抽象；只有重复复杂度或稳定边界出现后才抽象。

## Security
- 所有用户输入、URL、HTML、Markdown、文件路径、SQL 条件、命令参数都按不可信处理。
- C++ 代码必须防缓冲区溢出、use-after-free、整数溢出。
- 外部依赖、skills、MCP、CLI 安装前必须做供应链审查。

## UI And UX
- 用户可见流程必须有 Mini-PRD 或等价验收标准。
- UI 变更必须检查 OBS 属性面板布局、参数可见性控制、多配置切换、空状态、加载态、错误态和可访问性。
- 设计类任务应组合 UI/UX skill、OBS 界面截图证据，而不是只写代码后脑补效果。

## Verification And Release
- 未真实运行的命令不得写成通过。
- 失败命令必须记录退出码、摘要、影响和后续动作。
- 发版前必须跑 release verification profile，并清理临时资产。
