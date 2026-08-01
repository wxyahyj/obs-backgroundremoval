# obs-backgroundremoval standalone 项目长期记忆

## 项目定位（2026-07-24 确认）
- 用途：**游戏瞄准辅助**（截屏→YOLO 检测→移动鼠标）。**不需要 OBS**（OBS 是直播软件，瞄准用不上）。
- 主线决策（最终）：**standalone 当主线，择优把 AiMod 的顺对齐进来**。曾考虑「AiMod 主线 + 打包瞄准栈」，评估后放弃——瞄准栈虽可剥离（FullAimBridge 干净门面，~1万行可复制），但 AiMod 缺 Web/配置/模型管理且负债多（硬编码/闭源lib/cout/自旋），搬过去≈重建；择优对齐改动更小、保留全部已落地能力。
- 对齐清单（收益/改动比序）：①拆 Display/preview 编码线程 ②检测改版本固定解码（用 model_version 直走 postprocess，不猜 shape）③aim 控制器 tick 移出主环 ④letterbox 借 SIMD ⑤配置收敛单面。
- 瞄准栈搬运事实（备查）：FullAimBridge 仅 include MouseControllerInterface+Detection.h+Factory+Windows.h，不依赖 ConfigStore/TrackerEngine；AiMod 接入点 AiMod.cpp:424-442(cout stub)；slim 路径 AimEngine+types.h 4文件870行零依赖。

## 瞄准栈搬运（关键事实）
- FullAimBridge 是干净门面：仅 include MouseControllerInterface + Detection.h + Factory + Windows.h，不依赖 ConfigStore/TrackerEngine/infer_c_api。
- 整栈 ~1 万行可原样复制（8 后端 / 5 算法 / 4 预测器自包含 C++），适配仅 ~200 行。
- AiMod 接入点：`AiMod.cpp:424-442`（cout stub），替换为 AimEngine.tick / FullAimBridge.tick。
- slim 路径：AimEngine + types.h，4 文件 ~870 行零依赖（基础 PID + SendInput）。
- 外部依赖：OpenCV（AiMod 已有）、Eigen（可选可裁）、setupapi/hid（系统库）。

## AiMod 负债（搬瞄准时顺手清）
- 模型路径硬编码 `D:\ALow\...`（AiMod.cpp:42）→ 改可配置 / 相对路径
- pid_x64.lib 闭源带授权码（setName("1458679219")）→ 搬 standalone 开源 PID 后弃用
- 自旋轮询、cout 热路径 I/O

## knowledge-mem MCP 读法
- `mem_fs cat` 必须用 id 路径 `/memories/by-id/<id>.memory.md`（slug 路径返回错误内容 185b1f1d）
- `memory_search` 默认只搜 default 空间，搜 ncnn-debug 等需传 space_id

## 构建环境雷区
- 沙箱 cmake --build 调 MSBuild 崩溃（需本机 VS2022 / x64 Native Tools 编译）
- CUDA toolkit 装在 D:/CUDATool 未注册到 VS

## 目录
- standalone: E:\obs-heji\obs-backgroundremoval\standalone（src/aim, src/core, src/capture, src/web）
- AiMod: E:\obs-heji\AiMod（AiMod.cpp 单文件 + cap/ + yolodml/ + MotionSimulator + curve + pid）
- 父仓瞄准框架: E:\obs-heji\obs-backgroundremoval\src（MouseControllerInterface/Factory, AbstractMouseController, 8 后端, 5 算法, 预测器）
