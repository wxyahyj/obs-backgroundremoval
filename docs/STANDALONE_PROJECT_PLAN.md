# 独立软件项目计划书 v2.0 — 重做外围,复用核心

> 版本:v2.0 · 2026-08-02 · 状态:✅ 全部里程碑完成(M0-M7 交付)
> 决策:现有 `standalone/` 实现质量不达标(代码结构乱 / UI 差 / 不稳定 / 架构不满)。
> 方向:**重写外围(host/config/web/UI),复用主仓核心(推理/瞄准/跟踪)**,保持 WebUI 技术栈,功能必须**完全体**(与 OBS 插件全对齐)。

---

## 1. 现状与决策依据

### 1.1 现有 standalone(旧实现,仅作参考基线)

| 组件 | 行数 | 问题 |
|---|---|---|
| `src/host/EngineLoop.cpp` | 1598 | 巨型单文件,职责混杂(grab/infer/track/aim/web 状态全在一个循环) |
| `src/core/ConfigStore.cpp` | 1400 | 扁平键表散落,无数据模型 |
| `src/web/WebServer.cpp` | 779 | HTTP+WS+路由+静态文件一坨 |
| `webui/index.html + app.js` | 848+1345 | 单页堆叠,无组件化 |

**共同病根**:和插件本体(yolo-detector-filter.cpp 314KB)一样的"大文件主义"。核心推理/瞄准栈本身质量可用,复用。

### 1.2 复用清单(主仓 `src/`,不改或仅 stub 适配)

| 模块 | 文件 | 适配 |
|---|---|---|
| 推理 | `models/ModelYOLO.cpp/h`、`IYoloModel.h`、`DmlPreprocessor`、`CudaPreprocessor.cu` | obs_log/obs_module_file → stub(已有 obs_stubs.cpp) |
| 瞄准栈 | `AbstractMouseController`、`MouseControllerFactory`、8 后端、全部 PID/预测器 | 仅 obs_log → stub |
| 跟踪 | `HungarianAlgorithm`、`KalmanFilter`、`IMMFilter`、`TargetTracker`、`GhostTracker` | 仅 obs_log |
| 准星 | `CrosshairDetector` | obs_log stub |
| 后座 | `RecoilPatternManager` | `obs_module_config_path` 一行替换 |
| 配置底层 | `ConfigManager`(JSON) | 仅作底层读写,外围重写模型层 |
| 截图 | `standalone/third_party/screen_capture`(DXGI/WGC/GDI) | 保留,vendored |

### 1.3 技术栈(固定)

- C++17 + CMake + MSVC x64,Windows-only
- WebUI(纯 HTML/CSS/JS,无构建工具,内嵌资源) — **不要 Qt 页面**
- ONNX Runtime(DML/CUDA/TensorRT EP 检测沿用)
- D3D11(截图 + 可选叠加层)

---

## 2. 新架构(重写外围)

### 2.1 目录结构(新)

```
standalone/
├── CMakeLists.txt / cmake/
├── config/default.json
├── models/*.onnx
├── webui/                        ← 新前端(组件化)
├── src/
│   ├── app/                      ← 新:组装与生命周期
│   │   ├── main.cpp              ← 入口:解析参数→App
│   │   └── App.cpp/h             ← 组装 Engine+Web+Tray,启动/停止
│   ├── engine/                   ← 新:每帧环,拆小(替代 EngineLoop)
│   │   ├── Engine.h/.cpp         ← 循环调度 + 线程管理(≤300行)
│   │   ├── FramePipeline.h/.cpp  ← 帧状态机:grab→infer→track→aim→preview
│   │   └── EngineStats.h         ← FPS/分阶段延迟统计
│   ├── capture/                  ← 保留 FrameSource 薄封装 + ycap_c_api
│   ├── inference/                ← 新:InferEngine 包装 ModelYOLO(复用)
│   ├── aim/                      ← 复用 FullAimBridge/AimEngine + 主仓瞄准栈
│   ├── config/                   ← 新:配置模型重写
│   │   ├── ConfigModel.h/.cpp    ← 纯数据结构(5套配置/算法/热键/后座/准星)
│   │   ├── ConfigStore.h/.cpp    ← 加载/保存/迁移/热更新(≤400行)
│   │   └── ConfigKeyMap.h        ← obs 键 ↔ 模型字段映射表(唯一真源)
│   ├── web/                      ← 新:重写
│   │   ├── HttpServer.h/.cpp     ← winsock HTTP+WS 框架(可复用旧框架层)
│   │   ├── ApiRoutes.h/.cpp      ← 路由注册 + handler(薄)
│   │   └── WebUi.h               ← 内嵌资源服务
│   ├── overlay/                  ← 新:可选 D3D11 游戏内叠加(bbox/FOV)
│   ├── util/                     ← 新:Log.h(轮转)、CrashGuard、ErrorFallback
│   ├── net/                      ← 保留 RemoteInferClient(不扩展)
│   └── core/                     ← obs_stubs.cpp(保留)
├── third_party/screen_capture/   ← 保留
└── docs/OBS_PARITY.md            ← 对照表持续维护
```

### 2.2 线程模型(明确,一图)

```
Capture线程 ──帧──> Infer线程 ──dets──> Aim线程 ──鼠标──> (游戏)
                        │
Web线程(winsock) ──状态/配置──> EngineStats(atomic) <──所有线程写
                        │
                    FloatingPreview/Overlay(UI线程)
```

- 帧缓冲:无锁 ring buffer(沿用插件思路,4 槽 CAS)
- 结果:age-gated shared_ptr 消费(沿用)
- 配置:单写者(Web)+ 原子发布,热更新
- 每线程职责单一,文件 ≤500 行(例外:模型适配层)

### 2.3 API 设计(与旧版兼容为主,规整化)

| 方法 | 路径 | 变化 |
|---|---|---|
| GET/PUT | `/api/config` | 返回/合并 ConfigModel 全文档(含 obs_keys) |
| POST | `/api/engine/start\|stop\|reload_model\|reload_capture` | 不变 |
| POST | `/api/controller/test` | 不变 |
| GET | `/api/status` `/api/detections` `/api/preview.bmp` | 不变 |
| WS | `/ws` | 状态推送规范化(JSON schema 固定) |
| 新增 | `/api/crosshair/pick` | 取色(M4) |
| 新增 | `/api/overlay/toggle` | 叠加开关(M5) |

---

## 3. 实施里程碑

### M0 — 基线固化(本阶段,立即)
- [x] 修订计划书(v2.0)
- [ ] `.gitignore` 白名单 `!/standalone`(排除 build/models 大文件)
- [ ] `git add standalone/` 提交基线(旧实现保留可回滚)
- [ ] `git restore scripts/` 恢复 gate/qa/打包脚本
- [ ] 验证旧 standalone 可构建(基线锁定)

验收:HEAD 含完整 standalone + scripts,clean checkout 可复现。

### M1 — 新骨架 + 核心复用闭环
- [ ] 新目录结构(src/{app,engine,inference,config,web,overlay,util})
- [ ] CMake 重写:链接主仓核心(ModelYOLO+瞄准栈+跟踪)+ screen_capture
- [ ] 最小闭环:`grab → infer → log` 跑通(DXGI 截图 → ORT 推理 → 日志打印 dets)
- [ ] `Engine` 骨架:线程 + ring buffer + stats

验收:新工程构建成功,控制台看到真实检测输出。

### M2 — 配置层重写 + 全量键核对(功能完全体关键)
- [ ] `ConfigModel`:5 套配置/算法/热键/后座/准星/FOV 纯数据结构
- [ ] `ConfigStore`:load/save/migrate/热更新,user.json 格式
- [ ] `ConfigKeyMap`:提取 filter_properties.cpp 全部键名 → 模型字段对照表
- [ ] 输出 `docs/config-parity.md`:逐键 ✓/✗,✗ 即功能缺口

验收:filter_properties 每控件有对应字段,PUT 全键热生效。

### M3 — 引擎模块化重写
- [ ] `FramePipeline`:状态机拆解(替代 EngineLoop 1598 行)
- [ ] 瞄准栈接入:FullAimBridge + 8 后端 + 5 算法(从旧版移植逻辑,新文件组织)
- [ ] 准星/后座/扳机/FOV 逻辑对齐
- [ ] EngineStats 分阶段计时(preprocess/infer/post/total)

验收:热路径行为与 OBS 插件等价,延迟统计齐全。

### M4 — Web 层 + WebUI 重做
- [ ] `HttpServer` 框架层(与旧版解耦)+ `ApiRoutes` 薄路由
- [ ] WebUI 重做:组件化(侧栏导航/表单页/状态面板/预览页),多页结构
- [ ] WS 状态推送 + 前端实时渲染
- [ ] `/api/crosshair/pick` 取色 UI(屏幕点选)

验收:全部配置页可操作且热生效,状态实时刷新,取色可用。

### M5 — 功能完全体
- [ ] 游戏内叠加层(方案 B 置顶窗优先 → 方案 A D3D11 overlay 开关)
- [ ] OBS 场景数据导入(`import_obs_scene.py` + webui 入口)
- [ ] 远程推理客户端保留验证(不扩展)
- [ ] 缺失功能清单核对(对照 OBS_PARITY.md 全 ✓)

验收:与插件功能逐项对齐,无缺口。

### M6 — 稳定性
- [ ] 日志轮转 + 崩溃处理(UnhandledExceptionFilter → minidump + 重启)
- [ ] 回退:捕获失败→切 GDI;模型失败→提示;配置损坏→备份+default
- [ ] 8h 连续运行验证

验收:人为破坏(删配置/拔设备/坏模型)可自恢复。

### M7 — 打包 + 回归门禁
- [ ] Inno Setup 安装器 + 便携 ZIP(GPU 检测、模型检查)
- [ ] `scripts/qa/obs_parity_check.py` + `product-smoke.ps1` 跑通
- [ ] 发布文档(构建/安装/FAQ)

验收:干净机装完能跑,门禁全绿。

---

## 4. 工程纪律

- 每个新文件 ≤500 行;超了必须拆
- 每个模块:头文件自包含 + 注释说明线程归属
- 每里程碑结束:构建 + 冒烟 + 验收项
- 旧 standalone 代码是**参考实现**,逐步替换,不边写边删(每阶段结束统一清理)

---

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| M2 键核对遗漏 → 功能不完全体 | config-parity.md 机器可查,全 ✓ 才过 M2 |
| 复用核心 obs_log stub 不完整 | 链接期报错兜底 + 统一 stub 头 |
| WebUI 重做周期长 | 组件化增量,先表单页后状态页 |
| 叠加层与游戏冲突 | M5 默认置顶窗,overlay 开关 |
| 干净机缺 CUDA 库 | M7 GPU 检测 + 回落 DML/CPU |

---

## 6. 执行顺序

```
M0(基线) → M1(骨架闭环) → M2(配置层) → M3(引擎)
        → M4(Web+UI) → M5(完全体) → M6(稳定) → M7(打包门禁)
```

当前进度:全部完成。

## 7. 交付记录

| 里程碑 | 验证 |
|---|---|
| M0 基线 | standalone 64 文件入 git, scripts 恢复 |
| M1 骨架 | 构建 + 冒烟(dets=11, CUDA EP) |
| M2 配置层 | config_test ALL PASS(序列化/键映射/合并/恢复) |
| M3 引擎 | 双线程管线, controller created |
| M4 Web | 页面 200, PUT 热生效 |
| M5 功能 | preview 640x640 / pick RGB / overlay shown / OBS 导入映射 |
| M6 稳定 | host.log 轮转, minidump 就绪, 捕获 GDI 回退 |
| M7 打包门禁 | parity ok(0err/0warn), smoke ALL PASS, 便携 ZIP + 安装器 |
