# YOLO Aim 独立项目实施计划

> 目标：从 OBS 插件剥离为 **启动器 EXE + 可调用 DLL 组合**，UI 仅 WebUI，截图使用本目录 vendored 的 DXGI/WGC/GDI 高性能源码。

---

## 0. 产品形态（已定）

| 组件 | 类型 | 职责 |
|------|------|------|
| `yolo_capture.dll` | 共享库 | DXGI / WGC / GDI 截图（基于 `third_party/screen_capture`） |
| `yolo_infer.dll` | 共享库 | ONNX YOLO 推理（从现有 `ModelYOLO` / `YOLODetector` 剥离） |
| `yolo_aim.dll` | 共享库 | 跟踪 + PID/Aim + 鼠标后端编排 |
| `yolo_host.exe` | 启动器 | 进程生命周期、热键、主循环、内嵌 HTTP/WebSocket、打开浏览器 |
| `webui/` | 静态资源 | 唯一 UI；浏览器访问 `http://127.0.0.1:17890` |

**不做：** Qt 桌面窗体、OBS 插件壳、FluentWidgets。  
**启动器行为：** 双击 → 起引擎线程 + 本地 Web 服务 → 可选自动打开浏览器 → 托盘最小化（可选二期）。

---

## 1. 架构总览

```
┌──────────────── yolo_host.exe ─────────────────┐
│  main: 解析 config → 起 EngineThread → 起 Web  │
│  EngineThread:                                  │
│    capture.grab → infer → track → mouse.tick    │
│  WebServer (cpp-httplib):                       │
│    REST /api/*  +  WS /ws 推送状态              │
└───────────┬───────────┬───────────┬────────────┘
            │           │           │
     yolo_capture   yolo_infer   yolo_aim
            │           │           │
         DXGI/WGC     ORT+OpenCV   PID/后端
         GDI BGR      ONNX 模型    Logi/MAKCU/...
```

数据流（每帧）：

```
IFrameSource::grab(FramePacket)
  → YOLODetector::inference(cv::Mat BGR)
  → Tracker::update(dets)
  → MouseControllerInterface::process(...)
  → Web 状态快照（低频 JSON 推送）
```

---

## 2. 目录结构（本文件夹）

```
standalone/
├── PLAN.md                 ← 本文档
├── README.md               ← 构建与运行说明
├── CMakeLists.txt          ← 独立工程入口
├── cmake/
│   └── Dependencies.cmake
├── config/
│   └── default.json
├── models/                 ← 放 .onnx（gitignore 大文件）
├── include/yolo_aim/       ← 对外 C / C++ 头
│   ├── capture_c_api.h
│   ├── infer_c_api.h
│   ├── aim_c_api.h
│   └── types.h
├── src/
│   ├── capture/            ← 对 third_party 的薄封装 + FramePacket
│   ├── core/               ← 从主仓同步/移植的推理（二期链接或 copy）
│   ├── aim/                ← AimEngine
│   ├── input/              ← 鼠标后端（二期从主仓移植）
│   ├── web/                ← HTTP + WS + 静态文件
│   └── host/               ← main.cpp 启动器
├── third_party/
│   └── screen_capture/     ← 已拷贝：DXGI/WGC/GDI 源码
├── webui/                  ← 前端
│   ├── index.html
│   ├── css/app.css
│   └── js/app.js
└── scripts/
    └── build.ps1
```

与主仓 `obs-backgroundremoval` 的关系：

- **一期：** `standalone/` 自包含 CMake；推理/瞄准源码用相对路径 `../src/...` 引用（减少复制）。
- **二期：** 需要彻底独立发行时，再把 `core/aim/input` 物理拷贝进 `standalone/src`。

---

## 3. 截图库接入（已就绪源码）

来源：`E:\下载\C++高性能截图（DXGI-WGC-GDI）.zip`  
落位：`third_party/screen_capture/`

### 3.1 现成 API（C 导出，`ScreenCapture.h`）

```c
ScreenCaptureHandle Create(CaptureTypeEnum type); // GDI=0, DIRECTX=1, WGC=2
int Init(handle, w, h);                           // 屏幕居中 ROI
int InitRegion(handle, x, y, w, h);               // 绝对区域
int SetWindow(handle, HWND);                      // WGC/GDI 窗口模式
int SetRegion(handle, x, y, w, h);
const unsigned char* CaptureBGR(handle);          // 连续 BGR，宽高 GetWidth/Height
const unsigned char* CaptureBMP(handle);
void Release(handle);
```

### 3.2 选型策略

| 模式 | 枚举 | 适用 | 默认 |
|------|------|------|------|
| DXGI Desktop Duplication | `CAPTURE_DIRECTX` | 全屏/无边框、最低延迟 | **是** |
| WGC | `CAPTURE_WGC` | 指定窗口、Win10 1903+ | 可选 |
| GDI | `CAPTURE_GDI` | 兼容兜底 | 兜底 |

### 3.3 与推理对接注意

- `CaptureBGR()` 返回 **内部缓冲指针**，下一帧会覆盖 → 立即 `cv::Mat(h,w,CV_8UC3,ptr).clone()` 或零拷贝进推理队列后在消费完再抓下一帧。
- 推荐热路径：**单缓冲 + 推理线程持有上一帧 Mat 所有权**，采集线程只在推理空闲时 overwrite。
- `Initialize(w,h)` 为居中裁剪；瞄准场景优先 `InitRegion` 或居中 FOV 尺寸（如 320/640）。
- DXGI 无窗口句柄；窗口捕获用 WGC。
- 坐标：截取 ROI 的屏幕原点 `(x,y)` 必须写入 `FramePacket`，供鼠标控制换算。

### 3.4 一期封装

`src/capture/FrameSource.hpp`：

```cpp
struct FramePacket {
  cv::Mat bgr;           // BGR8
  int64_t pts_ns = 0;
  int origin_x = 0;      // 屏幕坐标
  int origin_y = 0;
  int screen_w = 0;
  int screen_h = 0;
};

class FrameSource {
public:
  enum class Backend { DXGI, WGC, GDI };
  bool open(Backend b, int x, int y, int w, int h);
  bool open_center(Backend b, int w, int h);
  bool grab(FramePacket& out);  // 内部 CaptureBGR + Mat 包装
  void close();
  int width() const;
  int height() const;
};
```

实现仅调用 C API 或直接链 `IScreenCapture`（二选一，推荐 **直接 C++ `IScreenCapture`** 免一层 map，C API 留给外部语言）。

---

## 4. DLL / 模块边界

### 4.1 `yolo_capture`（最先落地）

- 源：`third_party/screen_capture/*` + `src/capture/*`
- 导出：`capture_c_api.h`（可与现有 `ScreenCapture.h` 对齐，统一前缀 `ycap_`）
- 依赖：`d3d11` `dxgi` `windowsapp` `dwmapi`（WGC）

### 4.2 `yolo_infer`

- 源：主仓 `src/models/ModelYOLO.*` `IYoloModel.h` `Detection.h` `YOLODetector.*` `CudaPreprocessor` `DmlPreprocessor`
- 替换：`obs_log` → `ya_log_*`；`obs_module_file` → 配置里的绝对/相对路径
- 导出 C API：

```c
YoloInfer* yolo_infer_create(const char* model_path, const char* device);
void       yolo_infer_destroy(YoloInfer*);
int        yolo_infer_bgr(YoloInfer*, const uint8_t* bgr, int w, int h, int stride,
                          YoloDet* out, int max_out); // return count
```

### 4.3 `yolo_aim`

- 源：主仓跟踪 + `AbstractMouseController` + Factory + 各后端
- 配置：JSON（字段从 `MouseControllerConfig` / `ConfigManager` 映射）
- 导出：

```c
YoloAim* yolo_aim_create(void);
void     yolo_aim_destroy(YoloAim*);
int      yolo_aim_set_config_json(YoloAim*, const char* json);
int      yolo_aim_tick(YoloAim*, const YoloDet* dets, int n,
                       const AimFrameMeta* meta, AimDebug* debug_out);
```

### 4.4 `yolo_host`（启动器，无业务 UI）

1. 读 `config/default.json`（或 `%APPDATA%/YoloAim/config.json`）
2. 加载/创建 capture + infer + aim
3. 启动 `EngineThread` 主循环
4. 启动 Web（默认 `127.0.0.1:17890`）
5. `ShellExecute` 打开浏览器（可配置 `auto_open_browser`）
6. 控制台或托盘等待退出

---

## 5. WebUI 方案

### 5.1 服务端

- 库：**cpp-httplib**（单头，MIT）放 `third_party/httplib/httplib.h`
- 可选二期：WS 用 httplib 的 WebSocket 或独立 `ws` 轮询（一期 **HTTP 轮询 200ms** 足够）

### 5.2 REST 契约（一期）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 静态 `webui/index.html` |
| GET | `/api/status` | 运行状态、FPS、检测数、PID 摘要 |
| GET | `/api/config` | 当前完整配置 JSON |
| PUT | `/api/config` | 热更新配置（部分字段即时生效） |
| POST | `/api/engine/start` | 启动主循环 |
| POST | `/api/engine/stop` | 停止主循环 |
| POST | `/api/engine/reload_model` | 重载模型 |
| GET | `/api/detections` | 最近一帧检测框（归一化） |
| GET | `/api/health` | `{ok:true}` |

**安全：** 仅绑定 `127.0.0.1`；可选 `api_token` 头校验。

### 5.3 前端页面（一期最小）

单页 `webui/index.html`：

1. **总控**：Start / Stop、运行状态、推理 FPS、捕获后端
2. **捕获**：后端 DXGI/WGC/GDI、ROI 宽高、居中开关
3. **模型**：路径、device（cpu/dml/cuda）、conf/nms、目标类
4. **瞄准**：算法类型、FOV、热键 VK、PID 主参数滑条
5. **鼠标后端**：类型下拉（WindowsAPI / MAKCU / Logi / GvInput…）
6. **实时**：最近检测数量 + 简易 canvas 框（可选）

技术：原生 HTML/CSS/JS，无构建步骤；`fetch` 调 REST。

---

## 6. 配置文件（`config/default.json`）

```json
{
  "web": {
    "host": "127.0.0.1",
    "port": 17890,
    "auto_open_browser": true,
    "token": ""
  },
  "capture": {
    "backend": "dxgi",
    "mode": "center",
    "width": 640,
    "height": 640,
    "region": { "x": 0, "y": 0, "w": 640, "h": 640 },
    "monitor_index": 0
  },
  "infer": {
    "model_path": "models/yolo.onnx",
    "device": "dml",
    "confidence": 0.5,
    "nms": 0.45,
    "input_size": 640,
    "interval_frames": 1,
    "target_classes": [0]
  },
  "aim": {
    "enabled": false,
    "hotkey_vk": 0x02,
    "fov_radius": 120,
    "algorithm": "AdvancedPID",
    "controller": "WindowsAPI",
    "pid": {}
  }
}
```

完整 PID 字段二期从主仓 `ConfigManager::configToJson` 对齐。

---

## 7. 分阶段里程碑

### M0 — 工程骨架

- [x] 建立 `standalone/` 目录
- [x] 拷贝截图源码到 `third_party/screen_capture`
- [x] `PLAN.md` / `README.md` / 默认配置 / 头文件桩 / CMake 骨架 / 最小 WebUI
- [x] 能单独 CMake 配置（可先只编 capture + host 空壳）

### M1 — 截图闭环

- [x] CMake 编出 `yolo_capture`（DLL）
- [x] `FrameSource` 封装 DXGI 居中 640 抓 BGR
- [x] `yolo_host` 循环：抓帧 → 统计 FPS → `preview_last.bmp` + `/api/preview.bmp`
- [x] Web：`/api/status` 显示 capture FPS

**验收：** 控制台打印 FPS，BMP 640×640 可见。

### M2 — 推理闭环

- [x] OBS stubs（`obs-module.h` / `obs_log` / `UNUSED_PARAMETER`）+ `/utf-8`
- [x] 主仓 `ModelYOLO` 静态链入 host（`InferEngine`）
- [x] `grab → BGR → inference → dets`；预览绿框
- [x] Web：`/api/detections` + status `infer_*`

**验收：** `infer_ok=true`；有目标时 `dets>0` 绿框；CUDA 缺 cudnn 时 CPU fallback OK。

### M3 — 瞄准 + 鼠标（部分完成）

- [x] 独立 slim `AimEngine`（WindowsAPI `SendInput` 相对移动，不链完整 AbstractMouse）
- [x] FOV 最近目标 + 热键门控 + 动态 P PID tick
- [x] status：`aim_ok / aim_hotkey_down / aim_out_*`；默认 `aim.enabled=true`
- [ ] 完整移植 Logi/MAKCU/GvInput/NtUser 后端
- [ ] JSON 配置热加载（PUT `/api/config`）

**验收：** 热键按下且 FOV 内有框时 `aim_moved` / `aim_out` 非零。

### M4 — WebUI 完整化（约 2–3 天）

- [ ] 配置页双向绑定
- [ ] Start/Stop、重载模型
- [ ] 简易检测叠加预览（base64 JPEG 节流 or 仅框坐标 canvas）
- [ ] 打包：`webui/` 相对 exe 路径加载

### M5 — 打包与硬化（约 2 天）

- [ ] 复制 ORT/OpenCV/自有 DLL 到 `dist/`
- [ ] 仅监听 localhost、可选 token
- [ ] 崩溃日志、优雅退出
- [ ]（可选）托盘图标

---

## 8. 从主仓剥离的文件清单

### 可直接引用（改日志后）

| 模块 | 路径（相对主仓） |
|------|------------------|
| 检测结构 | `src/models/Detection.h` |
| 推理 | `src/models/ModelYOLO.cpp/h` `IYoloModel.h` `Model.h` |
| 封装 | `src/YOLODetector.cpp/h` |
| GPU 预处理 | `src/models/CudaPreprocessor.*` `DmlPreprocessor.*` |
| 鼠标接口 | `src/MouseControllerInterface.hpp` `MouseControllerFactory.*` |
| 各后端 | `src/*MouseController.*` `logi_driver.*` |
| 算法 | `src/AbstractMouseController.*` `aim_controller.*` `mpid.*` `SmithPredictor.*` … |
| 跟踪 | `src/TargetTracker.*` `KalmanFilter.hpp` `HungarianAlgorithm.*` |
| 配置序列化 | `src/ConfigManager.*`（去 `obs_module_config_path`） |

### 明确不搬

| 文件 | 原因 |
|------|------|
| `plugin-main.c` | OBS 模块入口 |
| `yolo-detector-filter*.c/cpp` | OBS 滤镜上帝类 |
| `filter_properties.cpp` | OBS 属性面板 |
| `filter_rendering.cpp` | OBS 绘制 |
| `obs-utils/*` | gs_texrender 取帧 |
| `YoloAimSettingsDialog.*` | Qt；由 WebUI 替代 |
| `QtFluentWidgets` | 不再依赖 |

### 必须重写的胶水

1. 原 `video_tick` / `filter_inference` 主循环 → `src/host/EngineLoop.cpp`
2. 原 `obs_data` 设置 → `config/*.json` + REST
3. 原 `gs_*` 取帧 → `FrameSource`（本截图库）

---

## 9. CMake 策略

```cmake
# standalone/CMakeLists.txt 要点
project(yolo_aim_standalone)
option(YA_BUILD_CAPTURE_SHARED "capture as DLL" ON)
option(YA_WITH_INFER "link ONNX inference" OFF)   # M1 先 OFF
option(YA_WITH_AIM "link aim/mouse" OFF)          # M1 先 OFF
option(YA_WITH_WGC "build WGC backend" ON)

add_library(yolo_capture ...)
add_executable(yolo_host src/host/main.cpp ...)
# M2+ 再 add_library(yolo_infer) / yolo_aim
```

依赖：

- **M1：** 仅 Windows SDK（d3d11/dxgi），无 OpenCV/ORT 也可跑截图自测
- **M2+：** OpenCV、onnxruntime（复用主仓 `../onnxruntime`）、nlohmann_json
- Web：header-only httplib

生成器：VS 2022 x64；与主仓相同，CUDA 若需要继续绕开 `enable_language(CUDA)`。

---

## 10. 线程与同步约定

| 线程 | 职责 | 约束 |
|------|------|------|
| Main | 起停、信号处理 | 不跑推理 |
| Engine | capture→infer→aim | 热路径不重建 Session/Controller |
| Web | HTTP 处理 | 只读写 `shared_mutex` 保护的 Snapshot/Config |
|（可选）Infer worker | 异步推理 | 与现 filter 线程池语义对齐 |

硬约束（沿用主仓经验）：

- 控制器创建放配置变更路径，不进每帧
- 配置更新用 double-buffer 或 mutex，避免半份 JSON

---

## 11. 风险与对策

| 风险 | 对策 |
|------|------|
| DXGI 独占全屏失败 | 自动回退 WGC/GDI；Web 提示 |
| WGC 需 WinRT 初始化 | `winrt::init_apartment` 在捕获线程 |
| BGR 指针生命周期 | 文档约定 + `FrameSource` 内 clone 或环形缓冲 |
| 主仓 `obs_log` 满天飞 | 先宏劫持 `#define obs_log(level,...) ya_log(...)` |
| 坐标漂移 | 单测：ROI 中心应对准屏幕中心像素 |
| ORT DLL 搜索路径 | host 启动时 `SetDllDirectory` / 旁路拷贝 |
| 误绑 0.0.0.0 | 默认 127.0.0.1，代码写死校验 |

---

## 12. 近期执行顺序（建议你按勾选推进）

```
1. 本目录 CMake 只编 yolo_capture + yolo_host（打印 FPS）   ← 下一步编码
2. 接 OpenCV Mat + 存图验证
3. 接 YOLODetector（YA_WITH_INFER=ON）
4. 接 Aim + WindowsAPI 鼠标
5. 填满 WebUI 控件
6. 多后端鼠标 / CUDA / 打包
```

---

## 13. 验收清单（产品级）

- [ ] 无 OBS、无 Qt 运行时依赖
- [ ] 双击 `yolo_host.exe` 后浏览器可配置并 Start
- [ ] DXGI 捕获 + 推理稳定运行 ≥30 分钟
- [ ] 修改 Web 上 conf/FOV 热生效
- [ ] 停止/退出无僵尸线程、无句柄泄漏
- [ ] `dist/` 可拷贝到另一台同架构机器运行（含 ORT DLL）

---

*文档版本：2026-07-23 · 与 `standalone/` 骨架同步*
