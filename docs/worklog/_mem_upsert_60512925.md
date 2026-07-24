# obs-backgroundremoval 项目完整历史记录（至 2026-07-23）

## 一、项目概述
- 仓库: E:\obs-heji\obs-backgroundremoval
- OBS 31.x YOLO 背景移除/目标检测插件
- standalone/: 独立 exe (yolo_host.exe + WebUI)，不依赖 OBS
- 分支: feature/add-logi-driver-controller
- 用户: 小鱼（昵称），偏好直白、直接改代码

## 二、项目早期（CMake CUDA 构建修复）
### 问题：CUDA/TensorRT EP 切不回去
- CMake ENABLE_CUDA 块变空壳，HAVE_CUDA/HAVE_ONNXRUNTIME_CUDA_EP 未定义
- ORT 包被换成 DML 版，无 providers_cuda/tensorrt.lib
- 修复：CMakeLists.txt ENABLE_CUDA 块重写
  - 不调 enable_language(CUDA)（VS 生成器无 NVIDIA CUDA MSBuild toolset）
  - 改用 nvcc 直编 CudaPreprocessor.cu → .obj → target_sources
  - 定义 HAVE_CUDA、加 CUDAToolkit include、链 CUDA::cudart
  - CUDA/TRT EP 宏检测改为 EXISTS 检查 .lib 路径
  - DML 检测修复：用具体 .lib 路径避免 try_compile 崩
  - CMakePresets.json 加 DIRECTML=OFF

## 三、standalone 创建（从 OBS 插件提取独立 exe）

### 阶段 A：核心引擎
1. **EngineLoop**（替代 OBS video_tick）
   - capture → infer → track → aim → preview 全管线
   - AiMod 双线程：capture_loop 独立线程 grab + 最新帧推理
   - Hot reload: 模型/截图/配置热切换
   - start()/stop() 完整生命周期

2. **ModelYOLO 链入**
   - 共享主仓 src/models/ModelYOLO.cpp（CMake 引用不改）
   - InferEngine 封装 run_bgr() / set_thresholds() / set_target_classes()
   - 自适应版本检测（见下文详细说明）

3. **截图后端**
   - DXGI center/region/GDI/WGC
   - vendored third_party/screen_capture/

### 阶段 B：瞄准栈
4. **FullAimBridge**
   - 包装主仓 MouseControllerFactory + 8 后端
   - 5 套配置槽、5 种算法、准星检测
   - Hot config apply（连续/热键模式切换）

5. **TrackerEngine**
   - OBS filter_inference 逻辑移植
   - class 硬门 + re-ID + coast

### 阶段 C：配置
6. **ConfigStore**
   - nlohmann_json 解析
   - OBS 键对齐（conf/nms/classes/aim/vision/tracking）
   - atomic write (.tmp + rename)
   - 按需分类：reload_model / reload_capture / recreate_controller

### 阶段 D：WebUI
7. **8 页 WebUI**（对应 OBS settings_page 0-7）
   - 0 模型与检测 / 1 视觉与区域 / 2 鼠标基础 / 3 PID
   - 4 扳机 / 5 追踪与高级 / 6 预测滤波 / 7 准星
   - 热字段即时 apply，冷字段需 reload

## 四、推理后处理演进（ModelYOLO.cpp）

### 4.1 最初问题
- 用户换多个模型（cs2j0x1tf2x3t_fp16, cs2.onnx, Bing_TF_v26）
- 不同模型输出格式完全不同，手选版本会出错

### 4.2 自适应版本检测（核心改造）
按输出 shape 自动判定：
- `[1,300,6]` → end2end-NMS（xyxy+conf+cls，如 cs2/Bing_TF）
- `[1,N,5+nc]` 奇数C + `[1,3,H,W,C]` 多尺度头 → YOLOv5（obj×class）
- `[1,C,N]` channels-first → YOLOv8/v11
- 手选版本错误也会被纠正（日志打 AUTO/force）

### 4.3 三个已知模型
| 模型 | 输出 | 类型 |
|------|------|------|
| cs2j0x1tf2x3t_fp16.onnx | [1,6300,9] + 3头 | YOLOv5, FP16 |
| cs2.onnx | [1,300,6] | end2end NMS |
| Bing_TF_320_v26m | [1,300,6] | end2end NMS |

### 4.4 NMS 改造
- class-aware NMS（同类才抑制，AiMod 风格）
- 中心距合并（IoU 压不掉的贴边鬼框，0.45×对角线阈值）
- secondBest margin < 0.05 过滤噪声
- 面积/宽高比过滤
- IoU 除零守卫

### 4.5 GPU 一致性
- 强制 CPU letterbox（与解码 pad/scale 一致），避免 GPU/CPU pad 不同导致框偏移
- H2D 拷贝到 device buffer，CUDA EP 仍跑网络

## 五、截图架构（AiMod 双线程）
- capture_loop 独立线程：DXGI grab → latest_frame_
- 主线程：取最新帧推理（丢弃中间帧）
- 日志区分 cap_fps / proc_fps
- capture_fps_fixed_ (fps×10 uint32 避免 atomic<double>)

## 六、悬浮窗（FloatingPreview）
- OBS-style 置顶 Win32 HWND
- WM_PAINT 锁指针画（不 copy 8MB buffer）
- WM_CLOSE: DestroyWindow + ReleaseCapture + engine 关配置
- PeekMessage 不用 DestroyWindow 后的无效句柄
- drag_ 在 WM_CLOSE/WM_DESTROY 重置
- user_closed 标志防 re-open loop

## 七、Start/Stop 修复
- Start: infer_enabled=true + reload_model
- Stop: infer_enabled=false + infer_.unload() + 清检测 + 关悬浮窗
- running_=false 移到 join 之前（UI 立即 STOPPED）
- start() 等待 2000ms（原 500ms 假失败）

## 八、Web/配置修复
- WebServer: send() 循环短写; json_escape 加 \r\t; 路径遍历加强
- ConfigStore: rename 失败后清理 .tmp
- help-tip 仅点击感叹号显示（hover 去掉不干扰滑条）
- auto-apply 限流（拖滑条时 800ms 内不刷 toast）
- 预览 30fps 服务端 / 25fps 客户端
- 预览框内烧录 c{id} xx% 标签 + class 着色

## 九、UI 适配
- 异形屏布局（超宽/窄屏/矮屏/竖屏）
- 100dvh + 预览 max-height
- 版本选择项 + 自适应识别提示
- 设备标签修正（cuda+cpu_pre → 推理在 GPU）

## 十、Bug 修复清单（2026-07-23 批量）
| 文件 | Bug | 修复 |
|------|-----|------|
| EngineLoop.cpp | static last_seen_seq 跨 start/stop | 成员变量重置 |
| EngineLoop.hpp | atomic<double> 非 lock-free | atomic<uint32_t> fps×10 |
| EngineLoop.cpp | start() 等 500ms 假失败 | 2000ms |
| EngineLoop.cpp | running_=false 在 join 后 | 移到 join 前 |
| ModelYOLO.cpp | IoU 除零 | unionArea<=1e-6 返回0 |
| ModelYOLO.cpp | loadClassNames 覆盖 numClasses_ | 只在更大时更新 |
| FloatingPreview.cpp | WM_PAINT 8MB copy | 锁指针画 |
| FloatingPreview.cpp | WM_CLOSE 只 Hide 不 Destroy | DestroyWindow + ReleaseCapture |
| FloatingPreview.cpp | PeekMessage 用无效句柄 | 去掉 |
| FloatingPreview.cpp | drag 中句柄销毁未重置 | 重置 drag_=false |
| TrackerEngine.cpp | re-ID 中途修改 track | 先找最佳再改 |
| TrackerEngine.cpp | coast 给 x/y 加速度漂移 | 只更新 centerX/Y |
| WebServer.cpp | send() 短写 | 循环发送 |
| WebServer.cpp | json_escape 不处理 \r\t | 加转义 |
| WebServer.cpp | 路径遍历只检查 .. | 加反斜杠检查 |
| ConfigStore.cpp | rename 失败不删 .tmp | 加 fs::remove(tmp) |

## 十一、构建环境
- VS: E:\VisualStudio\chanping (MSVC 14.44)
- CUDA: D:\CUDATool (nvcc V12.5.40)
- CMake: E:/VisualStudio/chanping/Common7/IDE/.../cmake.exe
- ORT: 1.27.1 cuda12 (含 providers_cuda + providers_tensorrt)
- 编译命令:
```
taskkill //F //IM yolo_host.exe
cmd //c "E:\VisualStudio\chanping\MSBuild\Current\Bin\MSBuild.exe E:\obs-heji\obs-backgroundremoval\standalone\build\yolo_host.vcxproj /p:Configuration=RelWithDebInfo /p:Platform=x64 /m:4 /v:minimal"
```
- CMake reconfigure:
```
"E:/VisualStudio/chanping/Common7/IDE/.../cmake.exe" -S standalone -B standalone/build
```

## 十二、参考项目
- 新建文件夹(3)/AiMod/: 别人的 YOLO aim 项目（AiMod 代码）
  - yolodml/: v5/v8/v11/v12/v7/vX 独立解码器
  - cap/: DXGI/GDI/WGC/OBS 截图
  - 关键差异：3线程架构（capture/detect/display）、class-aware NMS SSE优化
  - 已按此二改 EngineLoop 双线程 + NMS 逻辑

## 十三、当前配置
- user.json: Bing_TF_v26, input_size=640, device=cuda, conf=0.52, nms=0.37
- capture: center 640x640
- infer_enabled=true, aim_enabled=true

## 十四、待做
- NMS O(n²) 优化（300+候选）
- 性能优化: preview encode throttle
- OBS 悬浮窗完全对齐
- 10分钟自动化 QA cron 验证
- CMake standalone HAVE_CUDA 预处理（DML 模式未启用）

## 附录：2026-07-24 增量（配置一致性 + 中文 UI + AiMod 对比）

# 增量记忆：上次完整历史上传之后 → 2026-07-24

> 对应会话后续工作；完整历史见记忆 `60512925` / 本文档上级 worklog。

仓库: `E:\obs-heji\obs-backgroundremoval`  
分支: `feature/add-logi-driver-controller`  
standalone: `yolo_host.exe` + WebUI  

---

## A. OBS ↔ standalone 配置一致性

### ConfigStore
- `to_json`：嵌套 JSON + 根级 OBS 扁平键双写  
- `merge_into`：`jget_*2` 同时读 OBS 名与 alias  
- 对齐键：
  - `use_gpu` ↔ `infer.device`
  - `confidence_threshold` ↔ `infer.confidence`
  - `nms_threshold` ↔ `infer.nms`
  - `input_resolution` ↔ `infer.input_size`
  - `inference_interval_frames` ↔ `infer.interval_frames`
  - `target_class`（-1=全部）↔ `target_classes`
  - `is_inferencing` ↔ `infer.enabled`
  - vision 根级：`show_detection_results`、`bbox_line_width`、`floating_*`、`show_fov` / `fov_radius` 等  

### default.json
- conf **0.5**、nms **0.45**、input **640**、`target_class: -1`  
- `use_gpu` / `device` 双写  

### WebUI
- `data-obs-alias`  
- 单类别 `target_class` 输入  
- `fillFromConfig` / `collectPatch` 读写 OBS 根级别名  

### 文档
- `docs/worklog/obs-config-parity.md`（对照表、有意差异、验收、导入方式）  

### 有意未做
- `use_gpu_texture_inference`（无 OBS 纹理句柄）  
- `bbox_color` / `fov_color`  
- `detection_smoothing_*` 未进 Tracker  
- `use_region` 默认 standalone=true（OBS 默认 false）  

---

## B. Web 全中文界面（2026-07-24）

1. 去掉全部 `span.key` 英文配置键角标；CSS `span.key { display: none }`  
2. 顶栏：启动 / 停止；运行中 / 已停止 / 离线  
3. 状态：截图/推理正常失败、后端中文、居中裁切/固定区域、设备中文、帧率/目标数/瞄准槽位热键中文  
4. 选项：设备/截图模式/算法/控制器中文；保留品牌名 CUDA、MAKCU、GHUB 等  
5. `app.js`：`applyObsHelp` 不再拼英文 key；status/preview/det 中文  
6. **事故**：脚本曾误把 `data-key`/`data-path` 译成中文 → 已全部恢复为英文内部键（页面不显示）  

---

## C. AiMod 对比（`E:\obs-heji\AiMod`）

用户体感：**Aim（AiMod）比自家项目好用很多** — 分析成立。

| 维度 | AiMod | standalone |
|------|-------|------------|
| 检测 | 版本分文件、固定流水线、稳 | 自适应多格式，易猜错 layout |
| 线程 | 截图/推理/显示三线程 | 已有 capture_loop，主环仍重 |
| 瞄准 | 简单 PID + MotionSim + 曲线 | 8 后端 × 5 算法 × 5 槽，功能远强 |
| 产品 | 小、硬编码 | Web + OBS 配置，复杂度高 |

**体感差主因：检测链路摩擦，不是瞄准缺功能。**

AiMod 要点：
- `CaptureLoop` / `DetectionLoop` / `DisplayLoop`
- `yolodml`: V5/V7/V8/V10/V11/V12/X 独立 `GenerateProposals`
- `Detect`: LetterBox → ORT → Proposals → class-aware NMS → ScaleBoxes
- 瞄准：最近目标 → MotionSimulator → pid_x/y → mouseCurve（示例甚至只打印移动）
- 截图：DXGI/GDI/WGC/OBS 网络，与 standalone 同源思路

**建议落地「Aim 模式」**：锁死解码路径 + 最简目标选择 + 单套 PID 预设 + 推理与预览隔离。

---

## D. 未做 / 下一步候选

- 落地 Aim 模式专用管道  
- detection_smoothing 进 Tracker  
- bbox/fov 颜色映射  
- NMS O(n²) 优化、preview throttle  
- 推理与业务环三线程彻底分离  

---

## E. 编译与产物

```text
MSBuild standalone/build/yolo_host.vcxproj RelWithDebInfo x64
产物: standalone/build/RelWithDebInfo/yolo_host.exe
Web: 需将 standalone/webui/ 同步到 build/RelWithDebInfo/webui/
```


## 附录：用户体感与 Aim 差距（2026-07-24）
- 用户明确：E:\obs-heji\AiMod 体感比 standalone 好很多。
- 根因：AiMod 检测路径短专稳（分版本解码 + 截图/推理/显示三线程）；standalone 瞄准功能更强但检测/链路摩擦拖累体感。
- 建议：Aim 模式 = 锁死解码路径 + 最简目标选择 + 单套 PID 预设 + 推理与预览隔离。
- 本地备份：docs/worklog/memory-delta-2026-07-24.md 、docs/worklog/obs-config-parity.md
- 增量记忆 ID：404a0318-0000-4000-8000-000000000000
