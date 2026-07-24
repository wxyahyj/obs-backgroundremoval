# OBS vs Standalone 全范围功能扫描

扫描时间：2026-07-23  
方法：对照 OBS `filter_inference.cpp` / `yolo-detector-filter.cpp` 源码 + standalone 实现 + 静态契约脚本 `scripts/qa/obs_parity_check.py`

---

## 1. OBS 检测区域推理（权威路径）

| 步骤 | OBS 行为 | 文件 |
|------|----------|------|
| 1 | 全源帧提交到 ring buffer，并记录 `inputCropX/Y/W/H` | `yolo-detector-filter.cpp` video path |
| 2 | `use_region=true` 时 crop = `(regionX,regionY,regionW,regionH)`；否则整帧 | 同上 |
| 3 | 推理线程 **只对 ROI** 做 `model->inference(inferenceFrame)` | `filter_inference.cpp` |
| 4 | 检测框 ROI 归一化 → 像素 → **再归一化到全帧** | `filter_inference.cpp` remap 块 |
| 5 | `setDetectionsWithFrameSize(dets, fullW, fullH, cropX, cropY)` | aim 热路径 |

OBS 默认 `use_region=false` → 整源推理；开启后 = **检测区域内推理**。

---

## 2. 本轮已修（standalone）

### 2.1 真·全屏误解 / ROI 配置断链（高优先级）

| ID | 问题 | 修复 |
|----|------|------|
| `config_region_x_bug` | WebUI `data-path="capture.region_x"` 写入扁平键，ConfigStore **只读** `capture.region.x` → region 坐标永不生效 | `ConfigStore` 同时接受 `region_x/y/width/height` + nested `region` |
| `use_region` 缺失 | 无 OBS 同名开关 | 配置/序列化/status/UI 增加 `use_region` |
| ROI 推理语义 | center 已是 DXGI 裁切，但 full 模式无法软裁；缺 OBS remap | `resolve_infer_roi` + `remap_dets_roi_to_full` + `open_capture_obs_style` |

**当前语义（对齐 OBS）**

- `mode=center`（默认）：屏幕正中 `width×height` DXGI 裁切 → **整包即检测区域**，非全屏推理  
- `mode=region`：固定 `(region_x,region_y,region_width×region_height)` DXGI 裁切  
- `mode=full` + `use_region=true`：全屏捕获 → **软件 ROI 裁切 → YOLO → 框映回全帧**（与 OBS 同公式）  
- `mode=full` + `use_region=false`：全帧推理（OBS 默认关区域时）

### 2.2 涉及文件

- `standalone/src/host/EngineLoop.hpp` / `EngineLoop.cpp`
- `standalone/src/core/ConfigStore.cpp`
- `standalone/src/web/WebServer.cpp`
- `standalone/webui/index.html`
- `standalone/config/default.json`
- `scripts/qa/obs_parity_check.py` + 每 10 分钟 cron

---

## 3. 功能对照表（正确性）

| 能力 | OBS | Standalone | 状态 |
|------|-----|------------|------|
| 区域 ROI 推理 | `use_region` + region_* | center/region DXGI + full+use_region 软裁 | **已对齐**（本轮） |
| letterbox + 归一化框 | ModelYOLO | 同 ModelYOLO | OK（共享） |
| 输出 layout `[1,N,C]` | auto detect | 已 auto box-major | OK |
| 跟踪 Hungarian/re-id | filter 内 | TrackerEngine | 基本对齐 |
| 5 aim 槽 + hotkey/continuous | 有 | FullAimBridge | 基本对齐 |
| 5 算法 | 有 | 有 | 需真机细验 PID 数值 |
| 8 鼠标后端 | Factory | 同链 | 无设备时 create 失败属预期 |
| 扳机/后座 | 有 | 配置进 MC | 字段在；运行时依赖后端 |
| 预测/滤波 | Derivative/Smith/… | 配置透传 | 字段在 |
| 准星 | CrosshairDetector | EngineLoop 调 | 基本对齐 |
| FOV 动/静 | 有 | preview + aim | 基本对齐 |
| 预览不写盘 | OBS 悬浮窗 | memory BMP | OK |
| Web 0–7 页 | settings_page | index.html | 控件在；死字段见下 |
| GPU 预处理 HAVE_CUDA | 可选 | 日志 `cuda+cpu_pre` | **部分**（EP=CUDA，预处理仍 CPU） |

---

## 4. 仍开放 / 风险

| 级别 | ID | 说明 |
|------|-----|------|
| high | `det_count_noise` | 日志曾见 200–600 dets：conf/nms/NMS 质量或模型；需对照 OBS 同模型同 conf |
| med | `dxgi_fps` | 偶发 cap_fps 偏低；CaptureBGR 忙等 + Query 同步路径 |
| med | `gpu_preprocess` | `HAVE_CUDA` 未定义 → 预处理在 CPU |
| med | `cropOffset_unused` | AbstractMouseController 存 cropOffset 但选目标时几乎不用（OBS 在 remap 后 crop 常为 0，行为一致但脆弱） |
| low | Web 部分 vision/export 路径 | 已接线但导出文件路径 UX 弱 |
| low | Infinite Grid / React | 用户明确延后 |

---

## 5. 定时任务

- Automation id: `automation-f10cd1a3-ba5a-4270-92d3-aa239768a1d3`
- Cron: `*/10 * * * *`（本地时区每 10 分钟）
- 动作：跑 `scripts/qa/obs_parity_check.py`，写  
  - `.agent/logs/obs_parity_latest.json`  
  - `docs/worklog/obs-parity-scan.md`  
  fail 时按 OBS-first 规则修 standalone

---

## 6. 验收建议

1. 启动 `yolo_host`，确认日志：`open_center … 推理在检测区域 ROI` 且 `size=W×H` ≠ 全屏分辨率。  
2. Web status：`capture_mode=center`，`origin` 为屏幕中心偏移，`use_region=true`。  
3. 预览框应对齐裁切画面内目标，不应“像铺满整桌面坐标系”。  
4. 切 `mode=full` + `use_region` + region 640×640 中心：infer 只吃 ROI，框在全屏预览中落在中心区域。  
5. `python scripts/qa/obs_parity_check.py` → `status=ok|warn`，无 `region`/`coords` error。
