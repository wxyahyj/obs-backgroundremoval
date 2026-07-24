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
