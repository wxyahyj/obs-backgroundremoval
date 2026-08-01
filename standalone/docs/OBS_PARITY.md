# Standalone ↔ OBS 插件功能对照

## 已对齐（内核）

| 能力 | OBS | Standalone |
|------|-----|------------|
| 截图 | DXGI/WGC/GDI 区域 | `FrameSource` 同后端 + `reload_capture` |
| YOLO 推理 | `ModelYOLO` ORT | 同源码链入 host + `reload_model` |
| 多目标跟踪 | Hungarian 融合距离 + re-id | `TrackerEngine` |
| 5 套瞄准配置 | `mouse_config_%d` / `enable_config_%d` | `FullAimSettings::profiles[5]` + OBS 键 |
| 5 种算法 | Advanced/External/Aim/Slew/Adaptive | `AlgorithmType` 同枚举 + 全局 stamp |
| 8 鼠标后端 | Factory 全类型 | 同 Factory 源 + `/api/controller/test` |
| FOV / 动态 FOV | filter 动态收缩 | `FullAimBridge` + status `fov_px` |
| 扳机 / 后座 | `MouseControllerConfig` | 同结构热更新（ConfigStore） |
| 准星 aim origin | `CrosshairDetector` | 每 N 帧 HSV 扫描 → `setAimOrigin` |
| 配置键 | `filter_properties` OBS keys | `ConfigStore` + `aim.obs_keys` 扁平表 |
| Web 0–7 页 | Qt Fluent `settings_page` | `webui` 侧栏 + data-key 表单 |
| 用户配置持久化 | OBS 场景/filter 数据 | `config/user.json`（PUT 后写盘） |

## 热路径（与 OBS 一致）

```
grab → infer(interval) → TrackerEngine.update
  → [optional] CrosshairDetector.detect → aim_origin
  → FullAimBridge.tick:
       select_active_slot (continuous > hotkey)
       recreate_if_needed (仅 type/port 变)
       updateConfig / setDetectionsWithFrameSize
       setInferenceTimeMs / setAimOrigin / tick
  → preview (bbox + FOV)
```

## 配置 / API

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/api/config` | 全文档（含 `aim.obs_keys`、5 configs、crosshair） |
| PUT | `/api/config` | 合并 + 热 apply + 写 `config/user.json` |
| POST | `/api/engine/start\|stop` | 启停 |
| POST | `/api/engine/reload_model` | 冷重载 ORT session |
| POST | `/api/engine/reload_capture` | 冷重开截图后端/尺寸 |
| POST | `/api/controller/test` | MAKCU/Logi 等连通 |
| GET | `/api/status` | FPS / slot / controller / algorithm |
| GET | `/api/detections` | bbox + track_id |
| GET | `/api/preview.bmp` | 预览 |

启动时优先加载 `config/user.json`，否则 `default.json`。

## 已知差异

- 缺 `cudnn64_9.dll` 时 CUDA EP 日志可能“成功”但回落 CPU 路径（与 OBS 相同 ORT 行为）。
- 准星拾色 UI（屏幕点选）未做；HSV/manual RGB + 检测已通。
- OBS 插件工程未改；standalone 只读引用 parent `.cpp`。
- Qt Fluent 原生窗不在本期（Web 0–7 覆盖）。

## 构建

```bash
cmake -S standalone -B standalone/build -G "Visual Studio 17 2022" -A x64
cmake --build standalone/build --config RelWithDebInfo --target yolo_host
standalone/build/RelWithDebInfo/yolo_host.exe
```

`YA_WITH_AIM=ON`（默认）链接完整瞄准栈。
