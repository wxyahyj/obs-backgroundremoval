# OBS ↔ Standalone 配置一致性

> 生成于 2026-07-23。目标：WebUI / `user.json` / OBS 滤镜设置可互相导入关键字段。

## 1. 命名策略

Standalone 同时接受：

| 层 | 说明 |
|----|------|
| **嵌套 JSON** | `infer.confidence`、`capture.mode`、`aim.fov_radius` |
| **OBS 扁平键** | 根级 `confidence_threshold`、`use_gpu`、`inference_interval_frames` |
| **双写导出** | `ConfigStore::to_json` 同时写嵌套 + 根级 OBS 别名，便于对照 / 迁移 |

导入时：`jget_float2` / `jget_int2` / `jget_str2` 优先 OBS 名，再读 alias。

## 2. 推理 / 模型（page 0）

| OBS 键 | Standalone | 默认 (OBS) | 默认 (standalone default.json) | 备注 |
|--------|------------|------------|--------------------------------|------|
| `model_path` | `infer.model_path` + root | `""` | `models/yolo.onnx` | 空字符串不覆盖已有路径 |
| `model_version` | `infer.model_version` | YOLOv8(=1) | `0` | **加载后 shape 可自动改写**（v5/end2end） |
| `use_gpu` | `infer.device` / `use_gpu` | `cpu` | `cuda` | 值：`cpu`/`cuda`/`dml`/`tensorrt` |
| `input_resolution` | `infer.input_size` / `input_resolution` | `640` | `640` | 以模型实际输入为准 |
| `num_threads` | `infer.num_threads` | `4` | `4` | |
| `confidence_threshold` | `infer.confidence` | `0.5` | `0.5` | 已对齐 |
| `nms_threshold` | `infer.nms` | `0.45` | `0.45` | end2end 模型内部已 NMS，仍可读 |
| `target_class` | `infer.target_class` | `-1` | `-1` | **-1=全部**；单 id 写入 `target_classes` |
| `target_classes_text` | `infer.target_classes_text` | `""` | `""` | 优先于单类别 |
| `inference_interval_frames` | `infer.interval_frames` | `1` | `1` | |
| `is_inferencing` | `infer.enabled` / `is_inferencing` | `false` | `true` | standalone 无滤镜「开关」时默认开推理 |
| `use_gpu_texture_inference` | — | `false` | — | **未实现**（无 OBS 纹理源） |

## 3. 区域 / 视觉

| OBS 键 | Standalone | 默认 OBS | 默认 standalone | 备注 |
|--------|------------|----------|-----------------|------|
| `use_region` | `capture.use_region` + root | `false` | `true` | standalone 默认 center ROI |
| `region_x/y/width/height` | 同名 + nested `region` | 0,0,640,480 | 0,0,640,640 | center 时与 width/height 同步 |
| `show_detection_results` | `vision.*` + root | `true` | (vision 卡) | |
| `bbox_line_width` | `vision.bbox_line_width` | `2` | `2` | |
| `label_font_scale` | `vision.label_font_scale` | `0.35` | 有字段 | |
| `export_coordinates` | `vision.export_coordinates` | `false` | | |
| `coordinate_output_path` | `vision.coordinate_output_path` | `""` | | |
| `show_floating_window` | `vision.show_floating_window` | `false` | `false` | Win32 悬浮窗 |
| `floating_window_width/height` | 同 | 640/480 | 640/480 | |
| `show_track_id_in_floating_window` | 同 | `false` | | |
| `show_fov` / `fov_radius` / … | `aim` + `vision` + root | 见 OBS | Web page1 | 双写 |
| `bbox_color` / `fov_color` | — | 颜色 int | — | **未映射**（Web 用 CSS 着色） |
| `detection_smoothing_*` | — | false / 0.3 | — | **未接线**到 TrackerEngine |

## 4. 跟踪 (page 5)

| OBS 键 | Standalone `tracking.*` |
|--------|-------------------------|
| `iou_threshold` | ✓ |
| `tracking_weight_*` | ✓ (+ weight_* 别名) |
| `use_kalman_tracker` | ✓ `use_kalman` |
| `kalman_generate_threshold` | ✓ |
| `kalman_terminate_count` | ✓（可驱动 max_lost） |
| `show_kalman_predictions` | ✓ 存配置 |
| `show_kalman_trajectories` | ✓ 存配置 |
| `max_lost_frames` / re-id | ✓ standalone 扩展 |

## 5. 瞄准 / 鼠标 / 预测

- 5 配置槽：`enable_config_N`、`continuous_aim_N`、`hotkey_N`、`controller_type_N`、PID/扳机/后座/Smith/Slew/Bezier/Ghost 等  
- ConfigStore `apply_obs_slot_keys` + `obs_keys` 扁平导出  
- 全局：`algorithm_type_global`、`external_*`、`aim_*`、`mouse_config_select`  
- 详见 `ConfigStore.cpp` 内 `profile_to_json` / `apply_profile_object`

**仍缺 / 仅 OBS UI：**

- `show_pid_debug_window`（standalone 无独立 PID 调试窗）  
- `test_makcu_connection`（有 `/api/controller/test`，非同名键）  
- 组名键 `*_group`（UI 折叠用，无业务状态）  
- `motion_sim_*` 部分若未全部进 Web 页，以 ConfigStore obs 扁平为准

## 6. 导入 OBS 设置的推荐方式

1. 从 OBS 滤镜导出 / 复制 `settings` JSON（扁平键）。  
2. `PUT /api/config` body 为该扁平对象，或包在 `infer`/`aim` 下。  
3. 冷字段（`model_path`/`device`/`input_resolution`）会标 reload；调用  
   `POST /api/engine/reload_model`。  
4. 区域：`use_region` + `region_*` 或 `capture.mode=center|region|full`。

## 7. 有意差异（不是漏实现）

| 项 | OBS | Standalone |
|----|-----|------------|
| 视频源 | scene/source 全帧 + 可选 crop | DXGI **center/region 裁切截图** |
| `is_inferencing` 默认 | false（手动开） | 常 true（host 启动即跑） |
| `use_region` 默认 | false | true（检测区） |
| GPU 纹理推理 | 有 | 无（无 OBS 纹理句柄） |
| 悬浮窗 | OBS 滤镜 HWND | `FloatingPreview` 同源语义 |

## 8. 验收清单

- [x] conf/nms/input/device/interval 双名读写  
- [x] `target_class` / `target_classes_text`  
- [x] `is_inferencing` ↔ `infer.enabled`  
- [x] region + floating + FOV 根级导出  
- [ ] bbox/fov 颜色 int  
- [ ] detection_smoothing 进跟踪  
- [ ] OBS 完整 JSON 一键导入 E2E 测试  

## 9. 相关代码

- `standalone/src/core/ConfigStore.cpp` — `to_json` / `merge_into`  
- `standalone/config/default.json`  
- `standalone/webui/index.html` — `data-path` / `data-obs-alias`  
- OBS: `src/yolo-detector-filter.cpp` defaults ~2040+  
