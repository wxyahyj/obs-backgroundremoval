# Standalone HTTP API

Base: `http://127.0.0.1:17890`（仅本机）

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/api/health` | — | `{"ok":true}` |
| GET | `/api/status` | — | 运行态 + aim/controller 字段 |
| GET | `/api/detections` | — | `{count, items[{class_id,confidence,x,y,w,h,cx,cy,track_id}]}` |
| GET | `/api/preview.bmp` | — | BMP 预览 |
| GET | `/api/config` | — | 全量 OBS 键配置文档 |
| PUT/POST | `/api/config` | JSON patch | `{"ok":true,"config":...}` 并写 `config/user.json` |
| POST | `/api/engine/start` | — | `{"ok":true}` |
| POST | `/api/engine/stop` | — | `{"ok":true}` |
| POST | `/api/engine/reload_model` | — | 冷重载 ORT（需引擎 running） |
| POST | `/api/engine/reload_capture` | — | 冷重开截图（需引擎 running） |
| POST | `/api/controller/test` | `{controller,makcuPort,makcuBaudRate,logiDriverType}` | 后端连通测试 |
| GET | `/` | — | `webui/index.html` |

## status 字段（节选）

```json
{
  "running": true,
  "capture_ok": true,
  "infer_ok": true,
  "aim_ok": true,
  "capture_fps": 60.0,
  "infer_fps": 30.0,
  "infer_ms": 12.5,
  "last_det_count": 2,
  "active_slot": 0,
  "controller_type": 0,
  "controller": "WindowsAPI",
  "algorithm": 0,
  "algorithm_name": "AdvancedPID",
  "fov_px": 120,
  "backend": "dxgi",
  "last_error": ""
}
```

## 配置文档形状

顶层：`capture` / `infer` / `engine` / `tracking` / `aim` / `prediction` / `crosshair` / `vision`。

- `aim.configs[0..4]`：每槽 OBS 键（`p_min`、`auto_trigger_group`、`controller_type`…）
- `aim.obs_keys`：扁平 `enable_config_0` / `p_min_0` / …
- `aim.algorithm_type_global`、`external_kp_x`、FOV、neural 等全局项
- `crosshair.*`：准星 HSV / interval / search_radius

PUT 支持：嵌套文档、扁平 thin 键、以及 OBS 后缀键（`p_min_0`）。

## 持久化

- 启动优先：`config/user.json` → 否则 `default.json`
- 每次成功 PUT `/api/config` 写入 exe 旁 `config/user.json`
