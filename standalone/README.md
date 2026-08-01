# YOLO Aim Standalone

独立启动器 + DLL 组合，**不依赖 OBS**。UI 为本地 WebUI。

## 结构

- `third_party/screen_capture` — DXGI / WGC / GDI 高性能截图（已 vendored）
- `src/host` — 启动器
- `src/web` — 本地 HTTP API
- `webui` — 前端静态页
- `PLAN.md` — 完整实施计划

## 快速构建（M1：仅捕获）

```powershell
cd standalone
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config RelWithDebInfo --parallel
```

运行：

```powershell
.\build\RelWithDebInfo\yolo_host.exe
# 浏览器打开 http://127.0.0.1:17890
```

## 配置

见 `config/default.json`。

## 与主仓关系

推理/瞄准源码在后续里程碑通过 CMake 引用 `../src`；详见 `PLAN.md`。
