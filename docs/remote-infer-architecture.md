# 远程推理方案

## 架构

```
PC (standalone yolo_host.exe)              Android 手机
┌─────────────────────────┐                ┌──────────────────┐
│ DXGI 截图 → ROI → JPEG  │───TCP发图────  │ TCP Server:9999  │
│                         │                │ ↕ JPEG解码       │
│ RemoteInferClient      │◄───TCP收JSON──  │ ↕ ncnn 推理      │
│                         │                │ ↕ JSON回传       │
│ FullAimBridge (瞄准)    │                └──────────────────┘
│ WebUI (画框/预览)       │
└─────────────────────────┘
```

## PC 端文件

```
standalone/src/net/
├── RemoteInferClient.hpp   — TCP 客户端，发图收检测结果
└── RemoteInferClient.cpp   — 实现：TCP连接、JPEG编码、JSON解析、断线重连
```

需要额外依赖：`libjpeg-turbo`（或 `stb_image_write`）用于 BGR→JPEG。

## Android 端文件

```
app/src/main/java/com/littlefish/aimbot/remote/
├── RemoteInferServer.kt     — TCP 服务端核心，收图→推理→回传
├── RemoteInferService.kt    — Android 前台 Service（保活 + WiFi锁）
├── RemoteConfig.kt          — 配置持久化（SharedPreferences）
├── RemoteSettingsActivity.kt— 设置界面
└── (复用) inference/JniCallBack.kt  — 直接调用现有 ncnn/GPU 推理
```

## 协议

### PC → 手机 (推理请求)
```
[4B frame_len][JPEG bytes]
```

### 手机 → PC (推理响应)
```
[4B json_len][JSON bytes]
JSON: {"dets":[{"c":0,"s":0.92,"x":0.3,"y":0.2,"w":0.1,"h":0.3},...],"ms":12,"seq":142,"err":null}
```

### 手机 → PC (心跳，每秒2次)
```
[4B 0xFFFFFFFF][1B type=0x03][4B seq][8B ts_ns]
```

## 自定义模型

Android 端设置界面支持：
- 文件选择器导入 `.tflite` / `.param` + `.bin`
- 自动识别 ncnn 模型 pair（`.param` 和同名 `.bin`）
- 保存到 `filesDir/models/` 目录
- 下次启动自动加载

## 编译

### Android
```bash
cd Auto-aim_android-yolo
./gradlew assembleRelease  # 或 Android Studio 直接 Build
```

### PC
```bash
# 安装 libjpeg-turbo，并在 CMakeLists.txt 中链接
cmake --build standalone/build --config RelWithDebInfo
```
