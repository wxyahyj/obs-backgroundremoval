# Book-Driven Function/Performance Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 UI 里已有但未接线的 OneEuro/检测平滑接到主链路，并继续按跟踪/滤波理论做功能与性能优化。

**Architecture:** OneEuro 滤的是 aim 误差（像素），在 `AbstractMouseController::tick` 算完 `errorX/Y` 后、进 PID/IMM 前；检测框 EMA 在 `filter_inference` 关联完成后对 `trackedDetections` 中心/框做可选平滑。IMM CA/CT 预测改稀疏闭式，减热路径矩阵三重循环。配置走 5 槽 `use_one_euro_filter_%d` 与全局 `detection_smoothing_*`。

**Tech Stack:** C++17, OBS plugin, OpenCV, existing OneEuroFilter/IMMFilter/Hungarian

---

### Task 1: Config fields for OneEuro (5 slots)

**Files:**
- Modify: `src/MouseControllerInterface.hpp` (after immActiveModels)
- Modify: `src/yolo_detector_filter.h` MouseControlConfig (after immActiveModels)
- Modify: `src/yolo-detector-filter.cpp` local MouseControlConfig members + ctor defaults

- [ ] **Step 1: Add to MouseControllerInterface::Config**

```cpp
// OneEuro 误差滤波（压检测/关联抖动，快移时自动提高截止频率）
bool useOneEuroFilter = false;
float oneEuroMinCutoff = 1.0f;
float oneEuroBeta = 0.007f;
float oneEuroDCutoff = 1.0f;
```

- [ ] **Step 2: Same fields on yolo_detector_filter.h MouseControlConfig**

- [ ] **Step 3: Same on cpp-local MouseControlConfig + ctor defaults**

```cpp
useOneEuroFilter = false;
oneEuroMinCutoff = 1.0f;
oneEuroBeta = 0.007f;
oneEuroDCutoff = 1.0f;
```

---

### Task 2: Read/defaults/sync OneEuro settings

**Files:**
- Modify: `src/yolo-detector-filter.cpp` (defaults loop ~2210, read loop ~2816, mcConfig sync ~5773)

- [ ] **Step 1: Defaults in 5-config loop (after IMM defaults)**

```cpp
snprintf(propName, sizeof(propName), "use_one_euro_filter_%d", i);
obs_data_set_default_bool(settings, propName, false);
snprintf(propName, sizeof(propName), "one_euro_min_cutoff_%d", i);
obs_data_set_default_double(settings, propName, 1.0);
snprintf(propName, sizeof(propName), "one_euro_beta_%d", i);
obs_data_set_default_double(settings, propName, 0.007);
snprintf(propName, sizeof(propName), "one_euro_d_cutoff_%d", i);
obs_data_set_default_double(settings, propName, 1.0);
```

- [ ] **Step 2: Read after IMM reads**

```cpp
snprintf(propName, sizeof(propName), "use_one_euro_filter_%d", i);
tf->mouseConfigs[i].useOneEuroFilter = obs_data_get_bool(settings, propName);
snprintf(propName, sizeof(propName), "one_euro_min_cutoff_%d", i);
tf->mouseConfigs[i].oneEuroMinCutoff = (float)obs_data_get_double(settings, propName);
snprintf(propName, sizeof(propName), "one_euro_beta_%d", i);
tf->mouseConfigs[i].oneEuroBeta = (float)obs_data_get_double(settings, propName);
snprintf(propName, sizeof(propName), "one_euro_d_cutoff_%d", i);
tf->mouseConfigs[i].oneEuroDCutoff = (float)obs_data_get_double(settings, propName);
```

- [ ] **Step 3: Sync to mcConfig**

```cpp
mcConfig.useOneEuroFilter = cfg.useOneEuroFilter;
mcConfig.oneEuroMinCutoff = cfg.oneEuroMinCutoff;
mcConfig.oneEuroBeta = cfg.oneEuroBeta;
mcConfig.oneEuroDCutoff = cfg.oneEuroDCutoff;
```

Note: Qt dialog already uses keys `use_one_euro_filter_%1` etc. Keys must match.

---

### Task 3: Wire OneEuro into AbstractMouseController

**Files:**
- Modify: `src/AbstractMouseController.hpp`
- Modify: `src/AbstractMouseController.cpp` (error calc ~478, resetPidState ~1527)

- [ ] **Step 1: Include + members**

```cpp
#include "OneEuroFilter.hpp"
// members:
OneEuroFilter oneEuroX_;
OneEuroFilter oneEuroY_;
int oneEuroLockedTrackId_ = -1;
```

- [ ] **Step 2: After raw errorX/Y computed, before deadzone**

```cpp
if (config.useOneEuroFilter) {
    if (lockedTrackId != oneEuroLockedTrackId_) {
        oneEuroX_.reset();
        oneEuroY_.reset();
        oneEuroLockedTrackId_ = lockedTrackId;
    }
    oneEuroX_.setMinCutoff(config.oneEuroMinCutoff);
    oneEuroX_.setBeta(config.oneEuroBeta);
    oneEuroX_.setDCutoff(config.oneEuroDCutoff);
    oneEuroY_.setMinCutoff(config.oneEuroMinCutoff);
    oneEuroY_.setBeta(config.oneEuroBeta);
    oneEuroY_.setDCutoff(config.oneEuroDCutoff);
    float dtEuro = (deltaTime > 1e-4f) ? deltaTime : (1.0f / 60.0f);
    errorX = oneEuroX_.filter(errorX, dtEuro);
    errorY = oneEuroY_.filter(errorY, dtEuro);
}
```

- [ ] **Step 3: resetPidState also reset OneEuro**

```cpp
oneEuroX_.reset();
oneEuroY_.reset();
oneEuroLockedTrackId_ = -1;
```

---

### Task 4: Detection box EMA smoothing (global)

**Files:**
- Modify: `src/yolo_detector_filter.h` global filter fields (near tracking params)
- Modify: `src/yolo-detector-filter.cpp` defaults + update read
- Modify: `src/filter_inference.cpp` after trackedDetections built

- [ ] **Step 1: Global fields on yolo_detector_filter**

```cpp
bool detectionSmoothingEnabled = false;
float detectionSmoothingAlpha = 0.3f;
```

- [ ] **Step 2: Defaults + read**

```cpp
obs_data_set_default_bool(settings, "detection_smoothing_enabled", false);
obs_data_set_default_double(settings, "detection_smoothing_alpha", 0.3);
// update:
tf->detectionSmoothingEnabled = obs_data_get_bool(settings, "detection_smoothing_enabled");
tf->detectionSmoothingAlpha = (float)obs_data_get_double(settings, "detection_smoothing_alpha");
```

- [ ] **Step 3: After Hungarian path builds trackedDetections (non-Kalman), optional blend with previous track**

When matching succeeded, already have new det. For coast tracks leave as-is.
For matched dets, blend center with previous track position:

```cpp
// after assignment match, before push:
if (filter->detectionSmoothingEnabled) {
    float a = std::clamp(filter->detectionSmoothingAlpha, 0.01f, 1.0f);
    // blend new measurement toward previous track to reduce jitter
    // center = a * new + (1-a) * old
    newDetections[i].centerX = a * newDetections[i].centerX + (1.0f - a) * trackedTargets[j].centerX;
    newDetections[i].centerY = a * newDetections[i].centerY + (1.0f - a) * trackedTargets[j].centerY;
    newDetections[i].x = a * newDetections[i].x + (1.0f - a) * trackedTargets[j].x;
    newDetections[i].y = a * newDetections[i].y + (1.0f - a) * trackedTargets[j].y;
    newDetections[i].width = a * newDetections[i].width + (1.0f - a) * trackedTargets[j].width;
    newDetections[i].height = a * newDetections[i].height + (1.0f - a) * trackedTargets[j].height;
}
```

Note: alpha meaning in UI is often "smooth strength". Use `a` as measurement trust: higher alpha = more follow detection (less lag). Default 0.3 = stronger smooth. Document in UI already.

- [ ] **Step 4: Kalman path already filters boxes; skip extra EMA when useKalmanTracker**

---

### Task 5: IMM predictCA sparse closed form

**Files:**
- Modify: `src/IMMFilter.hpp` predictCA

- [ ] **Step 1: Replace O(n^3) CA predict with blockwise F*x and F*P*F'+Q**

CA state `[x,vx,ax,y,vy,ay]`. Two independent 3-state blocks.

```cpp
// for each axis block b=0 (x) and b=3 (y):
// x' = x + vx*dt + 0.5*ax*dt^2
// vx' = vx + ax*dt
// ax' = ax
// Then update 3x3 P block with constant-acceleration discrete Q already coded.
```

Keep existing Qacc construction. Avoid full 6x6 triple loops for state; for P can still use temp matrix but prefer per-block 3x3.

---

### Task 6: Smith diag log downshift (perf)

**Files:**
- Modify: `src/AbstractMouseController.cpp` Smith diagnosis (~587-600)

- [ ] **Step 1: Change smith diag from LOG_INFO every 60 frames to LOG_DEBUG every 120**

Same pattern as AdvancedPID log change already done.

---

### Task 7: Build + commit

**Files:** all above

- [ ] **Step 1: Build**

```powershell
Set-Location E:\obs-heji\obs-backgroundremoval
& "E:\VisualStudio\chanping\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build_x64_local --config RelWithDebInfo --parallel
```

Expected: 0 errors, DLL produced.

- [ ] **Step 2: Commit + push**

```bash
git add src/MouseControllerInterface.hpp src/yolo_detector_filter.h src/yolo-detector-filter.cpp src/AbstractMouseController.hpp src/AbstractMouseController.cpp src/filter_inference.cpp src/IMMFilter.hpp
git commit -m "feat: wire OneEuro + detection EMA; sparse IMM CA; less smith log"
git push
```

---

## Self-Review

1. **Spec coverage:** OneEuro wiring, detection smoothing wiring, IMM perf, log perf — all tasked.
2. **No placeholders:** code snippets concrete; keys match Qt dialog.
3. **Types:** useOneEuroFilter/bool, floats match dialog spin ranges.
4. **Out of scope this plan:** full JPDA/MHT, PHD, wiring TargetTracker class (still duplicate of filter_inference).

## Expected user-visible result

- Qt “启用OneEuro滤波” 真正影响 aim 误差
- “启用检测框平滑” 在非 Kalman 路径真正平滑框
- IMM 开时 CPU 略降（CA 预测更轻）
- Smith 诊断不再刷 LOG_INFO
