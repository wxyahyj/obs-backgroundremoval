# core（M2）

此处将接入主仓推理：

- `../../src/models/ModelYOLO.*`
- `../../src/YOLODetector.*`
- 日志宏替换 `obs_log` → `ya_log`

CMake 开关：`-DYA_WITH_INFER=ON`
