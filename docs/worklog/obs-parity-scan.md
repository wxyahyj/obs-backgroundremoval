# OBS ↔ standalone parity worklog

## OBS parity scan — 2026-08-02T02:01:17+08:00

- **status**: `ok` (errors=0 warns=0 infos=1)
- host: `http://127.0.0.1:17890` offline=False

- capture: mode=None size=NonexNone use_region=True origin=[960, 480] fps=44.61924833720174
- infer: ok=True device=cuda ms=24.3796 dets=None

| level | id | msg |
|-------|----|-----|
| info | `static_contract_ok` | OBS filter_inference ROI+remap patterns + standalone EngineLoop/ConfigStore hooks present |


## OBS parity scan — 2026-08-02T02:00:15+08:00

- **status**: `fail` (errors=2 warns=12 infos=1)
- host: `http://127.0.0.1:17890` offline=False

- capture: mode=None size=NonexNone use_region=True origin=[960, 480] fps=44.450373380639185
- infer: ok=True device=cuda ms=23.3288 dets=None

| level | id | msg |
|-------|----|-----|
| info | `static_contract_ok` | OBS filter_inference ROI+remap patterns + standalone EngineLoop/ConfigStore hooks present |
| error | `region_keys_missing` | capture.region_x / capture.region missing — UI cannot set ROI |
| warn | `use_region_missing` | use_region key missing from config document (OBS page1) |
| warn | `missing_capture` | config missing top-level 'capture' |
| warn | `missing_infer` | config missing top-level 'infer' |
| warn | `missing_aim` | config missing top-level 'aim' |
| warn | `missing_tracking` | config missing top-level 'tracking' |
| warn | `missing_vision` | config missing top-level 'vision' |
| warn | `missing_prediction` | config missing top-level 'prediction' |
| warn | `missing_crosshair` | config missing top-level 'crosshair' |
| error | `aim_slots` | aim.configs expected 5 slots, got NoneType len=n/a |
| warn | `infer_confidence` | infer.confidence missing |
| warn | `infer_nms` | infer.nms missing |
| warn | `infer_model_path` | infer.model_path missing |
| warn | `infer_device` | infer.device missing |


## OBS parity scan — 2026-08-02T01:59:02+08:00

- **status**: `fail` (errors=3 warns=12 infos=1)
- host: `http://127.0.0.1:17890` offline=False

- capture: mode=None size=NonexNone use_region=None origin=None fps=None
- infer: ok=None device=None ms=27.2443 dets=None

| level | id | msg |
|-------|----|-----|
| info | `static_contract_ok` | OBS filter_inference ROI+remap patterns + standalone EngineLoop/ConfigStore hooks present |
| error | `capture_fail` | running but capture_ok=false |
| error | `region_keys_missing` | capture.region_x / capture.region missing — UI cannot set ROI |
| warn | `use_region_missing` | use_region key missing from config document (OBS page1) |
| warn | `missing_capture` | config missing top-level 'capture' |
| warn | `missing_infer` | config missing top-level 'infer' |
| warn | `missing_aim` | config missing top-level 'aim' |
| warn | `missing_tracking` | config missing top-level 'tracking' |
| warn | `missing_vision` | config missing top-level 'vision' |
| warn | `missing_prediction` | config missing top-level 'prediction' |
| warn | `missing_crosshair` | config missing top-level 'crosshair' |
| error | `aim_slots` | aim.configs expected 5 slots, got NoneType len=n/a |
| warn | `infer_confidence` | infer.confidence missing |
| warn | `infer_nms` | infer.nms missing |
| warn | `infer_model_path` | infer.model_path missing |
| warn | `infer_device` | infer.device missing |


## OBS parity scan — 2026-07-23T19:53:12+08:00

- **status**: `offline_static_ok` (errors=0 warns=0 infos=4)
- host: `http://127.0.0.1:17890` offline=True

| level | id | msg |
|-------|----|-----|
| info | `static_contract_ok` | OBS filter_inference ROI+remap patterns + standalone EngineLoop/ConfigStore hooks present |
| info | `status_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |
| info | `config_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |
| info | `dets_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |


## OBS parity scan — 2026-07-23T19:52:37+08:00

- **status**: `offline` (errors=0 warns=0 infos=3)
- host: `http://127.0.0.1:17890` offline=True

| level | id | msg |
|-------|----|-----|
| info | `status_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |
| info | `config_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |
| info | `dets_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |


## OBS parity scan — 2026-07-23T19:52:05+08:00

- **status**: `offline` (errors=0 warns=0 infos=3)
- host: `http://127.0.0.1:17890` offline=True

| level | id | msg |
|-------|----|-----|
| info | `status_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |
| info | `config_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |
| info | `dets_err` | <urlopen error [WinError 10061] 由于目标计算机积极拒绝，无法连接。> |


