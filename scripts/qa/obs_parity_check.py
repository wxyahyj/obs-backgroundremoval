#!/usr/bin/env python3
"""OBS vs standalone functional parity probe (every-10-min friendly).

Compares live host status/config against OBS-expected semantics for:
  - capture ROI (not full-screen by default)
  - use_region / region_* fields
  - inference device honesty
  - detection coordinate ranges
  - dead feature smoke (config keys present)

Writes report under docs/worklog/ and .agent/logs/.
Exit 0 = ok or host offline (soft); exit 2 = hard functional fail.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = "http://127.0.0.1:17890"
LOG_DIR = ROOT / ".agent" / "logs"
WORKLOG = ROOT / "docs" / "worklog"
REPORT_JSON = LOG_DIR / "obs_parity_latest.json"
REPORT_MD = WORKLOG / "obs-parity-scan.md"


def get_json(path: str, timeout: float = 3.0):
    url = HOST + path
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def safe_get(path: str):
    try:
        return get_json(path), None
    except Exception as e:
        return None, str(e)


def check_status(st: dict) -> list[dict]:
    issues = []
    if not st:
        return [{"level": "error", "id": "host_offline", "msg": "status unreachable"}]

    mode = (st.get("capture_mode") or "").lower()
    w = int(st.get("capture_w") or 0)
    h = int(st.get("capture_h") or 0)
    origin = st.get("origin") or [0, 0]
    use_region = st.get("use_region")
    region_w = int(st.get("region_width") or 0)
    region_h = int(st.get("region_height") or 0)

    # OBS default for standalone: ROI capture, not full desktop
    if mode in ("center", "region"):
        if w >= 1800 and h >= 1000:
            issues.append(
                {
                    "level": "warn",
                    "id": "roi_looks_fullscreen",
                    "msg": f"capture {w}x{h} mode={mode} looks full-desktop; expect region-sized ROI",
                }
            )
        if w > 0 and h > 0 and (w < 64 or h < 64):
            issues.append(
                {
                    "level": "error",
                    "id": "roi_too_small",
                    "msg": f"capture {w}x{h} invalid",
                }
            )
    elif mode in ("full", "fullscreen"):
        if use_region is False:
            issues.append(
                {
                    "level": "info",
                    "id": "full_no_region",
                    "msg": "full capture without use_region → full-frame infer (OBS default when use_region=false)",
                }
            )
        elif use_region is True and region_w > 0 and region_h > 0:
            if region_w >= w and region_h >= h and w > 0:
                issues.append(
                    {
                        "level": "warn",
                        "id": "use_region_noop",
                        "msg": "use_region=true but region size >= full frame",
                    }
                )

    if st.get("running") and not st.get("capture_ok"):
        issues.append({"level": "error", "id": "capture_fail", "msg": "running but capture_ok=false"})
    if st.get("running") and st.get("infer_enabled") and not st.get("infer_ok"):
        issues.append({"level": "error", "id": "infer_fail", "msg": "infer enabled but infer_ok=false"})

    dev_req = (st.get("device_requested") or st.get("device") or "").lower()
    dev_act = (st.get("device_actual") or st.get("device") or "").lower()
    if dev_req and dev_act:
        base = lambda s: s.split("+")[0]
        if base(dev_req) != base(dev_act) and "cpu" in dev_act and "cpu" not in dev_req:
            issues.append(
                {
                    "level": "warn",
                    "id": "device_fallback",
                    "msg": f"device requested={dev_req} actual={dev_act}",
                }
            )

    # origin should be non-zero for center crop on typical 1080p/1440p
    if mode == "center" and w > 0 and h > 0:
        ox, oy = int(origin[0] or 0), int(origin[1] or 0)
        if ox == 0 and oy == 0 and w < 1600:
            issues.append(
                {
                    "level": "info",
                    "id": "origin_zero",
                    "msg": "center mode origin is [0,0] — verify screen metrics / capture open",
                }
            )

    fps = float(st.get("capture_fps") or 0)
    if st.get("running") and st.get("capture_ok") and fps > 0 and fps < 10:
        issues.append(
            {
                "level": "warn",
                "id": "low_capture_fps",
                "msg": f"capture_fps={fps:.1f} (DXGI path may be stalling)",
            }
        )

    return issues


def check_config(cfg: dict) -> list[dict]:
    issues = []
    if not cfg:
        return [{"level": "error", "id": "config_offline", "msg": "config unreachable"}]

    cap = cfg.get("capture") or {}
    mode = (cap.get("mode") or "").lower()
    # WebUI region_x must round-trip
    if "region_x" not in cap and not (isinstance(cap.get("region"), dict) and "x" in cap["region"]):
        issues.append(
            {
                "level": "error",
                "id": "region_keys_missing",
                "msg": "capture.region_x / capture.region missing — UI cannot set ROI",
            }
        )
    if "use_region" not in cap and "use_region" not in cfg:
        issues.append(
            {
                "level": "warn",
                "id": "use_region_missing",
                "msg": "use_region key missing from config document (OBS page1)",
            }
        )

    # OBS parity feature keys
    required_top = ["capture", "infer", "aim", "tracking", "vision", "prediction", "crosshair"]
    for k in required_top:
        if k not in cfg:
            issues.append({"level": "warn", "id": f"missing_{k}", "msg": f"config missing top-level '{k}'"})

    aim = cfg.get("aim") or {}
    configs = aim.get("configs")
    if not isinstance(configs, list) or len(configs) < 5:
        issues.append(
            {
                "level": "error",
                "id": "aim_slots",
                "msg": f"aim.configs expected 5 slots, got {type(configs).__name__} len={len(configs) if isinstance(configs, list) else 'n/a'}",
            }
        )

    infer = cfg.get("infer") or {}
    for k in ("confidence", "nms", "model_path", "device"):
        if k not in infer and k not in cfg:
            issues.append({"level": "warn", "id": f"infer_{k}", "msg": f"infer.{k} missing"})

    # capture mode default should not imply full-screen inference
    if mode == "full" and not (cap.get("use_region") or cfg.get("use_region")):
        issues.append(
            {
                "level": "info",
                "id": "default_full_infer",
                "msg": "mode=full and use_region=false → full-frame YOLO (user-reported issue pattern)",
            }
        )

    return issues


def check_detections(dets_payload) -> list[dict]:
    issues = []
    if dets_payload is None:
        return [{"level": "info", "id": "dets_skip", "msg": "detections unavailable"}]

    dets = dets_payload
    if isinstance(dets_payload, dict):
        dets = dets_payload.get("detections") or dets_payload.get("dets") or []
    if not isinstance(dets, list):
        return [{"level": "warn", "id": "dets_shape", "msg": f"unexpected detections type {type(dets_payload)}"}]

    if not dets:
        return [{"level": "info", "id": "dets_empty", "msg": "0 detections (ok if no targets)"}]

    bad_norm = 0
    origin_cluster = 0
    for d in dets[:200]:
        x = float(d.get("x", d.get("cx", 0.5)))
        y = float(d.get("y", d.get("cy", 0.5)))
        w = float(d.get("w", d.get("width", 0)))
        h = float(d.get("h", d.get("height", 0)))
        if x < -0.05 or y < -0.05 or x > 1.05 or y > 1.05:
            bad_norm += 1
        if abs(x) < 0.02 and abs(y) < 0.02 and w < 0.05 and h < 0.05:
            origin_cluster += 1

    if bad_norm > len(dets) * 0.2:
        issues.append(
            {
                "level": "error",
                "id": "coords_out_of_range",
                "msg": f"{bad_norm}/{len(dets)} dets outside normalized [0,1] (layout/postprocess bug)",
            }
        )
    if origin_cluster > max(5, len(dets) * 0.5):
        issues.append(
            {
                "level": "error",
                "id": "boxes_top_left",
                "msg": f"{origin_cluster}/{len(dets)} dets clustered at origin (YOLO layout inversion?)",
            }
        )
    if len(dets) > 200:
        issues.append(
            {
                "level": "warn",
                "id": "too_many_dets",
                "msg": f"det_count={len(dets)} very high — check conf/nms/NMS class-aware path",
            }
        )
    return issues


def check_obs_source_contract() -> list[dict]:
    """Static checks against OBS source files (no OBS running required)."""
    issues = []
    fi = ROOT / "src" / "filter_inference.cpp"
    if not fi.exists():
        issues.append({"level": "warn", "id": "obs_src_missing", "msg": "filter_inference.cpp not found"})
        return issues
    text = fi.read_text(encoding="utf-8", errors="replace")
    for needle, mid in [
        ("cropX", "obs_crop_x"),
        ("cropWidth", "obs_crop_w"),
        ("newDetections = modelSnap->inference", "obs_infer_on_roi"),
        ("pixelX = det.x * cropWidth + cropX", "obs_remap"),
    ]:
        if needle not in text:
            issues.append(
                {
                    "level": "error",
                    "id": mid,
                    "msg": f"OBS filter_inference missing expected pattern: {needle}",
                }
            )

    eng = ROOT / "standalone" / "src" / "host" / "EngineLoop.cpp"
    if eng.exists():
        et = eng.read_text(encoding="utf-8", errors="replace")
        if "remap_dets_roi_to_full" not in et:
            issues.append(
                {
                    "level": "error",
                    "id": "standalone_no_remap",
                    "msg": "EngineLoop missing OBS-style ROI→full remap",
                }
            )
        if "resolve_infer_roi" not in et:
            issues.append(
                {
                    "level": "error",
                    "id": "standalone_no_roi_resolve",
                    "msg": "EngineLoop missing resolve_infer_roi",
                }
            )
        if "open_capture_obs_style" not in et:
            issues.append(
                {
                    "level": "error",
                    "id": "standalone_no_obs_open",
                    "msg": "EngineLoop missing open_capture_obs_style",
                }
            )
    cs = ROOT / "standalone" / "src" / "core" / "ConfigStore.cpp"
    if cs.exists():
        ct = cs.read_text(encoding="utf-8", errors="replace")
        if 'jget_int(cap, "region_x"' not in ct and "region_x" not in ct:
            issues.append(
                {
                    "level": "error",
                    "id": "config_region_x_bug",
                    "msg": "ConfigStore does not merge capture.region_x (WebUI path broken)",
                }
            )
        if "use_region" not in ct:
            issues.append(
                {
                    "level": "error",
                    "id": "config_use_region_missing",
                    "msg": "ConfigStore missing use_region",
                }
            )
    return issues


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORKLOG.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    issues: list[dict] = []
    static_issues = check_obs_source_contract()
    issues.extend(static_issues)
    if not static_issues:
        issues.append(
            {
                "level": "info",
                "id": "static_contract_ok",
                "msg": "OBS filter_inference ROI+remap patterns + standalone EngineLoop/ConfigStore hooks present",
            }
        )

    st, st_err = safe_get("/api/status")
    cfg, cfg_err = safe_get("/api/config")
    dets, dets_err = safe_get("/api/detections")

    offline = st is None and cfg is None
    if st_err:
        issues.append({"level": "info", "id": "status_err", "msg": st_err})
    if cfg_err:
        issues.append({"level": "info", "id": "config_err", "msg": cfg_err})
    if dets_err:
        issues.append({"level": "info", "id": "dets_err", "msg": dets_err})

    if st:
        issues.extend(check_status(st))
    if cfg:
        issues.extend(check_config(cfg))
    if dets is not None:
        issues.extend(check_detections(dets))

    errors = [i for i in issues if i["level"] == "error"]
    warns = [i for i in issues if i["level"] == "warn"]
    infos = [i for i in issues if i["level"] == "info"]

    if errors:
        status = "fail"
        exit_code = 2
    elif offline:
        status = "offline_static_ok" if not static_issues else "offline_static_fail"
        exit_code = 0 if not static_issues else 2
    elif warns:
        status = "warn"
        exit_code = 0
    else:
        status = "ok"
        exit_code = 0

    report = {
        "version": 1,
        "status": status,
        "verifiedAt": ts,
        "host": HOST,
        "offline": offline,
        "static_errors": len([i for i in static_issues if i["level"] == "error"]),
        "summary": {
            "errors": len(errors),
            "warnings": len(warns),
            "infos": len(infos),
        },
        "status_snapshot": st,
        "issues": issues,
    }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Append-friendly markdown summary
    lines = [
        f"## OBS parity scan — {ts}",
        "",
        f"- **status**: `{status}` (errors={len(errors)} warns={len(warns)} infos={len(infos)})",
        f"- host: `{HOST}` offline={offline}",
        "",
    ]
    if st:
        lines.append(
            f"- capture: mode={st.get('capture_mode')} size={st.get('capture_w')}x{st.get('capture_h')} "
            f"use_region={st.get('use_region')} origin={st.get('origin')} fps={st.get('capture_fps')}"
        )
        lines.append(
            f"- infer: ok={st.get('infer_ok')} device={st.get('device_actual')} ms={st.get('infer_ms')} dets={st.get('last_det_count')}"
        )
        lines.append("")
    if issues:
        lines.append("| level | id | msg |")
        lines.append("|-------|----|-----|")
        for i in issues:
            msg = str(i.get("msg", "")).replace("|", "\\|")
            lines.append(f"| {i.get('level')} | `{i.get('id')}` | {msg} |")
        lines.append("")
    else:
        lines.append("_No issues._\n")

    # Keep last ~80 scans in the md file
    prev = REPORT_MD.read_text(encoding="utf-8") if REPORT_MD.exists() else "# OBS ↔ standalone parity worklog\n\n"
    block = "\n".join(lines) + "\n"
    # prepend new scan
    if prev.startswith("#"):
        head, _, rest = prev.partition("\n")
        combined = head + "\n\n" + block + rest
    else:
        combined = "# OBS ↔ standalone parity worklog\n\n" + block + prev
    # trim length
    if len(combined) > 120_000:
        combined = combined[:120_000] + "\n\n…trimmed…\n"
    REPORT_MD.write_text(combined, encoding="utf-8")

    print(json.dumps({"status": status, "errors": len(errors), "warnings": len(warns), "report": str(REPORT_JSON)}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
