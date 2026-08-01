/* YOLO Aim WebUI — OBS-key forms, data-key / data-path / data-slot */

const CTRL_NAMES = [
  "WindowsAPI",
  "MAKCU",
  "LogiDriver",
  "GvInput",
  "TencInput",
  "NtUserSendInput",
  "NtUserInjectMouse",
  "NtUserInjectPointer",
];

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(path + " " + res.status + " " + t.slice(0, 200));
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

function $(id) {
  return document.getElementById(id);
}

let __msgQuietUntil = 0; // suppress spam while dragging sliders
function setMsg(text, ok) {
  const el = $("applyMsg");
  if (!el) return;
  // Quiet auto-apply chatter during range drag (user: adjusting params must not flash tips)
  if (ok !== false && text && /已自动应用/.test(text) && performance.now() < __msgQuietUntil) {
    return;
  }
  el.textContent = text || "";
  el.classList.toggle("err", ok === false);
}

function currentSlot() {
  const el = $("slot");
  return Math.max(0, Math.min(4, Number(el?.value || 0)));
}

function setSlot(n) {
  n = Math.max(0, Math.min(4, Number(n) || 0));
  const el = $("slot");
  if (el) el.value = String(n);
  document.querySelectorAll("[data-slot-btn]").forEach((b) => {
    b.classList.toggle("active", Number(b.getAttribute("data-slot-btn")) === n);
  });
  if (window.__cfg) fillFromConfig(window.__cfg);
}

function getByPath(obj, path) {
  if (!path) return undefined;
  return path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

function setByPath(obj, path, value) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cur[parts[i]] == null || typeof cur[parts[i]] !== "object") cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}

function readControl(el) {
  if (el.type === "checkbox") return !!el.checked;
  if (el.type === "number" || el.type === "range") {
    const n = Number(el.value);
    return Number.isFinite(n) ? n : el.value;
  }
  return el.value;
}

function writeControl(el, val) {
  if (val == null) return;
  if (el.type === "checkbox") {
    el.checked = !!val;
    return;
  }
  el.value = String(val);
  if (el.type === "range") {
    const span = document.getElementById(el.id + "Val");
    if (span) span.textContent = Number(val).toFixed(2);
  }
}

function collectPatch(root) {
  const scope = root || document;
  const patch = {};
  const slot = currentSlot();
  const slotObj = {};

  scope.querySelectorAll("[data-key]").forEach((el) => {
    const key = el.getAttribute("data-key");
    const path = el.getAttribute("data-path");
    const isSlot = el.hasAttribute("data-slot");
    const val = readControl(el);

    if (path) {
      setByPath(patch, path, val);
      // dual-write OBS root aliases for scene-JSON compatibility
      const obsAlias = el.getAttribute("data-obs-alias");
      if (obsAlias) patch[obsAlias] = val;
      return;
    }
    if (isSlot) {
      slotObj[key] = val;
      if (key === "enable_config") slotObj.enabled = val;
      if (key === "hotkey") slotObj.hotkey_vk = val;
      if (key === "controller_type") {
        slotObj.controller = CTRL_NAMES[Number(val)] || "WindowsAPI";
      }
      return;
    }
    patch[key] = val;
  });

  if (Object.keys(slotObj).length) {
    if (!patch.aim) patch.aim = {};
    patch.slot = slot;
    Object.assign(patch, slotObj);
    patch.aim.configs = [{}, {}, {}, {}, {}];
    patch.aim.configs[slot] = { ...slotObj };
    patch.aim.mouse_config_select = slot;
  }

  if (patch.aim && patch.aim.enabled != null) patch.aim_enabled = !!patch.aim.enabled;
  if (patch.aim_enabled != null && patch.aim) patch.aim.enabled = !!patch.aim_enabled;

  return patch;
}

function slotFieldsFromConfig(c, slot) {
  if (c.aim && Array.isArray(c.aim.configs) && c.aim.configs[slot]) {
    return c.aim.configs[slot];
  }
  if (c.aim && c.aim.obs_keys) {
    const out = {};
    const suf = "_" + slot;
    for (const [k, v] of Object.entries(c.aim.obs_keys)) {
      if (k.endsWith(suf)) out[k.slice(0, -suf.length)] = v;
    }
    return out;
  }
  return {};
}

function fillFromConfig(c) {
  if (!c) return;
  const slot = currentSlot();
  const cfgSlot = slotFieldsFromConfig(c, slot);

  document.querySelectorAll("[data-key]").forEach((el) => {
    const key = el.getAttribute("data-key");
    const path = el.getAttribute("data-path");
    const obsAlias = el.getAttribute("data-obs-alias");
    const isSlot = el.hasAttribute("data-slot");
    let val;
    if (path) val = getByPath(c, path);
    // OBS flat root alias fallback (e.g. confidence_threshold)
    if (val == null && obsAlias && c[obsAlias] != null) val = c[obsAlias];
    if (val == null && key && c[key] != null) val = c[key];
    if (path && val == null) {
      // nested OBS names under infer.*
      if (path.startsWith("infer.") && c.infer) {
        const leaf = path.slice(6);
        val = c.infer[leaf];
        if (val == null && obsAlias) val = c.infer[obsAlias];
      }
    }
    if (isSlot && val == null) {
      val = cfgSlot[key];
      if (val == null && key === "enable_config") val = cfgSlot.enabled;
      if (val == null && key === "hotkey") val = cfgSlot.hotkey_vk ?? cfgSlot.hotkey;
    } else if (val == null && !path) {
      val = getByPath(c, "aim." + key);
    }
    writeControl(el, val);
  });

  const en = document.querySelector('[data-key="aim_enabled"]');
  if (en && c.aim) writeControl(en, c.aim.enabled);

  const conf = $("confidence");
  if (conf) {
    const v =
      c.infer?.confidence ??
      c.infer?.confidence_threshold ??
      c.confidence_threshold ??
      c.confidence;
    if (v != null) {
      conf.value = v;
      if ($("confidenceVal")) $("confidenceVal").textContent = Number(v).toFixed(2);
    }
  }
  const nms = $("nms");
  if (nms) {
    const v = c.infer?.nms ?? c.infer?.nms_threshold ?? c.nms_threshold ?? c.nms;
    if (v != null) {
      nms.value = v;
      if ($("nmsVal")) $("nmsVal").textContent = Number(v).toFixed(2);
    }
  }

  // sync algo panel highlight
  const algo = Number(c.aim?.algorithm_type_global ?? c.aim?.algorithm ?? 0);
  showAlgoPanel(algo, false);
  setSlotUi(slot);
  // keep dual range/number pairs in sync after fill
  document.querySelectorAll(".dual-control").forEach((wrap) => {
    const range = wrap.querySelector('input[type="range"]');
    const num = wrap.querySelector('input[type="number"]');
    if (range && num && num.value !== "") range.value = num.value;
  });
}

function setSlotUi(n) {
  document.querySelectorAll("[data-slot-btn]").forEach((b) => {
    b.classList.toggle("active", Number(b.getAttribute("data-slot-btn")) === n);
  });
}

function showAlgoPanel(n, writeSelect) {
  n = Math.max(0, Math.min(4, Number(n) || 0));
  document.querySelectorAll(".algo-tab").forEach((t) => {
    t.classList.toggle("active", Number(t.getAttribute("data-algo")) === n);
  });
  document.querySelectorAll(".algo-panel").forEach((p) => {
    p.classList.toggle("active", Number(p.getAttribute("data-algo-panel")) === n);
  });
  if (writeSelect !== false) {
    const sel = $("algoSelect");
    if (sel) sel.value = String(n);
  }
}

async function loadConfig() {
  try {
    __suppressAuto = true;
    const c = await api("/api/config");
    window.__cfg = c;
    if (c.aim && c.aim.mouse_config_select != null) {
      setSlot(c.aim.mouse_config_select);
    } else {
      fillFromConfig(c);
    }
    await loadModels();
    setMsg("配置已加载 · 改控件即自动应用", true);
  } catch (e) {
    setMsg(String(e), false);
  } finally {
    setTimeout(() => {
      __suppressAuto = false;
    }, 50);
  }
}

async function putConfig(patch, opts) {
  const res = await api("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  // Avoid fillFromConfig on every auto-apply (steals focus / resets caret).
  if (res.config) {
    window.__cfg = res.config;
    if (opts && opts.refill) fillFromConfig(res.config);
  }
  return res;
}

let __applyTimer = null;
let __applyBusy = false;
let __suppressAuto = false;

function coldActionFor(el) {
  return el.getAttribute("data-cold") || "";
}

function scheduleAutoApply(fromEl) {
  if (__suppressAuto) return;
  clearTimeout(__applyTimer);
  const delay = fromEl && (fromEl.type === "range" || fromEl.type === "number") ? 280 : 120;
  __applyTimer = setTimeout(() => autoApplyNow(fromEl), delay);
}

async function autoApplyNow(fromEl) {
  if (__applyBusy || __suppressAuto) return;
  __applyBusy = true;
  try {
    // Prefer single-control patch when possible; otherwise active page.
    let patch = {};
    if (fromEl && fromEl.hasAttribute("data-key")) {
      const key = fromEl.getAttribute("data-key");
      const path = fromEl.getAttribute("data-path");
      const isSlot = fromEl.hasAttribute("data-slot");
      const val = readControl(fromEl);
      if (path) {
        setByPath(patch, path, val);
      } else if (isSlot) {
        const slot = currentSlot();
        patch.slot = slot;
        patch[key] = val;
        if (key === "enable_config") patch.enabled = val;
        if (key === "hotkey") patch.hotkey_vk = val;
        if (key === "controller_type") {
          patch.controller = CTRL_NAMES[Number(val)] || "WindowsAPI";
        }
        patch.aim = { configs: [{}, {}, {}, {}, {}], mouse_config_select: slot };
        patch.aim.configs[slot] = { [key]: val };
        if (key === "enable_config") patch.aim.configs[slot].enabled = val;
        if (key === "controller_type") {
          patch.aim.configs[slot].controller_type = val;
          patch.aim.configs[slot].controller = CTRL_NAMES[Number(val)] || "WindowsAPI";
        }
      } else {
        patch[key] = val;
      }
      // conf/nms dual flat+nested
      if (key === "confidence" || path === "infer.confidence") {
        patch.confidence = val;
        setByPath(patch, "infer.confidence", val);
      }
      if (key === "nms" || path === "infer.nms") {
        patch.nms = val;
        setByPath(patch, "infer.nms", val);
      }
    } else {
      const page = document.querySelector(".page.active") || document;
      patch = collectPatch(page);
    }

    await putConfig(patch, { refill: false });

    const cold = fromEl ? coldActionFor(fromEl) : "";
    if (cold === "reload_model") {
      await api("/api/engine/reload_model", { method: "POST" });
      setMsg("已自动应用并重载模型", true);
    } else if (cold === "reload_capture") {
      await api("/api/engine/reload_capture", { method: "POST" });
      setMsg("已自动应用并重开截图", true);
    } else {
      setMsg("已自动应用", true);
    }
    setTimeout(refresh, 200);
  } catch (e) {
    setMsg("自动应用失败: " + e, false);
  } finally {
    __applyBusy = false;
  }
}

function modelFolderValue() {
  const el = $("modelFolder");
  return (el && el.value ? el.value : "").trim();
}

function modelRecursive() {
  const el = $("modelRecursive");
  return !el || !!el.checked;
}

async function loadModels(opts) {
  const sel = $("modelPick");
  if (!sel) return;
  try {
    const folder = (opts && opts.folder) || modelFolderValue();
    const recursive = opts && opts.recursive != null ? opts.recursive : modelRecursive();
    let url = "/api/models?recursive=" + (recursive ? "1" : "0");
    if (folder) url += "&dir=" + encodeURIComponent(folder);
    const data = await api(url);
    const cur = data.current || window.__cfg?.infer?.model_path || "";
    if ($("modelFolder") && data.models_dir && !$("modelFolder").value) {
      $("modelFolder").value = data.models_dir;
    }
    if (data.folder && $("modelFolder") && (opts && opts.fillFolder)) {
      $("modelFolder").value = data.folder;
    }
    sel.innerHTML = "";
    const models = data.models || [];
    if (!models.length) {
      const o = document.createElement("option");
      o.value = "";
      o.textContent = folder
        ? "(该文件夹未找到 .onnx/.engine，检查路径或递归选项)"
        : "(未找到模型，请填写模型文件夹并扫描)";
      sel.appendChild(o);
    } else {
      for (const m of models) {
        const o = document.createElement("option");
        o.value = m.path;
        const sizeKb = m.size ? "  (" + Math.round(m.size / 1024) + " KB)" : "";
        o.textContent = (m.dir ? m.dir + " / " : "") + m.name + sizeKb;
        if (m.path === cur || m.name === (cur.split(/[/\\]/).pop())) o.selected = true;
        sel.appendChild(o);
      }
    }
    if (cur && $("modelPath")) $("modelPath").value = cur;
    if (opts && opts.msg !== false) {
      setMsg(
        "识别到 " + models.length + " 个模型" + (folder ? " · " + folder : ""),
        true
      );
    }
    return data;
  } catch (e) {
    setMsg("模型列表: " + e, false);
    return null;
  }
}

async function scanModelFolder(selectFirst) {
  const folder = modelFolderValue();
  if (!folder) {
    setMsg("请先填写模型文件夹路径（本机绝对路径）", false);
    return;
  }
  try {
    setMsg("正在扫描 " + folder + " …", true);
    const res = await api("/api/models/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dir: folder,
        recursive: modelRecursive(),
        select_first: !!selectFirst,
      }),
    });
    // res is already models json
    const models = res.models || [];
    if ($("modelFolder") && res.folder) $("modelFolder").value = res.folder;
    if ($("modelFolder") && res.models_dir) $("modelFolder").value = res.models_dir;
    await loadModels({ folder: res.folder || folder, recursive: modelRecursive(), msg: false });
    if (models.length === 1 || selectFirst) {
      const path = models[0] && models[0].path;
      if (path) {
        if ($("modelPath")) $("modelPath").value = path;
        if ($("modelPick")) $("modelPick").value = path;
        await putConfig(
          { model_path: path, infer: { model_path: path } },
          { refill: false }
        );
        try {
          await api("/api/engine/reload_model", { method: "POST" });
        } catch (_) {}
      }
    }
    setMsg(
      "文件夹已设 · 识别 " + models.length + " 个模型" +
        (models.length ? " · 下拉选择即可加载" : ""),
      true
    );
    setTimeout(refresh, 300);
  } catch (e) {
    setMsg("扫描失败: " + e, false);
  }
}

function drawOverlay(dets) {
  const img = $("preview");
  const canvas = $("overlay");
  if (!img || !canvas) return;
  const ctx = canvas.getContext("2d");
  const w = img.clientWidth || 1;
  const h = img.clientHeight || 1;
  if (canvas.width !== Math.round(w) || canvas.height !== Math.round(h)) {
    canvas.width = Math.round(w);
    canvas.height = Math.round(h);
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const fovEl = document.querySelector('[data-key="fov_radius"]');
  const fov = Number(fovEl?.value) || 120;
  // assume capture ~ square; scale FOV by width
  const scale = canvas.width / 640;
  // Colored FOV + detection boxes (dark theme)
  ctx.strokeStyle = "rgba(61, 155, 255, 0.9)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(canvas.width / 2, canvas.height / 2, fov * scale, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = "rgba(139, 124, 255, 0.75)";
  ctx.beginPath();
  ctx.moveTo(canvas.width / 2 - 12, canvas.height / 2);
  ctx.lineTo(canvas.width / 2 + 12, canvas.height / 2);
  ctx.moveTo(canvas.width / 2, canvas.height / 2 - 12);
  ctx.lineTo(canvas.width / 2, canvas.height / 2 + 12);
  ctx.stroke();

  // Match vision.show_detection_results when known
  const showEl = document.querySelector('[data-key="show_detection_results"]');
  if (showEl && showEl.type === "checkbox" && !showEl.checked) return;

  const thickEl = document.querySelector('[data-key="bbox_line_width"]');
  const lineW = Math.max(1, Math.min(10, Number(thickEl?.value) || 2));
  const items = (dets && dets.items) || [];
  const maxDraw = 64;
  for (let i = 0; i < items.length && i < maxDraw; i++) {
    const d = items[i];
    // Guard invalid coords (air/ghost)
    if (!(d.w > 0 && d.h > 0) || d.x < -0.05 || d.y < -0.05) continue;
    if (d.x + d.w > 1.05 || d.y + d.h > 1.05) continue;
    const x = d.x * canvas.width;
    const y = d.y * canvas.height;
    const bw = d.w * canvas.width;
    const bh = d.h * canvas.height;
    const hue = ((Number(d.class_id) || 0) * 47) % 360;
    ctx.strokeStyle = `hsl(${hue} 80% 55%)`;
    ctx.lineWidth = lineW;
    ctx.strokeRect(x, y, bw, bh);
    const tid = d.track_id != null && d.track_id >= 0 ? ` t${d.track_id}` : "";
    const label = `c${d.class_id}${tid} ${((d.confidence || 0) * 100).toFixed(0)}%`;
    ctx.font = "11px Cascadia Mono, Consolas, monospace";
    const tw = ctx.measureText(label).width + 6;
    // Label INSIDE box top so it stays on the detection (not floating on "air")
    let lx = x;
    let ly = y;
    if (ly < 2) ly = y;
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(lx, ly, tw, 14);
    ctx.fillStyle = "#eef3fb";
    ctx.fillText(label, lx + 3, ly + 11);
  }
}

/* ---- poll / preview scheduling (avoid thrashing BMP + JSON every tick) ---- */
let __previewBusy = false;
let __previewLastAt = 0;
let __detDumpAt = 0;
let __statusBusy = false;
let __detBusy = false;
let __engineRunning = false;
const PREVIEW_MIN_MS = 40; // ~25 FPS client pull (server default ~30 FPS)
const STATUS_MS = 400;
const DET_MS = 150;
const DET_DUMP_MS = 1500;

function setText(id, text) {
  const el = $(id);
  if (!el) return;
  const t = text == null ? "" : String(text);
  if (el.textContent !== t) el.textContent = t;
}

function refreshPreview() {
  const img = $("preview");
  if (!img) return;
  // Skip fetching preview when the user disabled it (saves CPU + network).
  const peCb = document.querySelector('[data-key="preview_enabled"]');
  if (peCb && !peCb.checked) {
    const hint = $("previewHint");
    if (hint) hint.textContent = "预览已关闭（勾选开启）";
    return;
  }
  const now = performance.now();
  if (__previewBusy) return;
  if (now - __previewLastAt < PREVIEW_MIN_MS) return;
  __previewBusy = true;
  __previewLastAt = now;
  // Decode off-DOM then swap — avoids layout thrash on broken mid-load frames
  const probe = new Image();
  probe.decoding = "async";
  probe.onload = () => {
    img.src = probe.src;
    const hint = $("previewHint");
    if (hint) hint.textContent = "预览 · " + new Date().toLocaleTimeString();
    drawOverlay(window.__dets);
    __previewBusy = false;
  };
  probe.onerror = () => {
    const hint = $("previewHint");
    if (hint) hint.textContent = "暂无预览（未启动或截图失败）";
    __previewBusy = false;
  };
  probe.src = "/api/preview.bmp?t=" + Date.now();
}

function applyStatus(s) {
  const running = !!s.running;
  __engineRunning = running;
  const badge = $("runBadge");
  if (badge) {
    const t = running ? "运行中" : "已停止";
    if (badge.textContent !== t) {
      badge.textContent = t;
      badge.className = "badge " + (running ? "on" : "off");
    }
  }
  if ($("sCapture")) {
    const sz =
      s.capture_size ||
      (s.capture_w != null && s.capture_h != null ? s.capture_w + "x" + s.capture_h : "");
    setText(
      "sCapture",
      (s.capture_ok ? "正常" : "失败") +
        (sz ? " " + sz : "") +
        " / 推理 " +
        (s.infer_ok ? "正常" : "关闭")
    );
  }
  const backendMap = { dxgi: "DXGI 桌面复制", wgc: "WGC 窗口捕获", gdi: "GDI 截图" };
  setText("sBackend", backendMap[String(s.backend || "").toLowerCase()] || s.backend || "—");
  if ($("sMode")) {
    const mode = s.capture_mode || "—";
    const origin =
      s.origin && Array.isArray(s.origin)
        ? " 裁切原点=(" + s.origin[0] + "," + s.origin[1] + ")"
        : "";
    const roi =
      s.infer_roi && Array.isArray(s.infer_roi) && s.infer_roi.length >= 4
        ? " 检测区=" + s.infer_roi[2] + "x" + s.infer_roi[3] + "@(" + s.infer_roi[0] + "," + s.infer_roi[1] + ")"
        : s.capture_region && Array.isArray(s.capture_region)
          ? " 区域=" + JSON.stringify(s.capture_region)
          : "";
    const fullTag =
      s.infer_is_fullscreen === true
        ? " ⚠疑似全屏"
        : " · 检测区域推理（非全屏）";
    const label =
      mode === "center"
        ? "居中裁切"
        : mode === "region"
          ? "固定区域"
          : mode === "full"
            ? "全屏捕获"
            : mode;
    setText("sMode", label + " " + (s.capture_size || "") + origin + roi + fullTag);
  }
  if ($("sDevice")) {
    const req = String(s.device_requested || s.device || "—").toLowerCase();
    const act = String(s.device_actual || s.device || "—").toLowerCase();
    const ep = (d) => String(d).split("+")[0].split(" ")[0];
    const reqEp = ep(req);
    const actEp = ep(act);
    const nameMap = { cuda: "CUDA", dml: "DirectML", tensorrt: "TensorRT", cpu: "CPU" };
    const gpuOk =
      act.includes("cuda") || act.includes("dml") || act.includes("tensorrt");
    const same = reqEp === actEp || (reqEp === "cuda" && act.includes("cuda"));
    let label;
    if (act.includes("cpu_pre") && gpuOk) {
      label = "CUDA 正常：网络在 GPU 推理，预处理在 CPU";
    } else if (same && gpuOk) {
      label = (nameMap[actEp] || actEp.toUpperCase()) + " 正常 · 全 GPU 路径";
    } else if (same && actEp === "cpu") {
      label = "CPU 推理";
    } else if (actEp === "cpu" && reqEp !== "cpu") {
      label = "请求 " + (nameMap[reqEp] || req) + " → 已回退到 CPU";
    } else {
      label =
        "请求 " +
        (nameMap[reqEp] || req) +
        " → 实际 " +
        (nameMap[actEp] || act);
    }
    setText("sDevice", label);
    if (gpuOk) {
      $("sDevice").style.color = "var(--ok)";
    } else if (actEp === "cpu" && reqEp !== "cpu") {
      $("sDevice").style.color = "var(--warn)";
    } else {
      $("sDevice").style.color = "var(--text)";
    }
    $("sDevice").style.fontWeight = gpuOk ? "600" : "500";
    $("sDevice").style.textDecoration = "none";
  }
  setText("sModel", s.model_path || "—");
  setText(
    "sFps",
    "截图 " +
      (s.capture_fps ?? 0).toFixed(1) +
      " / 推理 " +
      (s.infer_fps ?? 0).toFixed(1)
  );
  setText("sFrames", s.frame_count ?? 0);
  setText(
    "sDets",
    (s.last_det_count ?? 0) + "（" + (s.infer_ms ?? 0).toFixed(1) + " 毫秒）"
  );
  setText(
    "sAim",
    (s.aim_ok ? "开启" : "关闭") +
      " 槽位=" +
      (s.active_slot ?? -1) +
      " 热键=" +
      (s.aim_hotkey_down ? "按下" : "松开")
  );
  const algoMap = {
    AdvancedPID: "高级 PID",
    ExternalPID: "外部 PID",
    AimController: "瞄准控制",
    SlewRate: "限速控制",
    AdaptivePID: "自适应 PID",
  };
  const ctrlMap = {
    WindowsAPI: "系统接口",
    MAKCU: "MAKCU 硬件",
    LogiDriver: "罗技驱动",
    GvInput: "GvInput",
    TencInput: "TencInput",
    NtUserSendInput: "系统注入输入",
    NtUserInjectMouse: "注入鼠标",
    NtUserInjectPointer: "注入指针",
  };
  const ctrl = ctrlMap[s.controller] || s.controller || "—";
  const algoRaw = s.algorithm_name || s.algorithm || "—";
  const algo = algoMap[algoRaw] || algoRaw;
  setText("sCtrl", ctrl + " / " + algo + " 视野=" + (s.fov_px ?? "—"));
  setText("sErr", s.last_error || s.aim_error || "—");
}

async function refreshStatus() {
  if (__statusBusy) return;
  __statusBusy = true;
  try {
    const s = await api("/api/status");
    applyStatus(s);
    if (s.running && s.capture_ok) refreshPreview();
  } catch (e) {
    __engineRunning = false;
    const badge = $("runBadge");
    if (badge) {
      badge.textContent = "离线";
      badge.className = "badge off";
    }
    setText("sErr", String(e));
    if ($("previewHint")) $("previewHint").textContent = "无法连接启动器";
  } finally {
    __statusBusy = false;
  }
}

function formatDetSummary(d) {
  const items = (d && d.items) || [];
  if (!items.length) return "（无检测目标）";
  const max = 12;
  const lines = items.slice(0, max).map((it, i) => {
    const conf = ((it.confidence || 0) * 100).toFixed(0);
    const tid = it.track_id != null && it.track_id >= 0 ? " t" + it.track_id : "";
    return (
      "#" +
      i +
      " c" +
      it.class_id +
      tid +
      " " +
      conf +
      "% [" +
      (it.x || 0).toFixed(3) +
      "," +
      (it.y || 0).toFixed(3) +
      " " +
      (it.w || 0).toFixed(3) +
      "x" +
      (it.h || 0).toFixed(3) +
      "]"
    );
  });
  if (items.length > max) lines.push("… +" + (items.length - max) + " more");
  return lines.join("\n");
}

async function refreshDetections() {
  if (__detBusy || !__engineRunning) return;
  __detBusy = true;
  try {
    const d = await api("/api/detections");
    window.__dets = d;
    drawOverlay(d);
    const now = performance.now();
    if ($("detBox") && now - __detDumpAt >= DET_DUMP_MS) {
      __detDumpAt = now;
      $("detBox").textContent = formatDetSummary(d);
    }
  } catch (_) {
    window.__dets = { items: [] };
    drawOverlay({ items: [] });
    if ($("detBox")) $("detBox").textContent = "（无检测目标）";
  } finally {
    __detBusy = false;
  }
}

/** @deprecated use refreshStatus — kept for call sites after apply/start */
async function refresh() {
  await refreshStatus();
  if (__engineRunning) await refreshDetections();
}

function showPage(page) {
  document.querySelectorAll(".nav-item").forEach((b) => {
    b.classList.toggle("active", b.getAttribute("data-page") === page);
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("active", p.getAttribute("data-page") === page);
  });
  try {
    localStorage.setItem("ya_page", page);
  } catch (_) {}
}

// nav
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => showPage(btn.getAttribute("data-page")));
});

// slot chips
document.querySelectorAll("[data-slot-btn]").forEach((btn) => {
  btn.addEventListener("click", () => setSlot(btn.getAttribute("data-slot-btn")));
});

// algo tabs
document.querySelectorAll(".algo-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const n = Number(tab.getAttribute("data-algo"));
    showAlgoPanel(n, true);
  });
});
const algoSelect = $("algoSelect");
if (algoSelect) {
  algoSelect.addEventListener("change", () => showAlgoPanel(algoSelect.value, false));
}

// range live labels
["confidence", "nms"].forEach((id) => {
  const el = $(id);
  const v = $(id + "Val");
  if (!el || !v) return;
  const sync = () => {
    v.textContent = Number(el.value).toFixed(2);
  };
  el.addEventListener("input", sync);
  sync();
});

// Auto-apply: any data-key control change → PUT (+ cold reload if marked)
document.addEventListener(
  "change",
  (ev) => {
    const el = ev.target;
    if (!el || !el.closest) return;
    if (
      el.id === "modelPick" ||
      el.id === "modelUpload" ||
      el.id === "fileImportCfg" ||
      el.id === "modelFolder" ||
      el.id === "modelRecursive"
    )
      return;
    if (el.hasAttribute("data-key") || el.closest("[data-key]")) {
      const target = el.hasAttribute("data-key") ? el : el.closest("[data-key]");
      scheduleAutoApply(target || el);
    }
  },
  true
);
document.addEventListener(
  "input",
  (ev) => {
    const el = ev.target;
    if (!el || !el.hasAttribute) return;
    if (!el.hasAttribute("data-key")) return;
    if (el.type === "range" || el.type === "number" || el.type === "text") {
      scheduleAutoApply(el);
    }
  },
  true
);

// Model folder + picker
const modelPick = $("modelPick");
if (modelPick) {
  modelPick.addEventListener("change", async () => {
    const path = modelPick.value;
    if (!path) return;
    if ($("modelPath")) $("modelPath").value = path;
    try {
      await putConfig({ model_path: path, infer: { model_path: path } }, { refill: false });
      await api("/api/engine/reload_model", { method: "POST" });
      setMsg("已选用模型并重载: " + path.split(/[/\\]/).pop(), true);
      setTimeout(refresh, 400);
    } catch (e) {
      setMsg(String(e), false);
    }
  });
}
const btnRefreshModels = $("btnRefreshModels");
if (btnRefreshModels) {
  btnRefreshModels.onclick = () => loadModels({ folder: modelFolderValue() });
}
const btnScanFolder = $("btnScanFolder");
if (btnScanFolder) {
  btnScanFolder.onclick = () => scanModelFolder(false);
}
const modelFolderEl = $("modelFolder");
if (modelFolderEl) {
  modelFolderEl.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") {
      ev.preventDefault();
      scanModelFolder(false);
    }
  });
  // do not auto-apply folder text as model_path
  modelFolderEl.addEventListener("change", (ev) => {
    ev.stopPropagation();
  });
}
const modelUpload = $("modelUpload");
if (modelUpload) {
  modelUpload.onchange = async () => {
    const f = modelUpload.files && modelUpload.files[0];
    if (!f) return;
    try {
      setMsg("上传中 " + f.name + " …", true);
      const buf = new Uint8Array(await f.arrayBuffer());
      const res = await fetch("/api/models/upload?name=" + encodeURIComponent(f.name), {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: buf,
      });
      const j = await res.json();
      if (!res.ok || !j.ok) throw new Error(j.error || res.statusText);
      if ($("modelPath")) $("modelPath").value = j.path || j.model_path || f.name;
      await putConfig(
        {
          model_path: j.path || j.model_path,
          infer: { model_path: j.path || j.model_path },
        },
        { refill: false }
      );
      await api("/api/engine/reload_model", { method: "POST" });
      await loadModels({ folder: modelFolderValue() });
      setMsg("上传并重载: " + (j.path || f.name), true);
      setTimeout(refresh, 400);
    } catch (e) {
      setMsg("上传失败: " + e, false);
    } finally {
      modelUpload.value = "";
    }
  };
}

$("btnStart").onclick = async () => {
  try {
    // Ensure infer_enabled is on in config before start
    await putConfig({ infer: { enabled: true }, infer_enabled: true }, { refill: false });
    const res = await api("/api/engine/start", { method: "POST" });
    setMsg(res && res.ok === false ? JSON.stringify(res) : "引擎已启动（推理开启）", true);
    setTimeout(refresh, 300);
  } catch (e) {
    setMsg(String(e), false);
  }
};
$("btnStop").onclick = async () => {
  try {
    // Persist infer off so reload won't immediately YOLO again
    await putConfig({ infer: { enabled: false }, infer_enabled: false }, { refill: false }).catch(
      () => {}
    );
    const res = await api("/api/engine/stop", { method: "POST" });
    setMsg(res && res.ok === false ? JSON.stringify(res) : "引擎已停止（推理已卸载）", true);
    // clear client overlays
    window.__dets = { items: [] };
    drawOverlay(window.__dets);
    refresh();
  } catch (e) {
    setMsg(String(e), false);
  }
};
$("btnReloadCfg").onclick = () => loadConfig();
$("btnApplyAll").onclick = async () => {
  try {
    const page = document.querySelector(".page.active") || document;
    const patch = collectPatch(page);
    await putConfig(patch, { refill: true });
    setMsg("当前页已强制应用", true);
  } catch (e) {
    setMsg(String(e), false);
  }
};

const btnReloadModel = $("btnReloadModel");
if (btnReloadModel) {
  btnReloadModel.onclick = async () => {
    try {
      const page = document.querySelector('.page[data-page="0"]') || document;
      await putConfig(collectPatch(page));
      const res = await api("/api/engine/reload_model", { method: "POST" });
      setMsg(res.ok ? "模型重载已请求" : JSON.stringify(res), !!res.ok);
      setTimeout(refresh, 500);
    } catch (e) {
      setMsg(String(e), false);
    }
  };
}
const btnReloadCapture = $("btnReloadCapture");
if (btnReloadCapture) {
  btnReloadCapture.onclick = async () => {
    try {
      const page = document.querySelector('.page[data-page="0"]') || document;
      await putConfig(collectPatch(page));
      const res = await api("/api/engine/reload_capture", { method: "POST" });
      setMsg(res.ok ? "截图重开已请求" : JSON.stringify(res), !!res.ok);
      setTimeout(refresh, 500);
    } catch (e) {
      setMsg(String(e), false);
    }
  };
}

const btnExportCfg = $("btnExportCfg");
if (btnExportCfg) {
  btnExportCfg.onclick = async () => {
    try {
      const cfg = await api("/api/config");
      const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "yolo_aim_config.json";
      a.click();
      URL.revokeObjectURL(a.href);
      setMsg("已导出配置 JSON", true);
    } catch (e) {
      setMsg(String(e), false);
    }
  };
}
const fileImportCfg = $("fileImportCfg");
if (fileImportCfg) {
  fileImportCfg.onchange = async () => {
    const f = fileImportCfg.files && fileImportCfg.files[0];
    if (!f) return;
    try {
      const text = await f.text();
      await putConfig(JSON.parse(text));
      setMsg("已导入并应用", true);
      await loadConfig();
    } catch (e) {
      setMsg(String(e), false);
    } finally {
      fileImportCfg.value = "";
    }
  };
}

const btnTestCtrl = $("btnTestCtrl");
if (btnTestCtrl) {
  btnTestCtrl.onclick = async () => {
    try {
      const typeEl = document.querySelector('[data-key="controller_type"]');
      const portEl = document.querySelector('[data-key="makcu_port"]');
      const baudEl = document.querySelector('[data-key="makcu_baud_rate"]');
      const logiEl = document.querySelector('[data-key="logi_driver_type"]');
      const body = JSON.stringify({
        controller: CTRL_NAMES[Number(typeEl?.value || 0)] || "WindowsAPI",
        makcuPort: portEl?.value || "COM5",
        makcuBaudRate: Number(baudEl?.value || 4000000),
        logiDriverType: Number(logiEl?.value || 0),
      });
      const res = await api("/api/controller/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      setMsg(res.message || (res.ok ? "连接 OK" : "失败"), !!res.ok);
    } catch (e) {
      setMsg(String(e), false);
    }
  };
}

/* ===== Dual control: range + number share one data-key ===== */
function guessDualRange(num) {
  const stepAttr = num.getAttribute("step");
  const step = stepAttr != null && stepAttr !== "" ? Number(stepAttr) : 1;
  let min = num.hasAttribute("min") ? Number(num.min) : NaN;
  let max = num.hasAttribute("max") ? Number(num.max) : NaN;
  const v = Number(num.value);
  const key = (num.getAttribute("data-key") || "").toLowerCase();
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    // heuristics by key / step
    if (/percent|ratio|alpha|weight|scale|curvature|random|gain_rate|smoothing|damping/.test(key)) {
      min = 0;
      max = /percent/.test(key) ? 100 : 1;
    } else if (/radius|pixel|move|offset|thickness|width|height|area|frames|interval|delay|cooldown|duration|speed|points|step|threads|resolution|size/.test(key)) {
      min = 0;
      max = Math.max(100, Math.abs(v || 0) * 4, 800);
    } else if (step > 0 && step < 1) {
      min = 0;
      max = Math.max(2, Math.abs(v || 1) * 5);
    } else {
      min = 0;
      max = Math.max(100, Math.abs(v || 10) * 4);
    }
    if (v < 0 || /offset|error/.test(key)) {
      min = -max;
    }
  }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max) || max <= min) max = min + 100;
  // expand if current value outside
  if (Number.isFinite(v)) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max, step: Number.isFinite(step) && step > 0 ? step : 1 };
}

function upgradePlainNumbersToDual(root) {
  const scope = root || document;
  scope.querySelectorAll('input[type="number"][data-key]').forEach((num) => {
    if (num.closest(".dual-control")) return;
    if (num.disabled) return;
    // skip very large free-text-like ints (baud, paths handled elsewhere)
    const key = num.getAttribute("data-key") || "";
    if (/baud|port$|path|template/.test(key) && !num.hasAttribute("min")) return;
    const field = num.closest(".field");
    if (!field || field.classList.contains("check")) return;
    const { min, max, step } = guessDualRange(num);
    const wrap = document.createElement("div");
    wrap.className = "dual-control";
    const range = document.createElement("input");
    range.type = "range";
    range.min = String(min);
    range.max = String(max);
    range.step = String(step);
    range.value = num.value || String(min);
    num.parentNode.insertBefore(wrap, num);
    wrap.appendChild(range);
    wrap.appendChild(num);
  });
}

function wireDualControls(root) {
  const scope = root || document;
  upgradePlainNumbersToDual(scope);
  scope.querySelectorAll(".dual-control").forEach((wrap) => {
    if (wrap.dataset.wired === "1") return;
    const range = wrap.querySelector('input[type="range"]');
    const num = wrap.querySelector('input[type="number"]');
    if (!range || !num) return;
    wrap.dataset.wired = "1";
    // Prefer number for collectPatch (has data-key); strip from range to avoid double patch
    if (range.hasAttribute("data-key") && num.hasAttribute("data-key")) {
      range.removeAttribute("data-key");
      range.removeAttribute("data-path");
      range.removeAttribute("data-slot");
      range.removeAttribute("data-cold");
    } else if (range.hasAttribute("data-key") && !num.hasAttribute("data-key")) {
      ["data-key", "data-path", "data-slot", "data-cold"].forEach((a) => {
        if (range.hasAttribute(a)) num.setAttribute(a, range.getAttribute(a));
        range.removeAttribute(a);
      });
    }
    // ensure range min/max cover value
    const nv = Number(num.value);
    if (Number.isFinite(nv)) {
      if (nv < Number(range.min)) range.min = String(nv);
      if (nv > Number(range.max)) range.max = String(nv);
    }
    const sync = (from, to) => {
      const v = from.value;
      if (to.value !== v) to.value = v;
      const span = wrap.querySelector(".val");
      if (span) {
        const n = Number(v);
        span.textContent = Number.isFinite(n)
          ? n.toFixed(String(from.step || "1").includes(".") ? 2 : 0)
          : v;
      }
    };
    range.addEventListener("input", () => {
      sync(range, num);
      scheduleAutoApply(num);
    });
    num.addEventListener("input", () => {
      // expand range if user types outside
      const n = Number(num.value);
      if (Number.isFinite(n)) {
        if (n < Number(range.min)) range.min = String(n);
        if (n > Number(range.max)) range.max = String(n);
      }
      sync(num, range);
      scheduleAutoApply(num);
    });
    num.addEventListener("change", () => {
      sync(num, range);
      scheduleAutoApply(num);
    });
    sync(num.value !== "" ? num : range, num.value !== "" ? range : num);
  });
}

/* ===== OBS labels + help tooltips (same long_description as filter_properties) ===== */
function applyObsHelp(root) {
  const HELP = window.OBS_HELP || {};
  const LABEL = window.OBS_LABEL || {};
  const scope = root || document;
  // Strip ALL native tooltips on controls — they flash while dragging ranges.
  scope.querySelectorAll("[title]").forEach((el) => {
    if (el.classList && el.classList.contains("help-tip")) return;
    // keep a few intentional chrome titles
    if (el.id === "btnApplyAll" || el.id === "btnScanFolder") return;
    el.removeAttribute("title");
  });
  scope.querySelectorAll("[data-key]").forEach((el) => {
    const key = el.getAttribute("data-key");
    if (!key) return;
    const help = HELP[key];
    const lab = LABEL[key];
    if (el.hasAttribute("title")) el.removeAttribute("title");
    const field = el.closest(".field");
    if (!field) return;
    if (field.dataset.obsDecorated === "1") return;
    field.dataset.obsDecorated = "1";
    const labelSpan =
      field.querySelector(":scope > span.label") ||
      field.querySelector("span.label");
    // 全中文：只替换中文标签，不再附加英文配置键
    if (labelSpan && lab) {
      const tipKeep = labelSpan.querySelector(".help-tip");
      labelSpan.textContent = lab;
      if (tipKeep) labelSpan.appendChild(tipKeep);
    }
    // 去掉任何残留的英文 key 角标
    field.querySelectorAll("span.key").forEach((k) => k.remove());
    if (help) {
      let tip = field.querySelector(".help-tip");
      if (!tip) {
        tip = document.createElement("button");
        tip.type = "button";
        tip.className = "help-tip";
        tip.setAttribute("aria-label", "参数说明");
        tip.setAttribute("tabindex", "0");
        // CLICK-ONLY bubble (no hover show — user: only on ! icon click)
        tip.innerHTML =
          '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M12 10.5v6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="7.2" r="1.1" fill="currentColor"/></svg>';
        const bubble = document.createElement("div");
        bubble.className = "help-bubble";
        bubble.textContent = help;
        tip.appendChild(bubble);
        tip.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          document.querySelectorAll(".help-tip.open").forEach((t) => {
            if (t !== tip) t.classList.remove("open");
          });
          tip.classList.toggle("open");
        });
        if (labelSpan) labelSpan.appendChild(tip);
        else field.appendChild(tip);
      }
    }
  });
  scope.querySelectorAll(".dual-control input[title], input[type=range][title]").forEach((el) => {
    el.removeAttribute("title");
  });
  if (!window.__yaHelpOutside) {
    window.__yaHelpOutside = true;
    document.addEventListener("click", (e) => {
      if (e.target && e.target.closest && e.target.closest(".help-tip")) return;
      document.querySelectorAll(".help-tip.open").forEach((t) => t.classList.remove("open"));
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        document.querySelectorAll(".help-tip.open").forEach((t) => t.classList.remove("open"));
      }
    });
    // While dragging any range, suppress auto-apply toast spam
    document.addEventListener(
      "pointerdown",
      (e) => {
        const t = e.target;
        if (t && (t.type === "range" || (t.closest && t.closest(".dual-control")))) {
          __msgQuietUntil = performance.now() + 800;
        }
      },
      true
    );
  }
}

// restore last page
try {
  const p = localStorage.getItem("ya_page");
  if (p != null) showPage(p);
} catch (_) {}

// Defer heavy DOM upgrades so first paint isn't blocked
requestAnimationFrame(() => {
  wireDualControls(document);
  applyObsHelp(document);
});
loadConfig();
refreshStatus();
setInterval(refreshStatus, STATUS_MS);
setInterval(refreshDetections, DET_MS);
// Pause expensive polls when tab hidden; flush pending auto-apply so last edits aren't lost
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    flushPendingConfig();
    return;
  }
  refreshStatus();
  if (__engineRunning) refreshDetections();
});

/** Flush debounced PUT immediately (close tab / hide / beforeunload). */
function flushPendingConfig() {
  if (__suppressAuto) return;
  if (__applyTimer) {
    clearTimeout(__applyTimer);
    __applyTimer = null;
  }
  // Prefer full active-page snapshot so multi-field edits aren't partial
  try {
    const page = document.querySelector(".page.active") || document;
    const patch = collectPatch(page);
    if (!patch || !Object.keys(patch).length) return;
    // keepalive so request survives page unload
    const body = JSON.stringify(patch);
    if (navigator.sendBeacon) {
      const blob = new Blob([body], { type: "application/json" });
      navigator.sendBeacon("/api/config", blob);
      setMsg("已保存配置", true);
    } else {
      // sync XHR fallback (deprecated but works on unload)
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", "/api/config", false);
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.send(body);
    }
  } catch (e) {
    console.warn("flushPendingConfig", e);
  }
}

window.addEventListener("pagehide", flushPendingConfig);
window.addEventListener("beforeunload", flushPendingConfig);
