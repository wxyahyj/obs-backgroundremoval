/* YoloAim WebUI — SPA:状态轮询 + OBS 分类中文配置 */
"use strict";

const api = {
  get: (p) => fetch(p).then((r) => r.json()),
  post: (p) => fetch(p, { method: "POST" }).then((r) => r.json()),
  put: (p, body) =>
    fetch(p, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: typeof body === "string" ? body : JSON.stringify(body),
    }).then((r) => r.json()),
};

/* ---------- 页面切换 ---------- */
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    document.getElementById("page-" + btn.dataset.page).classList.add("active");
    if (btn.dataset.page === "config") loadConfig();
  });
});

/* ---------- 状态页 ---------- */
function setMsg(text, cls) {
  const m = document.getElementById("status-msg");
  m.textContent = text || "";
  m.className = "msg" + (cls ? " " + cls : "");
}

function setVal(id, text, color) {
  const el = document.getElementById(id);
  el.textContent = text;
  if (color) el.style.color = color;
}

async function refreshStatus() {
  try {
    const s = await api.get("/api/status");
    setVal("st-running", s.running ? "运行中" : "已停止", s.running ? "var(--ok)" : "var(--danger)");
    setVal("st-fps", s.fps.toFixed(1));
    setVal("st-dets", String(s.detections));
    setVal("st-infer", s.infer_ms.toFixed(2) + " ms");
    setVal("st-slot", s.aim.slot >= 0 ? "槽 " + s.aim.slot : "-");
    setVal("st-ctrl", s.aim.controller_ok ? "OK" : "失败", s.aim.controller_ok ? "var(--ok)" : "var(--danger)");
  } catch (e) {}
}

async function refreshDets() {
  try {
    const d = await api.get("/api/detections");
    const tb = document.querySelector("#dets-table tbody");
    tb.innerHTML = "";
    (d.detections || []).slice(0, 12).forEach((x) => {
      const tr = document.createElement("tr");
      [x.track_id, x.class, (x.confidence * 100).toFixed(0) + "%",
       x.center_x.toFixed(3), x.center_y.toFixed(3),
       x.width.toFixed(3), x.height.toFixed(3)].forEach((v) => {
        const td = document.createElement("td");
        td.textContent = v;
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
  } catch (e) {}
}

document.getElementById("btn-start").addEventListener("click", async () => {
  const r = await api.post("/api/engine/start");
  setMsg(r.ok ? "引擎已启动" : "启动失败: " + r.error, r.ok ? "ok" : "err");
});
document.getElementById("btn-stop").addEventListener("click", async () => {
  const r = await api.post("/api/engine/stop");
  setMsg(r.ok ? "引擎已停止" : "错误: " + r.error, r.ok ? "ok" : "err");
});
document.getElementById("btn-reload-model").addEventListener("click", async () => {
  const r = await api.post("/api/engine/reload_model");
  setMsg(r.ok ? "模型重载请求已发出" : "错误: " + r.error, r.ok ? "ok" : "err");
});
document.getElementById("btn-reload-cap").addEventListener("click", async () => {
  const r = await api.post("/api/engine/reload_capture");
  setMsg(r.ok ? "截图重载请求已发出" : "错误: " + r.error, r.ok ? "ok" : "err");
});
document.getElementById("btn-test-ctrl").addEventListener("click", async () => {
  // 遍历全部后端测试(MAKCU 无串口会报未连接,属正常)
  const types = [
    [0, "WindowsAPI"], [1, "MAKCU"], [2, "LogiDriver"], [3, "GvInput"],
    [4, "TencInput"], [5, "NtUserSendInput"], [6, "NtUserInjectMouse"],
    [7, "NtUserInjectPointer"],
  ];
  let lines = [];
  for (const [id, name] of types) {
    const r = await api.put("/api/controller/test", {
      type: String(id), makcu_port: "COM5", makcu_baud_rate: 115200,
      logi_driver_type: 0,
    });
    lines.push(name + ": " + (r.ok ? "✓" : "✗ " + (r.error || "")));
  }
  setMsg(lines.join("\n"), "ok");
});

setInterval(refreshStatus, 1000);
setInterval(refreshDets, 2000);

/* ---------- 配置页:OBS 分类 + 中文 ---------- */
let configDoc = null;
let currentPage = 0;
let modelList = [];

async function loadConfig() {
  try {
    // 确保模型列表已加载(下拉渲染依赖),失败不阻塞配置
    if (modelList.length === 0) await loadModels();
    // 当前模型类别数(复选框组用)
    try {
      const st = await api.get("/api/status");
      window.curNumClasses = st.num_classes || 0;
    } catch (e) {}
    const r = await api.get("/api/config");
    configDoc = r.config || r;
    renderConfig();
  } catch (e) {
    setMsg("配置加载失败", "err");
  }
}

async function loadModels() {
  try {
    const r = await api.get("/api/models");
    modelList = r.models || [];
  } catch (e) {}
}

// 沿路径取配置值;{i} 已被调用方替换
function getByPath(doc, path) {
  let node = doc;
  for (const k of path) {
    if (node === null || node === undefined) return undefined;
    node = node[k];
  }
  return node;
}

function setByPath(doc, path, value) {
  let node = doc;
  for (let i = 0; i < path.length - 1; i++) {
    if (node[path[i]] === null || node[path[i]] === undefined ||
        typeof node[path[i]] !== "object") node[path[i]] = {};
    node = node[path[i]];
  }
  node[path[path.length - 1]] = value;
}

function renderConfig() {
  const root = document.getElementById("config-root");
  root.innerHTML = "";

  // 页 tab
  const tabs = document.createElement("div");
  tabs.className = "page-tabs";
  OBS_PAGES.forEach((page, idx) => {
    const b = document.createElement("button");
    b.className = "page-tab" + (idx === currentPage ? " active" : "");
    b.textContent = page.name;
    b.addEventListener("click", () => {
      currentPage = idx;
      renderConfig();
    });
    tabs.appendChild(b);
  });
  root.appendChild(tabs);

  const page = OBS_PAGES[currentPage];
  const body = document.createElement("div");
  body.className = "page-body";

  // 槽页:当前槽选择器
  if (page.slot) {
    const slotBar = document.createElement("div");
    slotBar.className = "slot-bar";
    slotBar.appendChild(document.createTextNode("当前配置: "));
    for (let i = 0; i < 5; i++) {
      const b = document.createElement("button");
      b.className = "btn" + (i === currentSlot() ? " primary" : "");
      b.textContent = "配置 " + i;
      b.addEventListener("click", () => {
        const doc = collectConfig();
        setByPath(doc, ["aim", "config_select"], i);
        api.put("/api/config", doc).then((r) => { if (r.ok) loadConfig(); });
      });
      slotBar.appendChild(b);
    }
    body.appendChild(slotBar);
  }

  // 组
  page.groups.forEach((group) => renderGroup(body, group));
  root.appendChild(body);
}

function currentSlot() {
  const v = getByPath(configDoc, ["aim", "config_select"]);
  return typeof v === "number" && v >= 0 && v < 5 ? v : 0;
}

function renderGroup(container, group) {
  const sec = document.createElement("div");
  sec.className = "section";
  const head = document.createElement("div");
  head.className = "sec-head";
  const span = document.createElement("span");
  span.textContent = group.name;
  const caret = document.createElement("span");
  caret.className = "caret";
  caret.textContent = "▾";
  head.appendChild(span);
  head.appendChild(caret);
  head.addEventListener("click", () => {
    sec.classList.toggle("collapsed");
    caret.textContent = sec.classList.contains("collapsed") ? "▸" : "▾";
  });
  sec.appendChild(head);

  const body = document.createElement("div");
  body.className = "sec-body";
  group.fields.forEach(([obsKey, path]) => renderField(body, obsKey, path));
  sec.appendChild(body);
  container.appendChild(sec);
}

function renderField(container, obsKey, pathTemplate) {
  // 槽路径替换当前槽 + 补 aim/slots 前缀
  const slot = currentSlot();
  const path = pathTemplate[0] === "{i}"
    ? ["aim", "slots", slot, ...pathTemplate.slice(1)]
    : pathTemplate.map((p) => (p === "{i}" ? slot : p));
  const val = getByPath(configDoc, path);

  const div = document.createElement("div");
  const label = document.createElement("label");
  label.textContent = fieldLabel(obsKey);

  // 模型路径:下拉选择(D:/AI + exe/models 扫描)+ 版本 + 加载按钮
  if (obsKey === "model_path") {
    div.className = "field";
    const row = document.createElement("div");
    row.className = "model-row";
    const sel = document.createElement("select");
    sel.dataset.path = path.join(".");
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "— 选择模型 —";
    sel.appendChild(placeholder);
    modelList.forEach((m) => {
      const o = document.createElement("option");
      o.value = m.path;
      o.textContent = m.name;
      sel.appendChild(o);
    });
    sel.value = val || "";
    const versionSel = document.createElement("select");
    versionSel.dataset.path = path.slice(0, -1).concat("model_version").join(".");
    (FIELD_OPTIONS.model_version || []).forEach(([text, v]) => {
      const o = document.createElement("option");
      o.value = v;
      o.dataset.type = "number";
      o.textContent = text;
      versionSel.appendChild(o);
    });
    versionSel.value = getByPath(configDoc, path.slice(0, -1).concat("model_version"));
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn primary";
    btn.textContent = "加载模型";
    btn.addEventListener("click", async () => {
      if (!sel.value) {
        setMsg("请先选择模型", "err");
        return;
      }
      const doc = collectConfig();
      const r = await api.put("/api/config", doc);
      if (r.ok) {
        const rr = await api.post("/api/engine/reload_model");
        setMsg(rr.ok ? "模型加载中…" : "重载失败: " + rr.error, rr.ok ? "ok" : "err");
      }
    });
    row.appendChild(sel);
    row.appendChild(versionSel);
    row.appendChild(btn);
    div.appendChild(label);
    div.appendChild(row);
    container.appendChild(div);
    return;
  }

  // 下拉框(obs 键有选项表)
  const options = FIELD_OPTIONS[obsKey];
  if (options) {
    div.className = "field";
    const sel = document.createElement("select");
    sel.dataset.path = path.join(".");
    options.forEach(([text, optVal]) => {
      const o = document.createElement("option");
      o.value = optVal;
      o.dataset.type = typeof optVal;
      o.textContent = text;
      sel.appendChild(o);
    });
    sel.value = val;
    div.appendChild(label);
    div.appendChild(sel);
    container.appendChild(div);
    return;
  }

  // 滑块(obs 键有范围表)
  const slider = FIELD_SLIDERS[obsKey];
  if (slider && typeof val === "number") {
    div.className = "field slider-field";
    const [min, max, step] = slider;
    const row = document.createElement("div");
    row.className = "slider-row";
    const input = document.createElement("input");
    input.type = "range";
    input.min = min;
    input.max = max;
    input.step = step || "any";
    input.value = val;
    input.dataset.path = path.join(".");
    const valSpan = document.createElement("span");
    valSpan.className = "slider-val";
    valSpan.textContent = Number(val).toFixed(step && step < 1 ? 2 : 0);
    input.addEventListener("input", () => {
      valSpan.textContent = Number(input.value).toFixed(step && step < 1 ? 2 : 0);
    });
    row.appendChild(input);
    row.appendChild(valSpan);
    div.appendChild(label);
    div.appendChild(row);
    container.appendChild(div);
    return;
  }

  if (typeof val === "boolean") {
    div.className = "field checkbox";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = val;
    input.dataset.path = path.join(".");
    div.appendChild(input);
    div.appendChild(label);
  } else if (typeof val === "number") {
    div.className = "field";
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = val;
    input.dataset.path = path.join(".");
    div.appendChild(label);
    div.appendChild(input);
  } else if (Array.isArray(val)) {
    // 目标类别:自动识别类别数 → 复选框组(勾选 = 过滤,全选/空 = 全部)
    const nc = window.curNumClasses || 0;
    if (obsKey === "target_classes_text" && nc > 0) {
      div.className = "field checkbox-group";
      const label = document.createElement("div");
      label.className = "group-label";
      label.textContent = fieldLabel(obsKey) + " (模型 " + nc + " 类)";
      div.appendChild(label);
      const box = document.createElement("div");
      box.className = "checkboxes";
      const sel = new Set(val);
      for (let i = 0; i < nc; i++) {
        const cdiv = document.createElement("label");
        cdiv.className = "checkbox-item";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = sel.has(i);
        cb.dataset.path = path.join(".");
        cb.dataset.cls = String(i);
        cdiv.appendChild(cb);
        cdiv.appendChild(document.createTextNode("类别 " + i));
        box.appendChild(cdiv);
      }
      div.appendChild(box);
      container.appendChild(div);
      return;
    }
    div.className = "field";
    const input = document.createElement("input");
    input.type = "text";
    // 目标类别:逗号分隔文本(空 = 全部),OBS 风格
    input.value = val.length ? val.join(",") : "";
    input.placeholder = "留空 = 全部类别,如: 0,1,2";
    input.dataset.path = path.join(".");
    input.dataset.arr = "1";
    div.appendChild(label);
    div.appendChild(input);
  } else {
    div.className = "field";
    const input = document.createElement("input");
    input.type = "text";
    input.value = String(val === undefined ? "" : val);
    input.dataset.path = path.join(".");
    div.appendChild(label);
    div.appendChild(input);
  }
  container.appendChild(div);
}

// 从表单收集 → 嵌套 JSON(仅渲染过的路径)
function collectConfig() {
  const out = {};
  // 类别复选框组:先聚合(空 = 全部)
  const clsMap = {};
  document.querySelectorAll("#config-root [data-cls]").forEach((el) => {
    const p = el.dataset.path;
    if (!clsMap[p]) clsMap[p] = [];
    if (el.checked) clsMap[p].push(Number(el.dataset.cls));
  });
  document.querySelectorAll("#config-root [data-path]").forEach((el) => {
    const path = el.dataset.path.split(".");
    let node = out;
    for (let i = 0; i < path.length - 1; i++) {
      const k = path[i];
      if (!(k in node) || node[k] === null || typeof node[k] !== "object") node[k] = {};
      node = node[k];
    }
    const key = path[path.length - 1];
    if (el.dataset.cls) {
      // 复选框组:由 clsMap 统一写(空数组 = 不过滤全部)
      node[key] = clsMap[el.dataset.path] || [];
    } else if (el.type === "checkbox") node[key] = el.checked;
    else if (el.type === "number" || el.type === "range") {
      const v = el.value;
      node[key] = v === "" ? 0 : Number(v);
    } else if (el.tagName === "SELECT") {
      const opt = el.selectedOptions[0];
      const t = opt && opt.dataset.type;
      node[key] = t === "number" ? Number(el.value) : el.value;
    } else if (el.dataset.arr) {
      // 逗号分隔数组(目标类别)
      node[key] = el.value.trim() === ""
        ? []
        : el.value.split(",").map((s) => Number(s.trim())).filter((n) => !isNaN(n));
    } else if (el.tagName === "TEXTAREA") {
      try { node[key] = JSON.parse(el.value); } catch (e) { node[key] = el.value; }
    } else node[key] = el.value;
  });
  return out;
}

document.getElementById("btn-save").addEventListener("click", async () => {
  const doc = collectConfig();
  const r = await api.put("/api/config", doc);
  setMsg(r.ok ? "配置已保存并应用" : "保存失败: " + r.error, r.ok ? "ok" : "err");
  if (r.ok) loadConfig();
});
document.getElementById("btn-reload").addEventListener("click", loadConfig);
loadModels();

/* ---- OBS 场景导入 ---- */
document.getElementById("btn-import-obs").addEventListener("click", () => {
  document.getElementById("file-import-obs").click();
});
document.getElementById("file-import-obs").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const text = await file.text();
  const r = await api.put("/api/config/import_obs", text);
  setMsg(r.ok ? "导入成功: 发现 " + r.filters_imported + " 个滤镜" : "导入失败: " + r.error,
         r.ok ? "ok" : "err");
  if (r.ok) loadConfig();
  e.target.value = "";
});

/* ---------- 预览页 ---------- */
let pickedColor = null;

document.getElementById("preview-img").addEventListener("click", async (e) => {
  const img = e.currentTarget;
  const rect = img.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top) / rect.height;
  const r = await api.put("/api/crosshair/pick", { x: nx, y: ny });
  if (r.ok) {
    pickedColor = { r: r.r, g: r.g, b: r.b };
    const sw = document.getElementById("pick-swatch");
    sw.style.background = `rgb(${r.r},${r.g},${r.b})`;
    document.getElementById("pick-rgb").textContent =
      `#${[r.r, r.g, r.b].map((v) => v.toString(16).padStart(2, "0")).join("")} ` +
      `(r=${r.r} g=${r.g} b=${r.b})`;
    document.getElementById("btn-pick-apply").disabled = false;
  }
});

document.getElementById("btn-pick-apply").addEventListener("click", async () => {
  if (!pickedColor) return;
  const doc = collectConfig();
  if (!doc.aim) doc.aim = {};
  doc.aim.crosshair_manual_r = pickedColor.r;
  doc.aim.crosshair_manual_g = pickedColor.g;
  doc.aim.crosshair_manual_b = pickedColor.b;
  doc.aim.crosshair_enabled = true;
  const r = await api.put("/api/config", doc);
  setMsg(r.ok ? "准星颜色已应用" : "应用失败: " + r.error, r.ok ? "ok" : "err");
  if (r.ok) loadConfig();
});

setInterval(() => {
  const img = document.getElementById("preview-img");
  if (document.getElementById("page-preview").classList.contains("active")) {
    img.src = "/api/preview.bmp?t=" + Date.now();
  }
}, 500);

refreshStatus();
