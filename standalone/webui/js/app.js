/* YoloAim WebUI — SPA:状态轮询 + 自动表单渲染配置 */
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

async function refreshStatus() {
  try {
    const s = await api.get("/api/status");
    setVal("st-running", s.running ? "运行中" : "已停止", s.running ? "var(--ok)" : "var(--danger)");
    setVal("st-fps", s.fps.toFixed(1));
    setVal("st-dets", String(s.detections));
    setVal("st-infer", s.infer_ms.toFixed(2) + " ms");
    setVal("st-slot", s.aim.slot >= 0 ? "槽 " + s.aim.slot : "-");
    setVal("st-ctrl", s.aim.controller_ok ? "OK" : "失败", s.aim.controller_ok ? "var(--ok)" : "var(--danger)");
  } catch (e) { /* 服务器未启动 */ }
}

function setVal(id, text, color) {
  const el = document.getElementById(id);
  el.textContent = text;
  if (color) el.style.color = color;
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
  const r = await api.put("/api/controller/test", { type: "WindowsAPI" });
  setMsg(r.ok ? "控制器测试通过" : "测试失败: " + r.error, r.ok ? "ok" : "err");
});

setInterval(refreshStatus, 1000);
setInterval(refreshDets, 2000);

/* ---------- 配置页:自动表单 ---------- */
let configDoc = null;

async function loadConfig() {
  try {
    const r = await api.get("/api/config");
    configDoc = r.config || {};
    renderConfig();
  } catch (e) {
    setMsg("配置加载失败", "err");
  }
}

// 字段友好名(蛇形 → 空格)
function fieldLabel(key) {
  return key.replace(/_/g, " ");
}

// 递归渲染对象 → 折叠 section + 字段
function renderSection(container, obj, path, title) {
  const sec = document.createElement("div");
  sec.className = "section";

  const head = document.createElement("div");
  head.className = "sec-head";
  const span = document.createElement("span");
  span.textContent = title || fieldLabel(path.split(".").pop());
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
  buildFields(body, obj, path);
  sec.appendChild(body);
  container.appendChild(sec);
}

function buildFields(container, obj, path) {
  for (const [key, val] of Object.entries(obj)) {
    const full = path ? path + "." + key : key;
    if (val === null || val === undefined) continue;
    if (typeof val === "object") {
      if (Array.isArray(val)) {
        // 数组:每个元素一个子 section
        val.forEach((item, i) => {
          if (item && typeof item === "object") {
            renderSection(container, item, full + "." + i, key + "[" + i + "]");
          } else {
            renderTextarea(container, full + "." + i, JSON.stringify(item), key + "[" + i + "]");
          }
        });
      } else {
        renderSection(container, val, full, key);
      }
      continue;
    }
    renderField(container, key, val, full);
  }
}

function renderField(container, key, val, path) {
  const div = document.createElement("div");
  if (typeof val === "boolean") {
    div.className = "field checkbox";
    const label = document.createElement("label");
    label.textContent = fieldLabel(key);
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.path = path;
    input.checked = val;
    div.appendChild(input);
    div.appendChild(label);
  } else if (typeof val === "number") {
    div.className = "field";
    const label = document.createElement("label");
    label.textContent = fieldLabel(key);
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = val;
    input.dataset.path = path;
    div.appendChild(label);
    div.appendChild(input);
  } else {
    div.className = "field";
    const label = document.createElement("label");
    label.textContent = fieldLabel(key);
    const input = document.createElement("input");
    input.type = "text";
    input.value = String(val);
    input.dataset.path = path;
    div.appendChild(label);
    div.appendChild(input);
  }
  container.appendChild(div);
}

function renderTextarea(container, path, text, title) {
  const div = document.createElement("div");
  div.className = "field";
  const label = document.createElement("label");
  label.textContent = title;
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.dataset.path = path;
  div.appendChild(label);
  div.appendChild(ta);
  container.appendChild(div);
}

function renderConfig() {
  const root = document.getElementById("config-root");
  root.innerHTML = "";
  for (const [key, val] of Object.entries(configDoc)) {
    if (val && typeof val === "object" && !Array.isArray(val)) {
      renderSection(root, val, key, key);
    }
  }
}

// 从表单收集 → 嵌套 JSON(只含渲染过的路径)
function collectConfig() {
  const out = {};
  document.querySelectorAll("#config-root [data-path]").forEach((el) => {
    const path = el.dataset.path.split(".");
    let node = out;
    for (let i = 0; i < path.length - 1; i++) {
      const k = path[i];
      if (!(k in node) || node[k] === null || typeof node[k] !== "object") node[k] = {};
      node = node[k];
    }
    const key = path[path.length - 1];
    if (el.type === "checkbox") node[key] = el.checked;
    else if (el.type === "number") {
      const v = el.value;
      node[key] = v === "" ? 0 : Number(v);
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

/* ---------- 预览页 ---------- */
setInterval(() => {
  const img = document.getElementById("preview-img");
  if (document.getElementById("page-preview").classList.contains("active")) {
    img.src = "/api/preview.bmp?t=" + Date.now();
  }
}, 500);

refreshStatus();
