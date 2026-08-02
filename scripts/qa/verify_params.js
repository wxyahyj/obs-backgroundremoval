// 全参数生效验证:遍历 ConfigDocument 所有叶子字段,逐个改值 → PUT → GET → 断言回读
// 用法: node verify_params.js <host> (需 host 已运行)
"use strict";
const HOST = process.argv[2] || "http://127.0.0.1:17890";

async function get(path) {
  const r = await fetch(HOST + path);
  return r.json();
}
async function put(path, body) {
  const r = await fetch(HOST + path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
  return r.json();
}

// 收集所有叶子路径 + 类型
function leafPaths(obj, prefix, out) {
  for (const [k, v] of Object.entries(obj)) {
    const p = prefix ? prefix + "." + k : k;
    if (v !== null && typeof v === "object") leafPaths(v, p, out);
    else out.push({ path: p, type: typeof v, value: v });
  }
  return out;
}

// 改值:数字+7、bool翻转、字符串+后缀
function mutate(v, type) {
  if (type === "number") return v + 7;
  if (type === "boolean") return !v;
  return v + "_T";
}

// 只读输出字段(序列化写出,不接受输入)
function isReadonly(path) {
  return /\.index$/.test(path); // slots[i].index 由后端生成
}

async function main() {
  const cfg0 = (await get("/api/config")).config;
  const leaves = leafPaths(cfg0, "", []);
  console.log("总字段数:", leaves.length);

  let pass = 0, fail = 0;
  const failed = [];
  for (const leaf of leaves) {
    if (isReadonly(leaf.path)) continue; // 只读字段跳过
    const seg = leaf.path.split(".");
    // slots 是数组:完整数组 patch(改一个元素)
    const isSlot = seg[0] === "aim" && seg[1] === "slots" && seg.length >= 4;
    let patch = {};
    const newVal = mutate(leaf.value, leaf.type);

    if (isSlot) {
      const slotIdx = Number(seg[2]);
      const slots = JSON.parse(JSON.stringify(cfg0.aim.slots));
      let node = slots[slotIdx];
      for (let i = 3; i < seg.length - 1; i++) node = node[seg[i]];
      node[seg[seg.length - 1]] = newVal;
      patch = { aim: { slots } };
    } else {
      let node = patch;
      for (let i = 0; i < seg.length - 1; i++) {
        node[seg[i]] = {};
        node = node[seg[i]];
      }
      node[seg[seg.length - 1]] = newVal;
    }

    const r = await put("/api/config", patch);
    if (!r.ok) {
      fail++;
      failed.push(leaf.path + " PUT失败:" + (r.error || ""));
      continue;
    }
    // 回读验证
    const cfg = (await get("/api/config")).config;
    let got = cfg;
    let ok = true;
    for (const s of seg) {
      if (got === undefined || got === null) { ok = false; break; }
      got = got[s];
    }
    // 浮点近似比较
    let match = false;
    if (typeof newVal === "number" && typeof got === "number") {
      match = Math.abs(got - newVal) < 1e-3;
    } else {
      match = JSON.stringify(got) === JSON.stringify(newVal);
    }
    if (ok && match) pass++;
    else {
      fail++;
      failed.push(leaf.path + " 期望=" + JSON.stringify(newVal) + " 实得=" + JSON.stringify(got));
    }
  }

  // 恢复初始配置(测试会污染 user.json)
  await put("/api/config", cfg0);
  console.log("配置已恢复为初始值");

  console.log(`\n结果: PASS=${pass} FAIL=${fail}`);
  if (failed.length) {
    console.log("失败字段:");
    failed.forEach((f) => console.log("  " + f));
  } else {
    console.log("ALL PARAMS ROUNDTRIP OK");
  }
}

main().catch((e) => { console.error("ERR", e); process.exit(1); });
