#!/bin/bash
# 综合功能验证:启动 host → 逐 API 断言 → 报告
set -u
HOST="http://127.0.0.1:17890"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXE="${1:-$ROOT/standalone/build/RelWithDebInfo/yolo_host.exe}"
FAIL=0
PASS=0

check() { # name cond
  if [ "$2" = "1" ]; then echo "  [PASS] $1"; PASS=$((PASS+1));
  else echo "  [FAIL] $1"; FAIL=$((FAIL+1)); fi
}

cd "$ROOT"
taskkill //F //IM yolo_host.exe >/dev/null 2>&1
"$EXE" > /tmp/verify.log 2>&1 &
HPID=$!
sleep 8

echo "== F1 引擎 =="
R=$(curl -s -m 3 $HOST/api/health)
check "health" "$(echo "$R" | grep -c '"ok":true')"
S=$(curl -s -m 3 $HOST/api/status)
check "running" "$(echo "$S" | grep -c '"running":true')"
check "capture_ok" "$(echo "$S" | grep -c '"capture_ok":true')"

echo "== F3/F4 模型 =="
M=$(curl -s -m 5 $HOST/api/models)
N=$(echo "$M" | grep -o '"name"' | wc -l)
check "models 列表非空(>=10)" "$([ "$N" -ge 10 ] && echo 1 || echo 0)"
C=$(curl -s -m 3 $HOST/api/config)
check "config 含 capture/infer/aim" "$(echo "$C" | grep -c '"aim"')"

echo "== F10 配置往返 =="
PUT=$(curl -s -m 3 -X PUT -H "Content-Type: application/json" \
  -d '{"aim":{"fov_radius":150,"slots":[{"enabled":true,"continuous_aim":true,"mc":{"pidPMin":0.2,"controllerType":2,"hotkeyVirtualKey":6}}]}}' $HOST/api/config)
check "PUT ok" "$(echo "$PUT" | grep -c '"ok":true')"
C2=$(curl -s -m 3 $HOST/api/config)
check "fov=150 生效" "$(echo "$C2" | grep -o '"fov_radius":[0-9]*' | head -1 | grep -c '150')"
check "槽0 持续瞄准生效" "$(echo "$C2" | grep -o '"continuous_aim":true' | head -1 | grep -c true)"
check "controllerType=2" "$(echo "$C2" | grep -o '"controllerType":2' | head -1 | grep -c 2)"

echo "== F7/F8 预览取色 =="
BMP=$(curl -s -m 5 -o /dev/null -w "%{http_code}" $HOST/api/preview.bmp)
check "preview.bmp 200" "$([ "$BMP" = "200" ] && echo 1 || echo 0)"
PK=$(curl -s -m 3 -X POST -H "Content-Type: application/json" -d '{"x":0.5,"y":0.5}' $HOST/api/crosshair/pick)
check "取色 RGB" "$(echo "$PK" | grep -c '"ok":true')"

echo "== F6 检测 =="
D=$(curl -s -m 3 $HOST/api/detections)
check "detections 端点" "$(echo "$D" | grep -c '"ok":true')"

echo "== F9 OBS 导入 =="
IMP=$(curl -s -m 3 -X POST -H "Content-Type: application/json" \
  -d '{"scenes":[{"sources":[{"type":"game_capture","filters":[{"type":"yolo-detector-filter","settings":{"confidence_threshold":0.55,"fov_radius":160}}]}]}]}' $HOST/api/config/import_obs)
check "导入成功" "$(echo "$IMP" | grep -c '"ok":true')"

echo "== F12 控制器 =="
CT=$(curl -s -m 5 -X POST -H "Content-Type: application/json" -d '{"type":"WindowsAPI"}' $HOST/api/controller/test)
check "控制器测试" "$(echo "$CT" | grep -c '"ok":true')"

echo "== F16 坐标导出 =="
curl -s -m 3 -X PUT -H "Content-Type: application/json" \
  -d '{"vision":{"export_coordinates":true,"coordinate_output_path":"C:/Users/Administrator/verify_dets.json"}}' $HOST/api/config > /dev/null
sleep 4
check "detections.json 生成" "$([ -f /c/Users/Administrator/verify_dets.json ] && echo 1 || echo 0)"
rm -f /c/Users/Administrator/verify_dets.json

echo "== F2 性能 =="
G=$(echo "$S" | grep -o '"grab_ms":[0-9.]*' | cut -d: -f2)
echo "  grab_ms=$G (ok if <60)"
I=$(echo "$S" | grep -o '"infer_ms":[0-9.]*' | cut -d: -f2)
echo "  infer_ms=$I"

echo "== 日志异常 =="
E=$(grep -a "FATAL\|Exception\|FAILED\|crash" /tmp/verify.log | wc -l)
check "无 FATAL/异常" "$([ "$E" = "0" ] && echo 1 || echo 0)"

kill $HPID 2>/dev/null
echo ""
echo "== 结果: PASS=$PASS FAIL=$FAIL =="
[ "$FAIL" = "0" ] && echo "VERIFY: ALL PASS" || echo "VERIFY: $FAIL FAILED"
