# test_remote.py — PC→手机 远程推理测试 (v2)
import socket, sys, json, io
from PIL import Image, ImageDraw

host = sys.argv[1] if len(sys.argv) > 1 else "192.168.31.171"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

img = Image.new('RGB', (320, 320), (20, 30, 50))
draw = ImageDraw.Draw(img)
for i in range(320):
    draw.line((i, 0, i, 30), fill=(int(i*255/320), 128, 255-int(i*255/320)))
draw.text((60, 140), "PC -> 手机 测试", fill=(255, 255, 0))
draw.text((60, 170), f"{host}:{port}", fill=(100, 200, 255))

buf = io.BytesIO()
img.save(buf, 'JPEG', quality=60)
jpeg = buf.getvalue()
print(f"图片 {len(jpeg)} 字节, 连接 {host}:{port}...")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(20)
try:
    s.connect((host, port))
    print("已连接!")
    s.sendall((1).to_bytes(4, 'big'))
    s.sendall(len(jpeg).to_bytes(4, 'big') + jpeg)
    print("已发送, 等待回复...")

    while True:
        hdr = s.recv(4)
        if not hdr: break
        l = int.from_bytes(hdr, 'big')
        if l == 0xFFFFFFFF or l == -1:
            s.recv(9); continue
        # 收 JSON
        resp = b''
        while len(resp) < l:
            c = s.recv(min(4096, l - len(resp)))
            if not c: break
            resp += c
        if resp:
            # 清理无效 UTF-8 字节（NaN 等）
            cleaned = resp.decode('utf-8', errors='replace')
            try:
                d = json.loads(cleaned)
                dets = d.get("dets", [])
                print(f"检测到 {len(dets)} 个目标, 耗时 {d.get('ms',0):.1f}ms")
                for det in dets[:8]:
                    print(f"  类别{d['c']} 置信度{d['s']:.2f} 位置({d['x']:.3f},{d['y']:.3f} {d['w']:.3f}x{d['h']:.3f})")
                if d.get("err"): print(f"错误: {d['err']}")
            except:
                print(f"原始回复: {resp[:200]}")
        break
except Exception as e:
    print(f"失败: {e}")
finally:
    s.close()
