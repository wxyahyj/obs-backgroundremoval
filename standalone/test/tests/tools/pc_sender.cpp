// pc_sender.cpp — 完整 PC→手机 远程推理发送工具
// 编译: cl /EHsc /std:c++17 pc_sender.cpp /link gdiplus.lib ws2_32.lib
// 使用: pc_sender.exe 手机IP 端口 [宽] [高] [质量]
// 示例: pc_sender.exe 192.168.1.100 9999 320 320 60
// 按 Enter 截一帧发送，q 退出

#define NOMINMAX
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus")
#pragma comment(lib, "ws2_32")

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

// ===== GDI+ JPEG 编码 =====
using namespace Gdiplus;
struct GdiInit { GdiInit() { GdiplusStartupInput i; GdiplusStartup(&t, &i, nullptr); } ~GdiInit() { GdiplusShutdown(t); } ULONG_PTR t; };
static GdiInit gdi;

bool bgr_to_jpeg(const uint8_t *bgr, int w, int h, int quality, std::vector<uint8_t> *out) {
    CLSID clsid; UINT n = 0, s = 0;
    GetImageEncoders(&n, &s);
    if (!s) return false;
    std::vector<ImageCodecInfo> ci(s / sizeof(ImageCodecInfo));
    GetImageEncoders(&n, &s, ci.data());
    bool found = false;
    for (auto &c : ci) {
        if (wcscmp(c.MimeType, L"image/jpeg") == 0) { clsid = c.Clsid; found = true; break; }
    }
    if (!found) return false;
    Bitmap bmp(w, h, w * 3, PixelFormat24bppRGB, const_cast<BYTE*>(bgr));
    IStream *st = nullptr;
    CreateStreamOnHGlobal(nullptr, TRUE, &st);
    EncoderParameters ep{}; ep.Count = 1; ep.Parameter[0].Guid = EncoderQuality;
    ep.Parameter[0].Type = EncoderParameterValueTypeLong; ep.Parameter[0].NumberOfValues = 1;
    ULONG qv = static_cast<ULONG>(quality); ep.Parameter[0].Value = &qv;
    if (bmp.Save(st, &clsid, &ep) != Ok) { st->Release(); return false; }
    STATSTG stat; st->Stat(&stat, STATFLAG_NONAME);
    ULONG len = static_cast<ULONG>(stat.cbSize.QuadPart);
    out->resize(len);
    LARGE_INTEGER z{}; st->Seek(z, STREAM_SEEK_SET, nullptr);
    st->Read(out->data(), len, nullptr); st->Release();
    return true;
}

// ===== GDI 截图 =====
bool gdi_capture(int x, int y, int w, int h, std::vector<uint8_t> *bgr) {
    HDC dc = GetDC(nullptr);
    HDC mem = CreateCompatibleDC(dc);
    HBITMAP hbmp = CreateCompatibleBitmap(dc, w, h);
    SelectObject(mem, hbmp);
    BitBlt(mem, 0, 0, w, h, dc, x, y, SRCCOPY | CAPTUREBLT);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 24;
    bmi.bmiHeader.biCompression = BI_RGB;

    bgr->resize(static_cast<size_t>(w) * h * 3);
    GetDIBits(mem, hbmp, 0, h, bgr->data(), &bmi, DIB_RGB_COLORS);

    DeleteObject(hbmp); DeleteDC(mem); ReleaseDC(nullptr, dc);
    return true;
}

// ===== TCP 通信 =====
bool tcp_send_jpeg(SOCKET fd, const uint8_t *jpeg, int len) {
    uint8_t hdr[4];
    hdr[0] = static_cast<uint8_t>((len >> 24) & 0xFF);
    hdr[1] = static_cast<uint8_t>((len >> 16) & 0xFF);
    hdr[2] = static_cast<uint8_t>((len >> 8) & 0xFF);
    hdr[3] = static_cast<uint8_t>(len & 0xFF);
    if (send(fd, (const char*)hdr, 4, 0) != 4) return false;
    int sent = 0;
    while (sent < len) {
        int n = send(fd, (const char*)jpeg + sent, len - sent, 0);
        if (n <= 0) return false;
        sent += n;
    }
    return true;
}

// 收响应（跳过心跳）
bool tcp_recv_response(SOCKET fd, std::string *out_json, int timeout_ms = 15000) {
    out_json->clear();
    #ifdef _WIN32
    DWORD to = timeout_ms;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&to, sizeof(to));
    #endif
    while (true) {
        uint8_t hdr[4];
        int n = recv(fd, (char*)hdr, 4, MSG_WAITALL);
        if (n != 4) {
            if (n == SOCKET_ERROR) { char buf[256]; snprintf(buf, sizeof(buf), "recv error: %d", WSAGetLastError()); *out_json = buf; }
            else *out_json = "disconnected";
            return false;
        }
        uint32_t len = (static_cast<uint32_t>(hdr[0]) << 24) |
                       (static_cast<uint32_t>(hdr[1]) << 16) |
                       (static_cast<uint32_t>(hdr[2]) << 8) |
                       static_cast<uint32_t>(hdr[3]);
        if (len == 0xFFFFFFFF || len == static_cast<uint32_t>(-1)) {
            // 心跳，跳过剩余 9 字节
            recv(fd, (char*)hdr, 9, MSG_WAITALL);
            continue;
        }
        if (len > 1024 * 1024) { *out_json = "too large"; return false; }
        std::vector<char> buf(len);
        int got = 0;
        while (got < (int)len) {
            int r = recv(fd, buf.data() + got, len - got, MSG_WAITALL);
            if (r <= 0) { *out_json = "recv body fail"; return false; }
            got += r;
        }
        out_json->assign(buf.data(), len);
        return true;
    }
}

// ===== 主函数 =====
int main(int argc, char **argv) {
    if (argc < 3) {
        printf("用法: pc_sender.exe 手机IP 端口 [W] [H] [JPEG质量]\n");
        printf("示例: pc_sender.exe 192.168.1.100 9999 320 320 60\n");
        printf("按 Enter 截一帧; q+Enter 退出\n");
        return 1;
    }

    std::string host = argv[1];
    int port = atoi(argv[2]);
    int cap_w = argc > 3 ? atoi(argv[3]) : 320;
    int cap_h = argc > 4 ? atoi(argv[4]) : 320;
    int quality = argc > 5 ? atoi(argv[5]) : 60;

    WSADATA wsa{};
    WSAStartup(MAKEWORD(2, 2), &wsa);

    while (true) {
        printf("\n连接 %s:%d ... ", host.c_str(), port);
        SOCKET fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<u_short>(port));
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
        if (connect(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
            printf("连接失败 (重试中)\n");
            closesocket(fd);
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        int opt = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&opt, sizeof(opt));
        printf("已连接!\n");

        while (true) {
            printf("\n按 Enter 截图发送 (q+Enter 退出): ");
            char cmd[256];
            if (!fgets(cmd, sizeof(cmd), stdin)) break;
            if (cmd[0] == 'q' || cmd[0] == 'Q') goto done;

            // 1. 截图
            std::vector<uint8_t> bgr;
            auto t0 = std::chrono::steady_clock::now();
            gdi_capture(0, 0, cap_w, cap_h, &bgr);
            auto t1 = std::chrono::steady_clock::now();
            double capture_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // 2. JPEG 编码
            std::vector<uint8_t> jpeg;
            if (!bgr_to_jpeg(bgr.data(), cap_w, cap_h, quality, &jpeg)) {
                printf("JPEG 编码失败\n"); continue;
            }
            auto t2 = std::chrono::steady_clock::now();
            double jpeg_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            // 3. TCP 发送
            auto t3 = std::chrono::steady_clock::now();
            s

            if (!tcp_send_jpeg(fd, jpeg.data(), (int)jpeg.size())) {
                printf("发送失败，准备重连…\n"); closesocket(fd); break;
            }
            auto t4 = std::chrono::steady_clock::now();
            double send_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

            // 4. 收响应
            std::string resp;
            if (!tcp_recv_response(fd, &resp)) {
                printf("收响应失败: %s\n", resp.c_str());
                closesocket(fd); break;
            }
            auto t5 = std::chrono::steady_clock::now();
            double rtt_ms = std::chrono::duration<double, std::milli>(t5 - t0).count();

            // 5. 解析 JSON
            printf("截图:%dms JPEG:%dms 发送:%dms RTT:%.0fms\n",
                   (int)capture_ms, (int)jpeg_ms, (int)send_ms, rtt_ms);
            try {
                auto j = nlohmann_json_parse(resp); // simplified
                printf("检测结果: %s\n", resp.c_str());
            } catch (...) {
                printf("原始回复: %s\n", resp.c_str());
            }
        }
        closesocket(fd);
    }

done:
    WSACleanup();
    return 0;
}
