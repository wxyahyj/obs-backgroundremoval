#include "RemoteSender.hpp"

// Windows JPEG encoder via GDI+ (不依赖外部库)
#define NOMINMAX
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus")
#pragma comment(lib, "ws2_32")

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace ya {
namespace {

using namespace Gdiplus;

struct GdiplusInit {
    GdiplusInit() {
        GdiplusStartupInput inp;
        GdiplusStartup(&token, &inp, nullptr);
    }
    ~GdiplusInit() { GdiplusShutdown(token); }
    ULONG_PTR token = 0;
};
static GdiplusInit g_gdi_init;

// 用 GDI+ 把 BGR buffer → JPEG bytes
bool gdi_bgr_to_jpeg(const uint8_t *bgr, int w, int h, int quality,
                     std::vector<uint8_t> *out) {
    CLSID clsid;
    // 获取 JPEG encoder CLSID
    UINT num = 0, size = 0;
    GetImageEncoders(&num, &size);
    if (size == 0) return false;
    std::vector<ImageCodecInfo> codecs(size / sizeof(ImageCodecInfo));
    GetImageEncoders(&num, &size, codecs.data());
    bool found = false;
    for (auto &c : codecs) {
        if (wcscmp(c.MimeType, L"image/jpeg") == 0) {
            clsid = c.Clsid;
            found = true;
            break;
        }
    }
    if (!found) return false;

    // Create Bitmap from BGR data
    Bitmap bmp(w, h, w * 3, PixelFormat24bppRGB, const_cast<BYTE*>(bgr));
    if (bmp.GetLastStatus() != Ok) return false;

    // Save to IStream
    IStream *stream = nullptr;
    CreateStreamOnHGlobal(nullptr, TRUE, &stream);

    // Quality parameter
    EncoderParameters enc{};
    enc.Count = 1;
    enc.Parameter[0].Guid = EncoderQuality;
    enc.Parameter[0].Type = EncoderParameterValueTypeLong;
    enc.Parameter[0].NumberOfValues = 1;
    ULONG q = static_cast<ULONG>(quality);
    enc.Parameter[0].Value = &q;

    Status st = bmp.Save(stream, &clsid, &enc);
    if (st != Ok) { stream->Release(); return false; }

    // Read back bytes
    STATSTG stat;
    stream->Stat(&stat, STATFLAG_NONAME);
    ULONG len = static_cast<ULONG>(stat.cbSize.QuadPart);
    out->resize(len);
    LARGE_INTEGER zero{};
    stream->Seek(zero, STREAM_SEEK_SET, nullptr);
    stream->Read(out->data(), len, nullptr);
    stream->Release();
    return true;
}

} // namespace

struct RemoteSender::Impl {
    SOCKET fd_ = INVALID_SOCKET;

    // DXGI capture
    bool dxgi_ok_ = false;
    // Fallback: GDI capture
    // ...

    bool connect_tcp(const std::string &host, int port) {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ == INVALID_SOCKET) return false;

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<u_short>(port));
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
            struct hostent *he = gethostbyname(host.c_str());
            if (!he) return false;
            memcpy(&addr.sin_addr, he->h_addr, he->h_length);
        }

        if (::connect(fd_, (sockaddr*)&addr, sizeof(addr)) < 0)
            return false;

        int opt = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, (const char*)&opt, sizeof(opt));
        return true;
    }

    bool send_jpeg(const uint8_t *jpeg, int len) {
        // 4字节长度前缀 + JPEG
        uint8_t hdr[4];
        hdr[0] = static_cast<uint8_t>((len >> 24) & 0xFF);
        hdr[1] = static_cast<uint8_t>((len >> 16) & 0xFF);
        hdr[2] = static_cast<uint8_t>((len >> 8) & 0xFF);
        hdr[3] = static_cast<uint8_t>(len & 0xFF);

        if (send(fd_, (const char*)hdr, 4, 0) != 4) return false;
        int sent = 0;
        while (sent < len) {
            int n = send(fd_, (const char*)jpeg + sent, len - sent, 0);
            if (n <= 0) return false;
            sent += n;
        }
        return true;
    }

    // 收 4字节长度 + JSON
    bool recv_json(std::vector<uint8_t> *buf) {
        uint8_t hdr[4];
        int n = recv(fd_, (char*)hdr, 4, MSG_WAITALL);
        if (n != 4) return false;
        int len = (hdr[0] << 24) | (hdr[1] << 16) | (hdr[2] << 8) | hdr[3];
        if (len <= 0 || len > 1024 * 1024) return false;
        buf->resize(len);
        n = recv(fd_, (char*)buf->data(), len, MSG_WAITALL);
        return n == len;
    }
};

RemoteSender::RemoteSender() : impl_(std::make_unique<Impl>()) {}
RemoteSender::~RemoteSender() { disconnect(); }

bool RemoteSender::connect(const std::string &host, int port) {
    disconnect();

    WSADATA wsa{};
    WSAStartup(MAKEWORD(2, 2), &wsa);
    return impl_->connect_tcp(host, port);
}

void RemoteSender::disconnect() {
    if (impl_->fd_ != INVALID_SOCKET) {
        closesocket(impl_->fd_);
        impl_->fd_ = INVALID_SOCKET;
    }
}

bool RemoteSender::is_connected() const {
    return impl_->fd_ != INVALID_SOCKET;
}

RemoteSender::Result RemoteSender::capture_and_send(
    int region_w, int region_h, int x, int y, int quality)
{
    Result r;
    if (!is_connected()) { r.error = "not connected"; return r; }

    // 1. DXGI capture (简化：GDI fallback)
    // TODO: 接入你现有的 DXScreenCapture
    // 这里先做一个假的测试用
    int w = 320, h = 320;

    // 2. 读帧（用你的 FrameSource）
    // 用现有的 FrameSource::grab() 或直接传一个测试图案
    // 此处仅测试连接，不实际截图
    // 用测试图案
    std::vector<uint8_t> test_bgr(static_cast<size_t>(w) * h * 3, 128);
    for (int i = 0; i < w * h; ++i) {
        // 红蓝渐变（测试用）
        test_bgr[i * 3 + 0] = static_cast<uint8_t>((i % w) * 255 / w);
        test_bgr[i * 3 + 1] = static_cast<uint8_t>(128);
        test_bgr[i * 3 + 2] = static_cast<uint8_t>((i / w) * 255 / h);
    }

    // 3. JPEG 编码
    std::vector<uint8_t> jpeg;
    if (!gdi_bgr_to_jpeg(test_bgr.data(), w, h, quality, &jpeg)) {
        r.error = "JPEG encode failed";
        return r;
    }

    // 4. TCP 发送
    if (!impl_->send_jpeg(jpeg.data(), static_cast<int>(jpeg.size()))) {
        r.error = "send failed";
        return r;
    }

    // 5. 收 JSON 响应
    std::vector<uint8_t> recv_buf;
    if (!impl_->recv_json(&recv_buf)) {
        r.error = "recv failed";
        return r;
    }

    r.ok = true;
    // JSON 解析交给调用者
    return r;
}
