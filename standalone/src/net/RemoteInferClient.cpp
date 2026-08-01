#include "RemoteInferClient.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32")
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET (-1)
#define closesocket(x) ::close(x)
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

// Simple JPEG encoder via stb_image_write if available, else fallback
// For production use turbojpeg. Here we use a minimal inline.
// Real impl should link libjpeg-turbo.

namespace ya {
namespace {

using clock = std::chrono::steady_clock;

// Minimal inline JPEG encoder (BGR → JPEG bytes)
// Quality 1-100. Uses simple baseline JPEG.
// For production: replace with turbojpeg / stb_image_write
bool bgr_to_jpeg(const uint8_t *bgr, int w, int h, int quality,
                 std::vector<uint8_t> *out) {
    // Stub: just copy raw BGR as JPEG placeholder
    // Real implementation: include stb_image_write.h or use turbojpeg
    // For now return false to fallback to raw BGR send
    (void)bgr; (void)w; (void)h; (void)quality;
    // Remove this once you have a real JPEG encoder linked
    return false;
}

// TCP helper: read exactly n bytes
bool recv_all(SOCKET fd, uint8_t *buf, int n) {
    while (n > 0) {
        int r = ::recv(fd, reinterpret_cast<char*>(buf), n, 0);
        if (r <= 0) return false;
        buf += r;
        n -= r;
    }
    return true;
}

// TCP helper: send all bytes
bool send_all(SOCKET fd, const uint8_t *data, int n) {
    while (n > 0) {
        int s = ::send(fd, reinterpret_cast<const char*>(data), n, 0);
        if (s <= 0) return false;
        data += s;
        n -= s;
    }
    return true;
}

} // namespace

struct RemoteInferClient::Impl {
    SOCKET fd_ = INVALID_SOCKET;
    std::string host_;
    int port_ = 9999;

    // Receive buffer reuse
    std::vector<uint8_t> recv_buf_;
    std::vector<uint8_t> jpeg_buf_;
};

RemoteInferClient::RemoteInferClient()
    : impl_(std::make_unique<Impl>()) {}

RemoteInferClient::~RemoteInferClient() { disconnect(); }

bool RemoteInferClient::connect(const std::string &host, int port) {
    disconnect();
    impl_->host_ = host;
    impl_->port_ = port;

#ifdef _WIN32
    WSADATA wsa{};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
#endif

    impl_->fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (impl_->fd_ == INVALID_SOCKET) return false;

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        // DNS resolve
        struct hostent *he = gethostbyname(host.c_str());
        if (!he) { disconnect(); return false; }
        memcpy(&addr.sin_addr, he->h_addr, he->h_length);
    }

    // Set 5s connect timeout
#ifdef _WIN32
    unsigned long mode = 1;
    ioctlsocket(impl_->fd_, FIONBIO, &mode);
    ::connect(impl_->fd_, (sockaddr*)&addr, sizeof(addr));
    fd_set fd_set_w;
    FD_ZERO(&fd_set_w);
    FD_SET(impl_->fd_, &fd_set_w);
    struct timeval tv{5, 0};
    int sel = select(0, nullptr, &fd_set_w, nullptr, &tv);
    mode = 0;
    ioctlsocket(impl_->fd_, FIONBIO, &mode);
    if (sel <= 0) { disconnect(); return false; }
#else
    if (::connect(impl_->fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        disconnect(); return false;
    }
#endif
    // Enable TCP_NODELAY for low latency
    int opt = 1;
    setsockopt(impl_->fd_, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char*>(&opt), sizeof(opt));

    std::cout << "[remote] connected to " << host << ":" << port << std::endl;
    return true;
}

void RemoteInferClient::disconnect() {
    if (impl_->fd_ != INVALID_SOCKET) {
        closesocket(impl_->fd_);
        impl_->fd_ = INVALID_SOCKET;
    }
}

bool RemoteInferClient::is_connected() const {
    return impl_->fd_ != INVALID_SOCKET;
}

RemoteResult RemoteInferClient::infer(const uint8_t *bgr, int w, int h,
                                       int quality, int timeout_ms) {
    RemoteResult result;
    if (!is_connected()) {
        result.error = "not connected";
        return result;
    }

    const auto t0 = clock::now();

    // 1. Encode JPEG (or send raw BGR as fallback)
    std::vector<uint8_t> payload;
    bool sent = false;

    if (bgr_to_jpeg(bgr, w, h, quality, &impl_->jpeg_buf_)) {
        // JPEG header
        int32_t len = static_cast<int32_t>(impl_->jpeg_buf_.size());
        uint8_t hdr[4];
        hdr[0] = static_cast<uint8_t>((len >> 24) & 0xFF);
        hdr[1] = static_cast<uint8_t>((len >> 16) & 0xFF);
        hdr[2] = static_cast<uint8_t>((len >> 8) & 0xFF);
        hdr[3] = static_cast<uint8_t>(len & 0xFF);
        if (send_all(impl_->fd_, hdr, 4) && send_all(impl_->fd_, impl_->jpeg_buf_.data(), len))
            sent = true;
    }

    if (!sent) {
        // Fallback: send raw BGR as RGBA (preconvert on phone = CPU cost)
        // Better: convert BGR → JPEG with mini lib
        result.error = "JPEG encoder not available (link libjpeg-turbo)";
        return result;
    }

    // 2. Set receive timeout
#ifdef _WIN32
    DWORD rcvto = static_cast<DWORD>(timeout_ms);
    setsockopt(impl_->fd_, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&rcvto), sizeof(rcvto));
#endif

    // 3. Read response header (4 bytes json length)
    uint8_t hdr[4];
    if (!recv_all(impl_->fd_, hdr, 4)) {
        result.error = "recv header failed";
        disconnect();
        return result;
    }
    int32_t json_len = (static_cast<int32_t>(hdr[0]) << 24) |
                       (static_cast<int32_t>(hdr[1]) << 16) |
                       (static_cast<int32_t>(hdr[2]) << 8) |
                       static_cast<int32_t>(hdr[3]);
    if (json_len <= 0 || json_len > 1024 * 1024) {
        result.error = "invalid json length";
        return result;
    }

    impl_->recv_buf_.resize(static_cast<size_t>(json_len));
    if (!recv_all(impl_->fd_, impl_->recv_buf_.data(), json_len)) {
        result.error = "recv json failed";
        return result;
    }

    const auto t1 = clock::now();
    rtt_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 4. Parse JSON (minimal inline, no dependency)
    std::string json(reinterpret_cast<char*>(impl_->recv_buf_.data()),
                     static_cast<size_t>(json_len));

    // Use nlohmann json if available, else simple key search
    // Here depend on ConfigStore already having nlohmann
    try {
        auto j = nlohmann::json::parse(json);
        if (j.contains("err") && !j["err"].is_null() && j["err"].is_string()) {
            result.error = j["err"].get<std::string>();
        }
        result.infer_ms = j.value("ms", 0.f);
        result.seq = j.value("seq", 0);
        if (j.contains("dets") && j["dets"].is_array()) {
            for (auto &item : j["dets"]) {
                RemoteDet d;
                d.class_id = item.value("c", 0);
                d.confidence = item.value("s", 0.f);
                d.x = item.value("x", 0.f);
                d.y = item.value("y", 0.f);
                d.w = item.value("w", 0.f);
                d.h = item.value("h", 0.f);
                result.dets.push_back(d);
            }
        }
    } catch (const std::exception &e) {
        result.error = std::string("json parse: ") + e.what();
    }

    return result;
}

} // namespace ya
