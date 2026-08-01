#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ya {

enum class CaptureBackend {
    Gdi = 0,
    Dxgi = 1,
    Wgc = 2,
};

struct FramePacket {
    // BGR8 tightly packed; empty if grab failed.
    // Ownership: filled by FrameSource (copy from capture internal buffer).
    std::vector<uint8_t> bgr;
    int width = 0;
    int height = 0;
    int channels = 3;
    int64_t pts_ns = 0;
    int origin_x = 0;
    int origin_y = 0;
    int screen_w = 0;
    int screen_h = 0;
};

class FrameSource {
public:
    FrameSource();
    ~FrameSource();

    FrameSource(const FrameSource&) = delete;
    FrameSource& operator=(const FrameSource&) = delete;

    bool open_center(CaptureBackend backend, int width, int height);
    bool open_region(CaptureBackend backend, int x, int y, int width, int height);
    bool set_window(void* hwnd);
    bool set_region(int x, int y, int width, int height);

    bool grab(FramePacket& out);
    void close();

    bool is_open() const;
    int width() const;
    int height() const;
    CaptureBackend backend() const { return backend_; }
    const std::string& last_error() const { return last_error_; }

    static CaptureBackend parse_backend(const std::string& name);
    static const char* backend_name(CaptureBackend b);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    CaptureBackend backend_ = CaptureBackend::Dxgi;
    std::string last_error_;
    int origin_x_ = 0;
    int origin_y_ = 0;
};

} // namespace ya
