#include "FrameSource.hpp"

#include "yolo_aim/capture_c_api.h"

#include <chrono>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ya {

struct FrameSource::Impl {
    YCapHandle handle = nullptr;
};

FrameSource::FrameSource() : impl_(std::make_unique<Impl>()) {}

FrameSource::~FrameSource() { close(); }

CaptureBackend FrameSource::parse_backend(const std::string& name) {
    if (name == "gdi" || name == "GDI") return CaptureBackend::Gdi;
    if (name == "wgc" || name == "WGC") return CaptureBackend::Wgc;
    return CaptureBackend::Dxgi;
}

const char* FrameSource::backend_name(CaptureBackend b) {
    switch (b) {
    case CaptureBackend::Gdi: return "gdi";
    case CaptureBackend::Wgc: return "wgc";
    default: return "dxgi";
    }
}

static YCapBackend to_ycap(CaptureBackend b) {
    switch (b) {
    case CaptureBackend::Gdi: return YCAP_GDI;
    case CaptureBackend::Wgc: return YCAP_WGC;
    default: return YCAP_DXGI;
    }
}

bool FrameSource::open_center(CaptureBackend backend, int width, int height) {
    close();
    backend_ = backend;
    impl_->handle = ycap_create(to_ycap(backend));
    if (!impl_->handle) {
        last_error_ = "ycap_create failed";
        return false;
    }
    if (!ycap_init_center(impl_->handle, width, height)) {
        last_error_ = "ycap_init_center failed";
        ycap_release(impl_->handle);
        impl_->handle = nullptr;
        return false;
    }
#ifdef _WIN32
    const int sw = GetSystemMetrics(SM_CXSCREEN);
    const int sh = GetSystemMetrics(SM_CYSCREEN);
    origin_x_ = (sw - width) / 2;
    origin_y_ = (sh - height) / 2;
#else
    origin_x_ = origin_y_ = 0;
#endif
    last_error_.clear();
    return true;
}

bool FrameSource::open_region(CaptureBackend backend, int x, int y, int width, int height) {
    close();
    backend_ = backend;
    impl_->handle = ycap_create(to_ycap(backend));
    if (!impl_->handle) {
        last_error_ = "ycap_create failed";
        return false;
    }
    if (!ycap_init_region(impl_->handle, x, y, width, height)) {
        last_error_ = "ycap_init_region failed";
        ycap_release(impl_->handle);
        impl_->handle = nullptr;
        return false;
    }
    origin_x_ = x;
    origin_y_ = y;
    last_error_.clear();
    return true;
}

bool FrameSource::set_window(void* hwnd) {
    if (!impl_->handle) return false;
    return ycap_set_window(impl_->handle, hwnd) != 0;
}

bool FrameSource::set_region(int x, int y, int width, int height) {
    if (!impl_->handle) return false;
    origin_x_ = x;
    origin_y_ = y;
    return ycap_set_region(impl_->handle, x, y, width, height) != 0;
}

bool FrameSource::grab(FramePacket& out) {
    if (!impl_->handle) {
        last_error_ = "not open";
        return false;
    }
    const unsigned char* ptr = ycap_capture_bgr(impl_->handle);
    if (!ptr) {
        last_error_ = "ycap_capture_bgr returned null";
        return false;
    }
    const int w = ycap_width(impl_->handle);
    const int h = ycap_height(impl_->handle);
    if (w <= 0 || h <= 0) {
        last_error_ = "invalid size";
        return false;
    }
    const size_t nbytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
    out.bgr.resize(nbytes);
    std::memcpy(out.bgr.data(), ptr, nbytes);
    out.width = w;
    out.height = h;
    out.channels = 3;
    out.origin_x = origin_x_;
    out.origin_y = origin_y_;
#ifdef _WIN32
    out.screen_w = GetSystemMetrics(SM_CXSCREEN);
    out.screen_h = GetSystemMetrics(SM_CYSCREEN);
#endif
    out.pts_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count();
    last_error_.clear();
    return true;
}

void FrameSource::close() {
    if (impl_ && impl_->handle) {
        ycap_release(impl_->handle);
        impl_->handle = nullptr;
    }
}

bool FrameSource::is_open() const { return impl_ && impl_->handle != nullptr; }

int FrameSource::width() const {
    return (impl_ && impl_->handle) ? ycap_width(impl_->handle) : 0;
}

int FrameSource::height() const {
    return (impl_ && impl_->handle) ? ycap_height(impl_->handle) : 0;
}

} // namespace ya
