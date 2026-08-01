#include "yolo_aim/capture_c_api.h"

#include "IScreenCapture.h"
#include "ScreenCaptureFactory.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
struct Entry {
    std::unique_ptr<IScreenCapture> cap;
};
std::mutex g_mu;
std::unordered_map<YCapHandle, Entry> g_map;

InternalCaptureType to_internal(YCapBackend b) {
    switch (b) {
    case YCAP_GDI: return INTERNAL_CAPTURE_GDI;
    case YCAP_WGC: return INTERNAL_CAPTURE_WGC;
    default: return INTERNAL_CAPTURE_DIRECTX;
    }
}
} // namespace

YCapHandle ycap_create(YCapBackend backend) {
    auto cap = ScreenCaptureFactory::CreateCaptureInstance(to_internal(backend));
    if (!cap) return nullptr;
    YCapHandle h = reinterpret_cast<YCapHandle>(cap.get());
    std::lock_guard<std::mutex> lock(g_mu);
    g_map[h].cap = std::move(cap);
    return h;
}

int ycap_init_center(YCapHandle h, int width, int height) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_map.find(h);
    if (it == g_map.end()) return 0;
    return it->second.cap->Initialize(width, height) ? 1 : 0;
}

int ycap_init_region(YCapHandle h, int x, int y, int width, int height) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_map.find(h);
    if (it == g_map.end()) return 0;
    return it->second.cap->InitializeRegion(x, y, width, height) ? 1 : 0;
}

int ycap_set_window(YCapHandle h, void* hwnd) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_map.find(h);
    if (it == g_map.end()) return 0;
#ifdef _WIN32
    return it->second.cap->SetWindow(static_cast<HWND>(hwnd)) ? 1 : 0;
#else
    (void)hwnd;
    return 0;
#endif
}

int ycap_set_region(YCapHandle h, int x, int y, int width, int height) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_map.find(h);
    if (it == g_map.end()) return 0;
    return it->second.cap->SetRegion(x, y, width, height) ? 1 : 0;
}

const unsigned char* ycap_capture_bgr(YCapHandle h) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_map.find(h);
    if (it == g_map.end()) return nullptr;
    return it->second.cap->CaptureBGR();
}

int ycap_width(YCapHandle h) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_map.find(h);
    if (it == g_map.end()) return 0;
    return it->second.cap->GetWidth();
}

int ycap_height(YCapHandle h) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_map.find(h);
    if (it == g_map.end()) return 0;
    return it->second.cap->GetHeight();
}

void ycap_release(YCapHandle h) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_map.find(h);
    if (it == g_map.end()) return;
    it->second.cap->Release();
    g_map.erase(it);
}
