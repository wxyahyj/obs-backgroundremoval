#include "ScreenCapture.h"
#include "ScreenCaptureFactory.h"
#include "IScreenCapture.h"
#include <mutex>
#include <iostream>
#include <string>
#include <unordered_map>


// 存储所有创建的截图器实例
static std::unordered_map<ScreenCaptureHandle, std::unique_ptr<IScreenCapture>> g_captureInstances;

SCREENCAPTURE_API ScreenCaptureHandle Create(CaptureTypeEnum type) {

    InternalCaptureType internalType;
    switch (type) {
    case CAPTURE_GDI:
        internalType = INTERNAL_CAPTURE_GDI;
        break;
    case CAPTURE_DIRECTX:
        internalType = INTERNAL_CAPTURE_DIRECTX;
        break;
    case CAPTURE_WGC:
        internalType = INTERNAL_CAPTURE_WGC;
        break;
    default:
        return nullptr;
    }

    // 使用工厂创建实例
    auto capture = ScreenCaptureFactory::CreateCaptureInstance(internalType);
    if (!capture) {
        return nullptr;
    }

    ScreenCaptureHandle handle = static_cast<ScreenCaptureHandle>(capture.get());
    {
        g_captureInstances[handle] = std::move(capture);
    }
    return handle;
}

SCREENCAPTURE_API int Init(ScreenCaptureHandle handle, int width, int height) {
    auto it = g_captureInstances.find(handle);
    if (it == g_captureInstances.end()) {
        return 0;
    }
    bool result = it->second->Initialize(width, height);
    return result ? 1 : 0;
}

SCREENCAPTURE_API int InitRegion(ScreenCaptureHandle handle, int x, int y, int width, int height) {
    auto it = g_captureInstances.find(handle);
    if (it == g_captureInstances.end()) {
        return 0;
    }
    bool result = it->second->InitializeRegion(x, y, width, height);
    return result ? 1 : 0;
}

SCREENCAPTURE_API int SetWindow(ScreenCaptureHandle handle, HWND hwnd) {
    auto it = g_captureInstances.find(handle);
    if (it == g_captureInstances.end()) {
        return 0;
    }
    bool result = it->second->SetWindow(hwnd);
    return result ? 1 : 0;
}

SCREENCAPTURE_API int SetRegion(ScreenCaptureHandle handle, int x, int y, int width, int height) {
    auto it = g_captureInstances.find(handle);
    if (it == g_captureInstances.end()) {
        return 0;
    }
    bool result = it->second->SetRegion(x, y, width, height);
    return result ? 1 : 0;
}


SCREENCAPTURE_API const unsigned char* CaptureBGR(ScreenCaptureHandle handle) {
    auto it = g_captureInstances.find(handle);
    if (it == g_captureInstances.end()) {
        return nullptr;
    }
    // 正确的写法
    const unsigned char* result = it->second->CaptureBGR();
    return result;
}

SCREENCAPTURE_API const unsigned char* CaptureBMP(ScreenCaptureHandle handle) {
    auto it = g_captureInstances.find(handle);
    if (it == g_captureInstances.end()) {
        return nullptr;
    }
    // 正确的写法
    const unsigned char* result = it->second->CaptureBMP();
    return result;
}

SCREENCAPTURE_API void Release(ScreenCaptureHandle handle) {
    auto it = g_captureInstances.find(handle);
    if (it != g_captureInstances.end()) {
        it->second->Release();
        g_captureInstances.erase(it);
    }
}
