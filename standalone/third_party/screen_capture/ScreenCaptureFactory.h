#ifndef SCREEN_CAPTURE_FACTORY_H
#define SCREEN_CAPTURE_FACTORY_H

#include "IScreenCapture.h"
#include <memory>

// 内部使用的捕获类型枚举
enum InternalCaptureType {
    INTERNAL_CAPTURE_GDI = 0,
    INTERNAL_CAPTURE_DIRECTX = 1,
    INTERNAL_CAPTURE_WGC = 2
};

// 工厂类
class ScreenCaptureFactory {
public:
    static std::unique_ptr<IScreenCapture> CreateCaptureInstance(InternalCaptureType type);
};

#endif // SCREEN_CAPTURE_FACTORY_H