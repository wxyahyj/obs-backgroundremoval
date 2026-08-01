#include "ScreenCaptureFactory.h"
#include "GDIScreenCapture.h"
#include "DXScreenCapture.h"
#include "WGCScreenCapture.h"

std::unique_ptr<IScreenCapture> ScreenCaptureFactory::CreateCaptureInstance(InternalCaptureType type) {
    switch (type) {
    case INTERNAL_CAPTURE_GDI:
        return std::unique_ptr<IScreenCapture>(new GDIScreenCapture());
    case INTERNAL_CAPTURE_DIRECTX:
        return std::unique_ptr<IScreenCapture>(new DXScreenCapture());
    case INTERNAL_CAPTURE_WGC:
        return std::unique_ptr<IScreenCapture>(new WGCScreenCapture());
    default:
        return nullptr;
    }
}