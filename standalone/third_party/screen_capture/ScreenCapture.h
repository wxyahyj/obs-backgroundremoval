#ifndef SCREEN_CAPTURE_H
#define SCREEN_CAPTURE_H

#include <stdint.h>
#include <Windows.h>

#ifdef SCREENCAPTURE_EXPORTS
#define SCREENCAPTURE_API __declspec(dllexport)
#else
#define SCREENCAPTURE_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

    typedef void* ScreenCaptureHandle;

    typedef enum {
        CAPTURE_GDI = 0,
        CAPTURE_DIRECTX = 1,
        CAPTURE_WGC = 2
    } CaptureTypeEnum;

    SCREENCAPTURE_API ScreenCaptureHandle Create(CaptureTypeEnum type);

    SCREENCAPTURE_API int Init(ScreenCaptureHandle handle, int width, int height);

    SCREENCAPTURE_API int InitRegion(ScreenCaptureHandle handle, int x, int y, int width, int height);

    SCREENCAPTURE_API int SetWindow(ScreenCaptureHandle handle, HWND hwnd);

    SCREENCAPTURE_API int SetRegion(ScreenCaptureHandle handle, int x, int y, int width, int height);

    SCREENCAPTURE_API const unsigned char* CaptureBGR(ScreenCaptureHandle handle);

    SCREENCAPTURE_API const unsigned char* CaptureBMP(ScreenCaptureHandle handle);

    SCREENCAPTURE_API void Release(ScreenCaptureHandle handle);


#ifdef __cplusplus
}
#endif

#endif // SCREEN_CAPTURE_H