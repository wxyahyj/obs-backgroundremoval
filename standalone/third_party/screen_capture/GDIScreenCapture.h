#ifndef GDI_SCREEN_CAPTURE_H
#define GDI_SCREEN_CAPTURE_H

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <memory>
#include <vector>
#include <cstdint>
#include <atomic>
#include <unordered_map>
#include "IScreenCapture.h"
#include <Windows.h>
#include <immintrin.h>

struct GDIResource {
    HDC screenDC = nullptr;
    HDC windowDC = nullptr;
    HDC memoryDC = nullptr;
    HBITMAP bitmap = nullptr;
    HWND targetWindow = nullptr;

    GDIResource() = default;
    ~GDIResource();

    bool Initialize(HWND target, int width, int height);
    void Release();

    GDIResource(const GDIResource&) = delete;
    GDIResource& operator=(const GDIResource&) = delete;
};

class GDIScreenCapture : public IScreenCapture {
public:
    GDIScreenCapture();
    ~GDIScreenCapture() override;

    bool Initialize(int width, int height) override;
    bool InitializeRegion(int x, int y, int width, int height) override;
    bool SetWindow(HWND hwnd) override;
    bool SetRegion(int x, int y, int width, int height) override;
    void Release() override;

    const unsigned char* CaptureBGR() override;
    const unsigned char* CaptureBMP() override;

    int GetWidth() const override { return captureWidth; }
    int GetHeight() const override { return captureHeight; }

private:
    bool IsSSE42Supported();
    inline bool ValidateBMPParameters(const BYTE* srcData, int x, int y, int width, int height, int srcPitch);
    inline bool IsResolutionFirstCall(int width, int height);

    bool PerformCapture();
    bool ExtractBGRData();
    bool ExtractBMPData();
    bool UpdateBitmapIfNeeded(int width, int height);

    void CheckAndUpdateScreenResolution();  // 新增：检查并更新屏幕分辨率
    void RecalculateCenterPosition();       // 新增：重新计算居中位置

    void ConvertDIBtoBGR_SIMD(const uint8_t* src, uint8_t* dst, int width, int height, int srcStride);

    GDIResource gdiResources;
    std::unique_ptr<PreallocatedMemoryPool> memoryPool;

    BITMAPINFOHEADER bitmapInfo;

    int captureX{ 0 };
    int captureY{ 0 };
    int captureWidth{ 0 };
    int captureHeight{ 0 };
    int captureTimeoutMs{ 0 };
    bool useCustomRegion{ false };
    bool useCenterMode{ false };

    int lastScreenWidth{ 0 };   // 新增：记录上次的屏幕宽度
    int lastScreenHeight{ 0 };  // 新增：记录上次的屏幕高度
    int originalWidth{ 0 };     // 新增：记录原始请求的宽度
    int originalHeight{ 0 };    // 新增：记录原始请求的高度

    std::vector<uint8_t> bgrBuffer;
    std::vector<uint8_t> bmpBuffer;
    std::vector<uint8_t> dibBuffer;

    std::atomic<bool> initialized{ false };
    bool sse42Supported;

    std::unordered_map<std::pair<int, int>, BMPHeaderCache, PairHash> bmpHeaderCache;

    std::vector<ResolutionFirstCall> resolutionFirstCalls;
};

#endif