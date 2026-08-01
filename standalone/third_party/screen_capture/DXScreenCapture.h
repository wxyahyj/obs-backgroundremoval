#ifndef DXSCREEN_CAPTURE_ADAPTER_H
#define DXSCREEN_CAPTURE_ADAPTER_H

#include "IScreenCapture.h"
#include <immintrin.h>
#include <atomic>

class DXScreenCaptureImpl {
public:
    DXScreenCaptureImpl();
    ~DXScreenCaptureImpl();

    bool Initialize(int width, int height);
    bool InitializeRegion(int x, int y, int width, int height);
    const uint8_t* CaptureBGR();
    const uint8_t* CaptureBMP();
    void SetRegion(int x, int y, int width, int height);
    void Release();
    int GetWidth() const { return outWidth; }
    int GetHeight() const { return outHeight; }
    int GetChannels() const { return outChannels; }

private:

    bool InitializeDXGI();
    void CleanupDXGI();
    bool InitializeTextures();
    void CleanupTextureCache();
    bool UpdateFrame();
    void ReleaseFrame();

    bool OnOutputChange();
    bool ReinitializeDuplication();
    void RecalculateCenterPosition();  // 新增：重新计算居中位置

    CachedTexturePair* GetCachedTexturePair(UINT width, UINT height);
    void* ConvertToBMP_Direct(const BYTE* srcData, int x, int y, int width, int height, int srcPitch, int* outSize);
    void* ConvertToBMP_Fast(const BYTE* srcData, int x, int y, int width, int height, int srcPitch, int* outSize);
    void ConvertBGRA2BGR_SIMD(const uint8_t* bgra, uint8_t* bgr, int width, int height, int stride);
    bool IsSSE42Supported();
    inline bool ValidateBMPParameters(const BYTE* srcData, int x, int y, int width, int height, int srcPitch);
    inline bool IsResolutionFirstCall(int width, int height);

    ID3D11Device* pd3dDevice;
    ID3D11DeviceContext* pd3dContext;
    IDXGIOutputDuplication* pDeskDupl;
    IDXGIOutput* pDXGIOutput;
    IDXGIAdapter1* pDXGIAdapter;
    IDXGIFactory1* pDXGIFactory;
    ID3D11Texture2D* frameTexture;

    int captureX;
    int captureY;
    int captureWidth;
    int captureHeight;
    int outWidth;
    int outHeight;
    int outChannels;

    std::vector<uint8_t> bgrBuffer;
    std::vector<uint8_t> bmpBuffer;
    size_t bgrBufferSize;
    size_t bmpBufferSize;

    std::unique_ptr<PreallocatedMemoryPool> memoryPool;
    CachedTexturePair texturePair_general;
    CachedTexturePair* currentTexturePair;
    bool textureInitialized;
    bool initialized;
    bool updated;

    bool sse42Supported;
    int screenWidth;
    int screenHeight;

    std::unordered_map<std::pair<int, int>, BMPHeaderCache, PairHash> bmpHeaderCache;
    std::vector<ResolutionFirstCall> resolutionFirstCalls;
    int frameCounter;

    std::atomic<int> reinitCount;
    bool useCenterMode;  // 新增：标记是否使用居中模式
};

class DXScreenCapture : public IScreenCapture {
public:
    DXScreenCapture();
    ~DXScreenCapture() override;

    bool Initialize(int width, int height) override;
    bool InitializeRegion(int x, int y, int width, int height) override;
    bool SetWindow(HWND hwnd) override { return false; }
    bool SetRegion(int x, int y, int width, int height) override;
    void Release() override;
    const unsigned char* CaptureBGR() override;
    const unsigned char* CaptureBMP() override;
    int GetWidth() const override;
    int GetHeight() const override;

private:
    std::unique_ptr<DXScreenCaptureImpl> impl;
    bool initialized;
};

#endif