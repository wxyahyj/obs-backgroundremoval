#define NOMINMAX
#include "DXScreenCapture.h"
#include <algorithm>
#include <intrin.h>
#include <thread>


DXScreenCaptureImpl::DXScreenCaptureImpl()
    : pd3dDevice(nullptr)
    , pd3dContext(nullptr)
    , pDeskDupl(nullptr)
    , pDXGIOutput(nullptr)
    , pDXGIAdapter(nullptr)
    , pDXGIFactory(nullptr)
    , frameTexture(nullptr)
    , captureX(0)
    , captureY(0)
    , captureWidth(0)
    , captureHeight(0)
    , outWidth(0)
    , outHeight(0)
    , outChannels(3)
    , bgrBufferSize(0)
    , bmpBufferSize(0)
    , currentTexturePair(nullptr)
    , textureInitialized(false)
    , initialized(false)
    , updated(false)
    , sse42Supported(false)
    , screenWidth(0)
    , screenHeight(0)
    , frameCounter(0)
    , reinitCount(0)
    , useCenterMode(false) {

    sse42Supported = IsSSE42Supported();
}

DXScreenCaptureImpl::~DXScreenCaptureImpl() {
    Release();
}

bool DXScreenCaptureImpl::IsSSE42Supported() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[2] & (1 << 20)) != 0;
}

inline bool DXScreenCaptureImpl::ValidateBMPParameters(const BYTE* srcData, int x, int y, int width, int height, int srcPitch) {
    return srcData && width > 0 && height > 0 && srcPitch > 0 && x >= 0 && y >= 0;
}

inline bool DXScreenCaptureImpl::IsResolutionFirstCall(int width, int height) {
    for (auto& rc : resolutionFirstCalls) {
        if (rc.IsFirstCall(width, height)) {
            return true;
        }
    }
    return false;
}

bool DXScreenCaptureImpl::InitializeDXGI() {
    if (initialized) return true;

    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pDXGIFactory);
    if (FAILED(hr)) return false;

    for (UINT i = 0; pDXGIFactory->EnumAdapters1(i, &pDXGIAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };

        UINT createFlags = 0;
#ifdef _DEBUG
        createFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        hr = D3D11CreateDevice(
            pDXGIAdapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            nullptr,
            createFlags,
            featureLevels,
            ARRAYSIZE(featureLevels),
            D3D11_SDK_VERSION,
            &pd3dDevice,
            nullptr,
            &pd3dContext
        );

        if (SUCCEEDED(hr)) {
            IDXGIOutput* tempOutput = nullptr;
            for (UINT j = 0; pDXGIAdapter->EnumOutputs(j, &tempOutput) != DXGI_ERROR_NOT_FOUND; ++j) {
                DXGI_OUTPUT_DESC outputDesc;
                if (SUCCEEDED(tempOutput->GetDesc(&outputDesc)) && outputDesc.AttachedToDesktop) {
                    pDXGIOutput = tempOutput;
                    screenWidth = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
                    screenHeight = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
                    break;
                }
                tempOutput->Release();
            }

            if (pDXGIOutput) break;

            pd3dDevice->Release();
            pd3dDevice = nullptr;
            pd3dContext->Release();
            pd3dContext = nullptr;
        }

        pDXGIAdapter->Release();
        pDXGIAdapter = nullptr;
    }

    if (!pd3dDevice || !pDXGIOutput) {
        CleanupDXGI();
        return false;
    }

    IDXGIOutput1* pDXGIOutput1 = nullptr;
    hr = pDXGIOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&pDXGIOutput1);
    if (FAILED(hr)) {
        CleanupDXGI();
        return false;
    }

    hr = pDXGIOutput1->DuplicateOutput(pd3dDevice, &pDeskDupl);
    pDXGIOutput1->Release();

    if (FAILED(hr)) {
        CleanupDXGI();
        return false;
    }

    memoryPool = std::make_unique<PreallocatedMemoryPool>();
    initialized = true;
    return true;
}

void DXScreenCaptureImpl::CleanupDXGI() {

    CleanupTextureCache();

    if (frameTexture) {
        frameTexture->Release();
        frameTexture = nullptr;
    }

    memoryPool.reset();

    if (pDeskDupl) {
        pDeskDupl->Release();
        pDeskDupl = nullptr;
    }
    if (pDXGIOutput) {
        pDXGIOutput->Release();
        pDXGIOutput = nullptr;
    }
    if (pDXGIAdapter) {
        pDXGIAdapter->Release();
        pDXGIAdapter = nullptr;
    }
    if (pd3dContext) {
        pd3dContext->Release();
        pd3dContext = nullptr;
    }
    if (pd3dDevice) {
        pd3dDevice->Release();
        pd3dDevice = nullptr;
    }
    if (pDXGIFactory) {
        pDXGIFactory->Release();
        pDXGIFactory = nullptr;
    }

    initialized = false;
}

bool DXScreenCaptureImpl::InitializeTextures() {
    if (textureInitialized) {
        CleanupTextureCache();
        textureInitialized = false;
    }

    D3D11_TEXTURE2D_DESC stagingDesc = {};
    stagingDesc.Width = captureWidth;
    stagingDesc.Height = captureHeight;
    stagingDesc.MipLevels = 1;
    stagingDesc.ArraySize = 1;
    stagingDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    stagingDesc.SampleDesc.Count = 1;
    stagingDesc.SampleDesc.Quality = 0;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    HRESULT hr = pd3dDevice->CreateTexture2D(&stagingDesc, nullptr, &texturePair_general.stagingTexture);

    if (FAILED(hr) || !texturePair_general.stagingTexture) {
        CleanupTextureCache();
        return false;
    }

    // 创建GPU同步Query
    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_EVENT;
    queryDesc.MiscFlags = 0;

    hr = pd3dDevice->CreateQuery(&queryDesc, &texturePair_general.syncQuery);
    if (FAILED(hr)) {
        CleanupTextureCache();
        return false;
    }

    texturePair_general.width = captureWidth;
    texturePair_general.height = captureHeight;
    texturePair_general.inUse = false;

    textureInitialized = true;
    return true;
}

void DXScreenCaptureImpl::CleanupTextureCache() {
    if (texturePair_general.isMapped && texturePair_general.stagingTexture && pd3dContext) {
        pd3dContext->Unmap(texturePair_general.stagingTexture, 0);
        texturePair_general.isMapped = false;
    }

    if (texturePair_general.stagingTexture) {
        texturePair_general.stagingTexture->Release();
        texturePair_general.stagingTexture = nullptr;
    }

    if (texturePair_general.syncQuery) {
        texturePair_general.syncQuery->Release();
        texturePair_general.syncQuery = nullptr;
    }

    textureInitialized = false;
}

CachedTexturePair* DXScreenCaptureImpl::GetCachedTexturePair(UINT width, UINT height) {
    if (textureInitialized &&
        texturePair_general.stagingTexture &&
        texturePair_general.width == width &&
        texturePair_general.height == height &&
        !texturePair_general.inUse) {

        texturePair_general.inUse = true;
        return &texturePair_general;
    }

    return nullptr;
}

void DXScreenCaptureImpl::RecalculateCenterPosition() {
    int centerX = (screenWidth - captureWidth) / 2;
    int centerY = (screenHeight - captureHeight) / 2;

    captureX = std::max(0, centerX);
    captureY = std::max(0, centerY);

    if (captureX + captureWidth > screenWidth) {
        captureWidth = screenWidth - captureX;
    }
    if (captureY + captureHeight > screenHeight) {
        captureHeight = screenHeight - captureY;
    }
}

bool DXScreenCaptureImpl::OnOutputChange() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (pDeskDupl) {
        pDeskDupl->Release();
        pDeskDupl = nullptr;
    }

    CleanupTextureCache();

    if (frameTexture) {
        frameTexture->Release();
        frameTexture = nullptr;
    }

    return ReinitializeDuplication();
}

bool DXScreenCaptureImpl::ReinitializeDuplication() {
    const int maxRetries = 5;
    int retryCount = 0;

    while (retryCount < maxRetries) {
        if (pDXGIOutput) {
            DXGI_OUTPUT_DESC outputDesc;
            HRESULT hr = pDXGIOutput->GetDesc(&outputDesc);
            if (SUCCEEDED(hr)) {
                screenWidth = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
                screenHeight = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
            }
        }

        if (useCenterMode) {
            RecalculateCenterPosition();
        }

        IDXGIOutput1* pDXGIOutput1 = nullptr;
        HRESULT hr = pDXGIOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&pDXGIOutput1);
        if (FAILED(hr)) {
            retryCount++;
            std::this_thread::sleep_for(std::chrono::milliseconds(200 * retryCount));
            continue;
        }

        hr = pDXGIOutput1->DuplicateOutput(pd3dDevice, &pDeskDupl);
        pDXGIOutput1->Release();

        if (SUCCEEDED(hr)) {
            if (!InitializeTextures()) {
                pDeskDupl->Release();
                pDeskDupl = nullptr;
                retryCount++;
                std::this_thread::sleep_for(std::chrono::milliseconds(200 * retryCount));
                continue;
            }

            reinitCount++;
            return true;
        }

        retryCount++;
        std::this_thread::sleep_for(std::chrono::milliseconds(200 * retryCount));
    }

    return false;
}

bool DXScreenCaptureImpl::UpdateFrame() {
    if (!pDeskDupl) return false;

    IDXGIResource* pDesktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;

    HRESULT hr = pDeskDupl->AcquireNextFrame(0, &frameInfo, &pDesktopResource);

    switch (hr) {
    case S_OK:
        break;
    case DXGI_ERROR_WAIT_TIMEOUT:
        updated = false;
        return true;
    case DXGI_ERROR_ACCESS_LOST:
    case DXGI_ERROR_DEVICE_REMOVED:
        if (OnOutputChange()) {
            updated = false;
            return true;
        }
        return false;
    default:
        return false;
    }

    if (frameInfo.AccumulatedFrames > 0) {
        if (frameTexture) {
            frameTexture->Release();
            frameTexture = nullptr;
        }

        hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&frameTexture);
        pDesktopResource->Release();

        if (SUCCEEDED(hr)) {
            updated = true;
            return true;
        }
    }
    else {
        updated = false;
    }

    pDeskDupl->ReleaseFrame();
    if (pDesktopResource) {
        pDesktopResource->Release();
    }
    return true;
}

void DXScreenCaptureImpl::ReleaseFrame() {
    if (pDeskDupl) {
        pDeskDupl->ReleaseFrame();
    }
}

void DXScreenCaptureImpl::ConvertBGRA2BGR_SIMD(const uint8_t* src, uint8_t* dest, int width, int height, int srcPitch) {
    if (sse42Supported && width >= 4) {
        for (int y = 0; y < height; y++) {
            const uint8_t* srcRow = src + y * srcPitch;
            uint8_t* destRow = dest + y * width * 3;

            if (y + 1 < height) {
                _mm_prefetch((const char*)(src + (y + 1) * srcPitch), _MM_HINT_T0);
            }

            int pixelX = 0;
            for (; pixelX <= width - 4; pixelX += 4) {
                __m128i bgra = _mm_loadu_si128((__m128i*)(srcRow + pixelX * 4));
                __m128i bgr = _mm_shuffle_epi8(bgra,
                    _mm_setr_epi8(
                        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                        0, 0, 0, 0
                    ));
                _mm_storeu_si128((__m128i*)(destRow + pixelX * 3), bgr);
            }

            for (; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }
    else {
        for (int y = 0; y < height; y++) {
            const uint8_t* srcRow = src + y * srcPitch;
            uint8_t* destRow = dest + y * width * 3;

            for (int pixelX = 0; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }
}

void* DXScreenCaptureImpl::ConvertToBMP_Direct(const BYTE* srcData, int x, int y, int width, int height, int srcPitch, int* outSize) {
    if (!ValidateBMPParameters(srcData, x, y, width, height, srcPitch)) {
        *outSize = 0;
        return nullptr;
    }

    BMPHeaderCache* headerCache = nullptr;
    {
        auto key = std::make_pair(width, height);
        auto it = bmpHeaderCache.find(key);
        if (it == bmpHeaderCache.end()) {
            bmpHeaderCache[key].Initialize(width, height);
        }
        headerCache = &bmpHeaderCache[key];
    }

    int totalSize = headerCache->totalSize;
    int rowSize = headerCache->rowSize;

    void* bmpData = memoryPool->GetBmpBlock(totalSize);
    if (!bmpData) {
        bmpData = malloc(totalSize);
    }

    if (!bmpData) {
        *outSize = 0;
        return nullptr;
    }

    if (totalSize >= sizeof(BMPFileHeader) + sizeof(BMPInfoHeader)) {
        memcpy(bmpData, &headerCache->fileHeader, sizeof(BMPFileHeader));
        memcpy((BYTE*)bmpData + sizeof(BMPFileHeader), &headerCache->infoHeader, sizeof(BMPInfoHeader));
    }

    BYTE* destData = (BYTE*)bmpData + sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    if (sse42Supported && width >= 4) {
        for (int y = 0; y < height; y++) {
            const BYTE* srcRow = srcData + (height - 1 - y) * srcPitch + x * 4;
            BYTE* destRow = destData + y * rowSize;

            if (y + 1 < height) {
                _mm_prefetch((const char*)(srcData + (height - 1 - (y + 1)) * srcPitch + x * 4), _MM_HINT_T0);
            }

            int pixelX = 0;
            for (; pixelX <= width - 4; pixelX += 4) {
                __m128i bgra = _mm_loadu_si128((__m128i*)(srcRow + pixelX * 4));
                __m128i bgr = _mm_shuffle_epi8(bgra,
                    _mm_setr_epi8(
                        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                        0, 0, 0, 0
                    ));
                _mm_storeu_si128((__m128i*)(destRow + pixelX * 3), bgr);
            }

            for (; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }
    else {
        for (int y = 0; y < height; y++) {
            const BYTE* srcRow = srcData + (height - 1 - y) * srcPitch + x * 4;
            BYTE* destRow = destData + y * rowSize;

            for (int pixelX = 0; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }

    *outSize = totalSize;
    return bmpData;
}

void* DXScreenCaptureImpl::ConvertToBMP_Fast(const BYTE* srcData, int x, int y, int width, int height, int srcPitch, int* outSize) {
    if (!ValidateBMPParameters(srcData, x, y, width, height, srcPitch)) {
        *outSize = 0;
        return nullptr;
    }

    int rowSize = ((width * 3 + 3) / 4) * 4;
    int imageSize = rowSize * height;
    int totalSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + imageSize;

    void* bmpData = memoryPool->GetBmpBlock(totalSize);
    if (!bmpData) {
        bmpData = malloc(totalSize);
    }

    if (!bmpData) {
        *outSize = 0;
        return nullptr;
    }

    BMPFileHeader* fileHeader = (BMPFileHeader*)bmpData;
    fileHeader->bfType = 0x4D42;
    fileHeader->bfSize = totalSize;
    fileHeader->bfReserved1 = 0;
    fileHeader->bfReserved2 = 0;
    fileHeader->bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    BMPInfoHeader* infoHeader = (BMPInfoHeader*)((BYTE*)bmpData + sizeof(BMPFileHeader));
    infoHeader->biSize = sizeof(BMPInfoHeader);
    infoHeader->biWidth = width;
    infoHeader->biHeight = height;
    infoHeader->biPlanes = 1;
    infoHeader->biBitCount = 24;
    infoHeader->biCompression = 0;
    infoHeader->biSizeImage = imageSize;
    infoHeader->biXPelsPerMeter = 0;
    infoHeader->biYPelsPerMeter = 0;
    infoHeader->biClrUsed = 0;
    infoHeader->biClrImportant = 0;

    BYTE* destData = (BYTE*)bmpData + sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    if (sse42Supported && width >= 4) {
        for (int y = 0; y < height; y++) {
            const BYTE* srcRow = srcData + (height - 1 - y) * srcPitch;
            BYTE* destRow = destData + y * rowSize;

            if (y + 1 < height) {
                _mm_prefetch((const char*)(srcData + (height - 1 - (y + 1)) * srcPitch), _MM_HINT_T0);
            }

            int pixelX = 0;
            for (; pixelX <= width - 4; pixelX += 4) {
                __m128i bgra = _mm_loadu_si128((__m128i*)(srcRow + pixelX * 4));
                __m128i bgr = _mm_shuffle_epi8(bgra,
                    _mm_setr_epi8(
                        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                        0, 0, 0, 0
                    ));
                _mm_storeu_si128((__m128i*)(destRow + pixelX * 3), bgr);
            }

            for (; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }
    else {
        for (int y = 0; y < height; y++) {
            const BYTE* srcRow = srcData + (height - 1 - y) * srcPitch;
            BYTE* destRow = destData + y * rowSize;

            for (int pixelX = 0; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }

    *outSize = totalSize;
    return bmpData;
}

bool DXScreenCaptureImpl::Initialize(int width, int height) {
    if (!InitializeDXGI()) {
        return false;
    }

    int centerX = (screenWidth - width) / 2;
    int centerY = (screenHeight - height) / 2;

    centerX = std::max(0, centerX);
    centerY = std::max(0, centerY);

    useCenterMode = true;

    bool result = InitializeRegion(centerX, centerY, width, height);

    useCenterMode = true;

    return result;
}

bool DXScreenCaptureImpl::InitializeRegion(int x, int y, int width, int height) {
    Release();

    if (!InitializeDXGI()) {
        return false;
    }

    useCenterMode = false;

    captureX = std::max(0, std::min(x, screenWidth));
    captureY = std::max(0, std::min(y, screenHeight));
    captureWidth = std::max(1, std::min(width, screenWidth - captureX));
    captureHeight = std::max(1, std::min(height, screenHeight - captureY));

    bool isNewResolution = true;
    for (auto& rc : resolutionFirstCalls) {
        if (rc.width == captureWidth && rc.height == captureHeight) {
            isNewResolution = false;
            break;
        }
    }
    if (isNewResolution) {
        resolutionFirstCalls.emplace_back(captureWidth, captureHeight);
    }

    if (memoryPool && !memoryPool->Initialize(captureWidth, captureHeight)) {
        return false;
    }

    if (!InitializeTextures()) {
        return false;
    }

    bgrBufferSize = captureWidth * captureHeight * 3;
    bmpBufferSize = captureWidth * captureHeight * 3 + 1024;
    bgrBuffer.resize(bgrBufferSize);
    bmpBuffer.resize(bmpBufferSize);

    outWidth = captureWidth;
    outHeight = captureHeight;
    outChannels = 3;

    return true;
}

void DXScreenCaptureImpl::SetRegion(int x, int y, int width, int height) {
    InitializeRegion(x, y, width, height);
}

const uint8_t* DXScreenCaptureImpl::CaptureBGR() {
    if (!initialized || !pd3dDevice || !pd3dContext || !pDeskDupl) {
        return nullptr;
    }

    // 分辨率首次调用检查（保持原逻辑）
    if (IsResolutionFirstCall(captureWidth, captureHeight)) {
        if (!initialized || !pd3dDevice || !pd3dContext || !pDeskDupl) {
            return nullptr;
        }
    }

    //  等待获取新帧
    while (true) {
        if (!UpdateFrame()) {
            return nullptr;
        }

        if (!updated || !frameTexture) {
            continue;
        }
        break;
    }

    //  获取纹理对
    CachedTexturePair* pTexturePair = GetCachedTexturePair(captureWidth, captureHeight);
    if (!pTexturePair || !pTexturePair->stagingTexture) {
        if (pTexturePair) pTexturePair->inUse = false;
        return nullptr;
    }

    //  定义捕获区域
    D3D11_BOX sourceBox = {
        (UINT)captureX, (UINT)captureY, 0,
        (UINT)(captureX + captureWidth), (UINT)(captureY + captureHeight), 1
    };

    //  GPU拷贝操作（异步）
    pd3dContext->CopySubresourceRegion(
        pTexturePair->stagingTexture, 0, 0, 0, 0,
        frameTexture, 0, &sourceBox
    );

    //  标记GPU同步点
    pd3dContext->End(pTexturePair->syncQuery);

    //  立即释放Desktop Duplication帧锁
    ReleaseFrame();

    //  关键：强制等待GPU完成拷贝
    BOOL queryData = FALSE;
    while (pd3dContext->GetData(pTexturePair->syncQuery, &queryData, sizeof(BOOL), 0) == S_FALSE) {
        // 主动让出CPU时间片，避免空转
        std::this_thread::yield();
    }

    //  关键：即使已Map，也重新Map以确保读取最新数据
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    if (pTexturePair->isMapped) {
        // 先Unmap
        pd3dContext->Unmap(pTexturePair->stagingTexture, 0);
        pTexturePair->isMapped = false;
    }

    // 重新Map（此时GPU已完成拷贝，Map不会阻塞）
    HRESULT hr = pd3dContext->Map(
        pTexturePair->stagingTexture, 0,
        D3D11_MAP_READ,
        0,  // flags=0，阻塞模式（但由于Query已完成，几乎不阻塞）
        &mappedResource
    );

    if (FAILED(hr)) {
        pTexturePair->inUse = false;
        return nullptr;
    }

    // 更新缓存的映射信息
    pTexturePair->isMapped = true;
    pTexturePair->mappedResource = mappedResource;

    //  SIMD转换 BGRA -> BGR
    BYTE* srcData = (BYTE*)mappedResource.pData;
    ConvertBGRA2BGR_SIMD(
        srcData,
        bgrBuffer.data(),
        captureWidth,
        captureHeight,
        mappedResource.RowPitch
    );

    memoryPool->ReleaseCurrent();

    //  释放纹理占用标志（保持Map状态）
    pTexturePair->inUse = false;

    return bgrBuffer.data();
}

const uint8_t* DXScreenCaptureImpl::CaptureBMP() {
    if (!initialized || !pd3dDevice || !pd3dContext || !pDeskDupl) {
        return nullptr;
    }

    if (IsResolutionFirstCall(captureWidth, captureHeight)) {
        if (!initialized || !pd3dDevice || !pd3dContext || !pDeskDupl) {
            return nullptr;
        }
    }


    while (true) {
        if (!UpdateFrame()) {
            return nullptr;
        }

        if (!updated || !frameTexture) {
            continue;
        }
        break;
    }

    CachedTexturePair* pTexturePair = GetCachedTexturePair(captureWidth, captureHeight);
    if (!pTexturePair || !pTexturePair->stagingTexture) {
        if (pTexturePair) pTexturePair->inUse = false;
        return nullptr;
    }

    D3D11_BOX sourceBox = {
        (UINT)captureX, (UINT)captureY, 0,
        (UINT)(captureX + captureWidth), (UINT)(captureY + captureHeight), 1
    };

    pd3dContext->CopySubresourceRegion(
        pTexturePair->stagingTexture, 0, 0, 0, 0,
        frameTexture, 0, &sourceBox
    );

    // 添加Query同步
    pd3dContext->End(pTexturePair->syncQuery);
    ReleaseFrame();

    // 等待GPU完成
    BOOL queryData = FALSE;
    while (pd3dContext->GetData(pTexturePair->syncQuery, &queryData, sizeof(BOOL), 0) == S_FALSE) {
        std::this_thread::yield();
    }

    // 重新Map确保同步
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    if (pTexturePair->isMapped) {
        pd3dContext->Unmap(pTexturePair->stagingTexture, 0);
        pTexturePair->isMapped = false;
    }

    HRESULT hr = pd3dContext->Map(
        pTexturePair->stagingTexture, 0,
        D3D11_MAP_READ,
        0,
        &mappedResource
    );

    if (FAILED(hr)) {
        pTexturePair->inUse = false;
        return nullptr;
    }

    pTexturePair->isMapped = true;
    pTexturePair->mappedResource = mappedResource;

    BYTE* srcData = (BYTE*)mappedResource.pData;
    int bmpSize = 0;
    void* bmpData = ConvertToBMP_Fast(srcData, 0, 0, captureWidth, captureHeight, mappedResource.RowPitch, &bmpSize);

    if (!bmpData || bmpSize == 0) {
        pTexturePair->inUse = false;
        return nullptr;
    }

    if (bmpSize > bmpBufferSize) {
        bmpBufferSize = bmpSize;
        bmpBuffer.resize(bmpBufferSize);
    }
    memcpy(bmpBuffer.data(), bmpData, bmpSize);

    memoryPool->ReleaseCurrent();

    pTexturePair->inUse = false;

    return bmpBuffer.data();
}

void DXScreenCaptureImpl::Release() {
    CleanupDXGI();
    bgrBuffer.clear();
    bmpBuffer.clear();
    bgrBufferSize = 0;
    bmpBufferSize = 0;

    bmpHeaderCache.clear();

    resolutionFirstCalls.clear();
    frameCounter = 0;
    updated = false;
}

DXScreenCapture::DXScreenCapture()
    : initialized(false) {
}

DXScreenCapture::~DXScreenCapture() {
    Release();
}

bool DXScreenCapture::Initialize(int width, int height) {
    Release();

    impl = std::make_unique<DXScreenCaptureImpl>();
    if (!impl->Initialize(width, height)) {
        impl.reset();
        return false;
    }

    initialized = true;
    return true;
}

bool DXScreenCapture::InitializeRegion(int x, int y, int width, int height) {
    Release();

    impl = std::make_unique<DXScreenCaptureImpl>();
    if (!impl->InitializeRegion(x, y, width, height)) {
        impl.reset();
        return false;
    }

    initialized = true;
    return true;
}

bool DXScreenCapture::SetRegion(int x, int y, int width, int height) {
    if (!initialized || !impl) {
        return false;
    }

    impl->SetRegion(x, y, width, height);
    return true;
}

const unsigned char* DXScreenCapture::CaptureBGR() {
    if (!initialized || !impl) {
        return nullptr;
    }

    return reinterpret_cast<const unsigned char*>(impl->CaptureBGR());
}

const unsigned char* DXScreenCapture::CaptureBMP() {
    if (!initialized || !impl) {
        return nullptr;
    }

    return reinterpret_cast<const unsigned char*>(impl->CaptureBMP());
}

int DXScreenCapture::GetWidth() const {
    if (!initialized || !impl) {
        return 0;
    }

    return impl->GetWidth();
}

int DXScreenCapture::GetHeight() const {
    if (!initialized || !impl) {
        return 0;
    }

    return impl->GetHeight();
}

void DXScreenCapture::Release() {
    if (impl) {
        impl->Release();
        impl.reset();
    }
    initialized = false;
}



