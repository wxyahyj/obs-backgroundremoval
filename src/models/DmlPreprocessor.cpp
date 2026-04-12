#include "DmlPreprocessor.h"
#include <algorithm>
#include <cmath>
#include <cstring>

DmlPreprocessor::DmlPreprocessor()
    : initialized_(false)
    , fenceValue_(0)
    , fenceEvent_(nullptr)
    , sharedHandle_(nullptr)
{
}

DmlPreprocessor::~DmlPreprocessor()
{
    release();
}

bool DmlPreprocessor::initialize(ID3D11Device* d3d11Device)
{
    if (initialized_) {
        return true;
    }
    
    if (!d3d11Device) {
        return false;
    }
    
    d3d11Device_ = d3d11Device;
    d3d11Device_->GetImmediateContext(&d3d11Context_);
    
    if (!createD3D12Device()) {
        return false;
    }
    
    fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent_) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

void DmlPreprocessor::release()
{
    if (!initialized_) {
        return;
    }
    
    // 清理缓存的staging texture
    cachedStagingTexture_.Reset();
    memset(&cachedStagingDesc_, 0, sizeof(cachedStagingDesc_));
    
    sharedResource_.Reset();
    sharedTexture_.Reset();
    
    if (sharedHandle_) {
        CloseHandle(sharedHandle_);
        sharedHandle_ = nullptr;
    }
    
    commandList_.Reset();
    commandAllocator_.Reset();
    commandQueue_.Reset();
    
    if (fenceEvent_) {
        CloseHandle(fenceEvent_);
        fenceEvent_ = nullptr;
    }
    
    fence_.Reset();
    d3d12Device_.Reset();
    d3d11Context_.Reset();
    d3d11Device_.Reset();
    
    initialized_ = false;
}

bool DmlPreprocessor::createD3D12Device()
{
    HRESULT hr = D3D12CreateDevice(
        nullptr,
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(&d3d12Device_)
    );
    
    if (FAILED(hr)) {
        return false;
    }
    
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;
    
    hr = d3d12Device_->CreateCommandQueue(
        &queueDesc,
        IID_PPV_ARGS(&commandQueue_)
    );
    
    if (FAILED(hr)) {
        return false;
    }
    
    hr = d3d12Device_->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&commandAllocator_)
    );
    
    if (FAILED(hr)) {
        return false;
    }
    
    hr = d3d12Device_->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        commandAllocator_.Get(),
        nullptr,
        IID_PPV_ARGS(&commandList_)
    );
    
    if (FAILED(hr)) {
        return false;
    }
    
    commandList_->Close();
    
    hr = d3d12Device_->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&fence_)
    );
    
    return SUCCEEDED(hr);
}

bool DmlPreprocessor::createSharedResource(int width, int height)
{
    D3D11_TEXTURE2D_DESC sharedDesc = {};
    sharedDesc.Width = width;
    sharedDesc.Height = height;
    sharedDesc.MipLevels = 1;
    sharedDesc.ArraySize = 1;
    sharedDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sharedDesc.SampleDesc.Count = 1;
    sharedDesc.SampleDesc.Quality = 0;
    sharedDesc.Usage = D3D11_USAGE_DEFAULT;
    sharedDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    sharedDesc.CPUAccessFlags = 0;
    sharedDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    
    HRESULT hr = d3d11Device_->CreateTexture2D(&sharedDesc, nullptr, &sharedTexture_);
    if (FAILED(hr)) {
        return false;
    }
    
    ComPtr<IDXGIResource1> dxgiResource;
    hr = sharedTexture_.As(&dxgiResource);
    if (FAILED(hr)) {
        return false;
    }
    
    hr = dxgiResource->CreateSharedHandle(
        nullptr,
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        nullptr,
        &sharedHandle_
    );
    
    if (FAILED(hr)) {
        return false;
    }
    
    hr = d3d12Device_->OpenSharedHandle(sharedHandle_, IID_PPV_ARGS(&sharedResource_));
    
    return SUCCEEDED(hr);
}

DmlPreprocessParams DmlPreprocessor::calculateParams(
    int srcWidth, int srcHeight,
    int dstWidth, int dstHeight)
{
    DmlPreprocessParams params;
    params.srcWidth = srcWidth;
    params.srcHeight = srcHeight;
    params.dstWidth = dstWidth;
    params.dstHeight = dstHeight;
    
    // 计算letterbox参数
    float scaleX = static_cast<float>(dstWidth) / srcWidth;
    float scaleY = static_cast<float>(dstHeight) / srcHeight;
    params.scale = std::min(scaleX, scaleY);
    
    int newWidth = static_cast<int>(srcWidth * params.scale);
    int newHeight = static_cast<int>(srcHeight * params.scale);
    
    params.padX = (dstWidth - newWidth) / 2;
    params.padY = (dstHeight - newHeight) / 2;
    
    return params;
}

bool DmlPreprocessor::preprocessFromTexture(
    ID3D11Texture2D* srcTexture,
    float* dstBuffer,
    int dstWidth,
    int dstHeight,
    DmlPreprocessParams* outParams)
{
    if (!initialized_ || !srcTexture || !dstBuffer) {
        return false;
    }
    
    D3D11_TEXTURE2D_DESC srcDesc;
    srcTexture->GetDesc(&srcDesc);
    
    DmlPreprocessParams params = calculateParams(
        srcDesc.Width, srcDesc.Height,
        dstWidth, dstHeight
    );
    
    if (outParams) {
        *outParams = params;
    }
    
    return copyAndPreprocess(srcTexture, dstBuffer, params);
}

bool DmlPreprocessor::copyAndPreprocess(
    ID3D11Texture2D* srcTexture,
    float* dstBuffer,
    const DmlPreprocessParams& params)
{
    D3D11_TEXTURE2D_DESC srcDesc;
    srcTexture->GetDesc(&srcDesc);
    
    // 检查是否需要重新创建staging texture
    // 使用params中的尺寸而不是srcDesc中的尺寸
    bool needNewStaging = false;
    if (!cachedStagingTexture_ ||
        cachedStagingDesc_.Width != params.srcWidth ||
        cachedStagingDesc_.Height != params.srcHeight ||
        cachedStagingDesc_.Format != srcDesc.Format) {
        needNewStaging = true;
    }
    
    if (needNewStaging) {
        cachedStagingTexture_.Reset();
        
        D3D11_TEXTURE2D_DESC stagingDesc = srcDesc;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.BindFlags = 0;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stagingDesc.MiscFlags = 0;
        
        HRESULT hr = d3d11Device_->CreateTexture2D(&stagingDesc, nullptr, &cachedStagingTexture_);
        if (FAILED(hr)) {
            return false;
        }
        cachedStagingDesc_ = stagingDesc;
    }
    
    d3d11Context_->CopyResource(cachedStagingTexture_.Get(), srcTexture);
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = d3d11Context_->Map(cachedStagingTexture_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        return false;
    }
    
    int channelSize = params.dstWidth * params.dstHeight;
    float* dstR = dstBuffer;
    float* dstG = dstBuffer + channelSize;
    float* dstB = dstBuffer + channelSize * 2;
    
    float padValue = 114.0f / 255.0f;
    
    for (int dy = 0; dy < params.dstHeight; dy++) {
        for (int dx = 0; dx < params.dstWidth; dx++) {
            int idx = dy * params.dstWidth + dx;
            
            int srcX = static_cast<int>((dx - params.padX) / params.scale);
            int srcY = static_cast<int>((dy - params.padY) / params.scale);
            
            if (srcX >= 0 && srcX < static_cast<int>(params.srcWidth) &&
                srcY >= 0 && srcY < static_cast<int>(params.srcHeight)) {
                
                unsigned char* pixel = static_cast<unsigned char*>(mapped.pData) +
                    srcY * mapped.RowPitch + srcX * 4;
                
                dstR[idx] = static_cast<float>(pixel[2]) / 255.0f;
                dstG[idx] = static_cast<float>(pixel[1]) / 255.0f;
                dstB[idx] = static_cast<float>(pixel[0]) / 255.0f;
            } else {
                dstR[idx] = padValue;
                dstG[idx] = padValue;
                dstB[idx] = padValue;
            }
        }
    }
    
    d3d11Context_->Unmap(cachedStagingTexture_.Get(), 0);
    
    return true;
}
