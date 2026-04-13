#include "DmlPreprocessor.h"
#include <algorithm>
#include <cmath>
#include <cstring>

DmlPreprocessor::DmlPreprocessor()
    : initialized_(false)
    , fenceValue_(0)
    , fenceEvent_(nullptr)
{
}

DmlPreprocessor::~DmlPreprocessor()
{
    release();
}

bool DmlPreprocessor::initialize()
{
    if (initialized_) {
        return true;
    }
    
    // 创建D3D12设备（用于未来的GPU加速预处理，可选）
    if (!createD3D12Device()) {
        // D3D12设备创建失败不影响基本功能
        // 可以在没有D3D12的情况下运行CPU预处理
    }
    
    fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent_) {
        // 即使fence创建失败，也可以继续
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
    
    commandList_.Reset();
    commandAllocator_.Reset();
    commandQueue_.Reset();
    
    if (fenceEvent_) {
        CloseHandle(fenceEvent_);
        fenceEvent_ = nullptr;
    }
    
    fence_.Reset();
    d3d12Device_.Reset();
    
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
    // 从源纹理获取OBS的D3D11设备
    // 这是关键修复：必须使用纹理所属的设备，而不是独立创建的设备
    // 因为CopyResource要求源和目标资源在同一个设备上
    ComPtr<ID3D11Device> obsDevice;
    HRESULT hr = srcTexture->GetDevice(&obsDevice);
    if (FAILED(hr) || !obsDevice) {
        return false;
    }
    
    // 获取OBS的D3D11设备上下文
    ComPtr<ID3D11DeviceContext> obsContext;
    obsDevice->GetImmediateContext(&obsContext);
    if (!obsContext) {
        return false;
    }
    
    D3D11_TEXTURE2D_DESC srcDesc;
    srcTexture->GetDesc(&srcDesc);
    
    // 检查是否需要重新创建staging texture
    // 使用params中的尺寸而不是srcDesc中的尺寸
    // 
    // 类型转换说明：
    // - D3D11_TEXTURE2D_DESC的Width/Height是UINT类型
    // - DmlPreprocessParams的srcWidth/srcHeight是int类型（便于循环索引和负值检查）
    // - 使用static_cast<UINT>进行显式转换，避免有符号/无符号比较警告
    // - 此处尺寸值不会为负数，转换是安全的
    bool needNewStaging = false;
    if (!cachedStagingTexture_ ||
        cachedStagingDesc_.Width != static_cast<UINT>(params.srcWidth) ||
        cachedStagingDesc_.Height != static_cast<UINT>(params.srcHeight) ||
        cachedStagingDesc_.Format != srcDesc.Format) {
        needNewStaging = true;
    }
    
    if (needNewStaging) {
        cachedStagingTexture_.Reset();
        
        D3D11_TEXTURE2D_DESC stagingDesc = srcDesc;
        stagingDesc.Width = static_cast<UINT>(params.srcWidth);
        stagingDesc.Height = static_cast<UINT>(params.srcHeight);
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.BindFlags = 0;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stagingDesc.MiscFlags = 0;
        
        // 使用OBS设备创建staging texture
        hr = obsDevice->CreateTexture2D(&stagingDesc, nullptr, &cachedStagingTexture_);
        if (FAILED(hr)) {
            return false;
        }
        cachedStagingDesc_ = stagingDesc;
    }
    
    // 使用OBS设备上下文执行CopyResource
    obsContext->CopyResource(cachedStagingTexture_.Get(), srcTexture);
    
    // 使用OBS设备上下文执行Map
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = obsContext->Map(cachedStagingTexture_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
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
    
    // 使用OBS设备上下文执行Unmap
    obsContext->Unmap(cachedStagingTexture_.Get(), 0);
    
    return true;
}
