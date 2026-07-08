#include "DmlPreprocessor.h"
#include <obs-module.h>
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
    if (initialized_) return true;
    fenceEvent_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    initialized_ = true;
    return true;
}

void DmlPreprocessor::release()
{
    if (!initialized_) return;
    cachedStagingTexture_.Reset();
    memset(&cachedStagingDesc_, 0, sizeof(cachedStagingDesc_));
    commandList_.Reset();
    commandAllocator_.Reset();
    commandQueue_.Reset();
    if (fenceEvent_) { CloseHandle(fenceEvent_); fenceEvent_ = nullptr; }
    fence_.Reset();
    d3d12Device_.Reset();
    initialized_ = false;
}

DmlPreprocessParams DmlPreprocessor::calculateParams(
    int srcWidth, int srcHeight, int dstWidth, int dstHeight)
{
    DmlPreprocessParams params;
    params.srcWidth = srcWidth; params.srcHeight = srcHeight;
    params.dstWidth = dstWidth; params.dstHeight = dstHeight;
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
    ID3D11Texture2D* srcTexture, float* dstBuffer,
    int dstWidth, int dstHeight, DmlPreprocessParams* outParams)
{
    if (!initialized_ || !srcTexture || !dstBuffer) return false;
    D3D11_TEXTURE2D_DESC srcDesc;
    srcTexture->GetDesc(&srcDesc);
    DmlPreprocessParams params = calculateParams(
        srcDesc.Width, srcDesc.Height, dstWidth, dstHeight);
    if (outParams) *outParams = params;
    return copyAndPreprocess(srcTexture, dstBuffer, params);
}

bool DmlPreprocessor::copyAndPreprocess(
    ID3D11Texture2D* srcTexture, float* dstBuffer,
    const DmlPreprocessParams& params)
{
    // D3D11 Immediate Context (CopyResource) 只能在渲染线程调用
    // 推理线程无法安全执行GPU→CPU纹理拷贝
    // 需要将此操作移到video_render回调中才能启用
    return false;
}
