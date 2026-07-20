#include "DmlPreprocessor.h"
#include <obs-module.h>
#include <algorithm>
#include <cmath>
#include <cstring>

DmlPreprocessor::DmlPreprocessor()
    : initialized_(false)
{
}

DmlPreprocessor::~DmlPreprocessor()
{
    release();
}

bool DmlPreprocessor::initialize()
{
    if (initialized_) return true;
    initialized_ = true;
    return true;
}

void DmlPreprocessor::release()
{
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

bool DmlPreprocessor::preprocessFromBgra(
    const uint8_t* bgraData,
    int srcWidth,
    int srcHeight,
    int srcStrideBytes,
    int dstWidth,
    int dstHeight,
    DmlPreprocessedFrame& outFrame,
    DmlPreprocessParams* outParams)
{
    if (!initialized_ || !bgraData || srcWidth <= 0 || srcHeight <= 0
        || dstWidth <= 0 || dstHeight <= 0)
        return false;

    DmlPreprocessParams params = calculateParams(
        srcWidth, srcHeight, dstWidth, dstHeight);
    if (outParams) *outParams = params;

    const int channels = 3;
    const int planeSize = dstWidth * dstHeight;
    const int totalPixels = planeSize * channels;
    const float padVal = 114.0f / 255.0f;

    const bool sizeChanged =
        outFrame.width != dstWidth || outFrame.height != dstHeight ||
        static_cast<int>(outFrame.data.size()) != totalPixels;

    if (sizeChanged) {
        outFrame.data.assign(static_cast<size_t>(totalPixels), padVal);
    }
    // size 不变：不整缓冲 assign；内容区会被覆盖，pad 保持

    outFrame.width = dstWidth;
    outFrame.height = dstHeight;
    outFrame.channels = channels;
    outFrame.srcWidth = srcWidth;
    outFrame.srcHeight = srcHeight;

    const int newWidth = static_cast<int>(srcWidth * params.scale);
    const int newHeight = static_cast<int>(srcHeight * params.scale);
    const float invScale = (params.scale > 1e-8f) ? (1.0f / params.scale) : 0.0f;
    const float inv255 = 1.0f / 255.0f;

    float* rPlane = outFrame.data.data();
    float* gPlane = outFrame.data.data() + planeSize;
    float* bPlane = outFrame.data.data() + 2 * planeSize;

    const int y0 = std::max(0, params.padY);
    const int y1 = std::min(dstHeight, params.padY + newHeight);
    const int x0 = std::max(0, params.padX);
    const int x1 = std::min(dstWidth, params.padX + newWidth);

    for (int dy = y0; dy < y1; ++dy) {
        const int sy = static_cast<int>((dy - params.padY) * invScale);
        const int srcY = std::clamp(sy, 0, srcHeight - 1);
        const uint8_t* row = bgraData + srcY * srcStrideBytes;
        const int rowBase = dy * dstWidth;
        int dx = x0;
        // 4 像素一批手写展开，减循环开销
        for (; dx + 3 < x1; dx += 4) {
            for (int k = 0; k < 4; ++k) {
                const int sx = static_cast<int>((dx + k - params.padX) * invScale);
                const int srcX = std::clamp(sx, 0, srcWidth - 1);
                const uint8_t* pixel = row + srcX * 4;
                const int idx = rowBase + dx + k;
                bPlane[idx] = pixel[0] * inv255;
                gPlane[idx] = pixel[1] * inv255;
                rPlane[idx] = pixel[2] * inv255;
            }
        }
        for (; dx < x1; ++dx) {
            const int sx = static_cast<int>((dx - params.padX) * invScale);
            const int srcX = std::clamp(sx, 0, srcWidth - 1);
            const uint8_t* pixel = row + srcX * 4;
            const int idx = rowBase + dx;
            bPlane[idx] = pixel[0] * inv255;
            gPlane[idx] = pixel[1] * inv255;
            rPlane[idx] = pixel[2] * inv255;
        }
    }

    return true;
}
