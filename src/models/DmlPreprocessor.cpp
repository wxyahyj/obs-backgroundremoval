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
    // Initialized successfully
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

    const int channels = 3; // RGB
    const int totalPixels = dstWidth * dstHeight * channels;
    outFrame.data.resize(totalPixels);
    outFrame.width = dstWidth;
    outFrame.height = dstHeight;
    outFrame.channels = channels;
    outFrame.srcWidth = srcWidth;
    outFrame.srcHeight = srcHeight;

    const int newWidth = static_cast<int>(srcWidth * params.scale);
    const int newHeight = static_cast<int>(srcHeight * params.scale);

    // Nearest-neighbour resize + BGRA->RGB + pad, written as CHW.
    for (int dy = 0; dy < dstHeight; ++dy) {
        for (int dx = 0; dx < dstWidth; ++dx) {
            int sx = dx - params.padX;
            int sy = dy - params.padY;

            float r = 0.0f, g = 0.0f, b = 0.0f;

            if (sx >= 0 && sx < newWidth && sy >= 0 && sy < newHeight) {
                int srcX = static_cast<int>(sx / params.scale);
                int srcY = static_cast<int>(sy / params.scale);
                srcX = std::clamp(srcX, 0, srcWidth - 1);
                srcY = std::clamp(srcY, 0, srcHeight - 1);

                const uint8_t* row = bgraData + srcY * srcStrideBytes;
                const uint8_t* pixel = row + srcX * 4;
                b = pixel[0] / 255.0f;
                g = pixel[1] / 255.0f;
                r = pixel[2] / 255.0f;
            }
            // else: pad region stays 0.0f (black padding)

            const int planeSize = dstWidth * dstHeight;
            outFrame.data[dy * dstWidth + dx] = r;
            outFrame.data[planeSize + dy * dstWidth + dx] = g;
            outFrame.data[2 * planeSize + dy * dstWidth + dx] = b;
        }
    }

    return true;
}
