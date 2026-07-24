#include "CudaPreprocessor.cuh"
#include <device_launch_parameters.h>
#include <cuda_surface_types.h>
#include <surface_functions.h>
#include <cstring>

__constant__ float cPadValue[4] = {114.0f / 255.0f, 114.0f / 255.0f, 114.0f / 255.0f, 1.0f};

// Samples full surface; src coords are absolute texture coords.
__global__ void letterboxKernelCrop(
    cudaSurfaceObject_t srcSurface,
    float* dstR,
    float* dstG,
    float* dstB,
    int cropX,
    int cropY,
    int cropW,
    int cropH,
    int dstWidth,
    int dstHeight,
    float scale,
    int padX,
    int padY
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dstWidth || dy >= dstHeight) return;

    int dstIdx = dy * dstWidth + dx;

    int localX = static_cast<int>((dx - padX) / scale);
    int localY = static_cast<int>((dy - padY) / scale);

    if (localX >= 0 && localX < cropW && localY >= 0 && localY < cropH) {
        int srcX = cropX + localX;
        int srcY = cropY + localY;
        uchar4 bgra;
        surf2Dread(&bgra, srcSurface, srcX * sizeof(uchar4), srcY);

        dstR[dstIdx] = static_cast<float>(bgra.z) / 255.0f;
        dstG[dstIdx] = static_cast<float>(bgra.y) / 255.0f;
        dstB[dstIdx] = static_cast<float>(bgra.x) / 255.0f;
    } else {
        dstR[dstIdx] = cPadValue[0];
        dstG[dstIdx] = cPadValue[1];
        dstB[dstIdx] = cPadValue[2];
    }
}

__global__ void letterboxKernel(
    cudaSurfaceObject_t srcSurface,
    float* dstR,
    float* dstG,
    float* dstB,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    float scale,
    int padX,
    int padY
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dstWidth || dy >= dstHeight) return;

    int dstIdx = dy * dstWidth + dx;

    int srcX = static_cast<int>((dx - padX) / scale);
    int srcY = static_cast<int>((dy - padY) / scale);

    if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
        uchar4 bgra;
        surf2Dread(&bgra, srcSurface, srcX * sizeof(uchar4), srcY);

        dstR[dstIdx] = static_cast<float>(bgra.z) / 255.0f;
        dstG[dstIdx] = static_cast<float>(bgra.y) / 255.0f;
        dstB[dstIdx] = static_cast<float>(bgra.x) / 255.0f;
    } else {
        dstR[dstIdx] = cPadValue[0];
        dstG[dstIdx] = cPadValue[1];
        dstB[dstIdx] = cPadValue[2];
    }
}

__global__ void preprocessBGRAKernel(
    const unsigned char* srcBGRA,
    float* dstR,
    float* dstG,
    float* dstB,
    int srcWidth,
    int srcHeight,
    int srcPitch,
    int dstWidth,
    int dstHeight,
    float scale,
    int padX,
    int padY
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dstWidth || dy >= dstHeight) return;

    int dstIdx = dy * dstWidth + dx;

    int srcX = static_cast<int>((dx - padX) / scale);
    int srcY = static_cast<int>((dy - padY) / scale);

    if (srcX >= 0 && srcX < srcWidth && srcY >= 0 && srcY < srcHeight) {
        int srcIdx = srcY * srcPitch + srcX * 4;

        dstR[dstIdx] = static_cast<float>(srcBGRA[srcIdx + 2]) / 255.0f;
        dstG[dstIdx] = static_cast<float>(srcBGRA[srcIdx + 1]) / 255.0f;
        dstB[dstIdx] = static_cast<float>(srcBGRA[srcIdx + 0]) / 255.0f;
    } else {
        dstR[dstIdx] = cPadValue[0];
        dstG[dstIdx] = cPadValue[1];
        dstB[dstIdx] = cPadValue[2];
    }
}

LetterboxParams calculateLetterboxParams(
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight
) {
    // Align with AiMod / ModelYOLO: round unpad + round pad
    LetterboxParams params;
    params.srcWidth = srcWidth;
    params.srcHeight = srcHeight;
    params.dstWidth = dstWidth;
    params.dstHeight = dstHeight;

    float scaleX = static_cast<float>(dstWidth) / srcWidth;
    float scaleY = static_cast<float>(dstHeight) / srcHeight;
    params.scale = (scaleX < scaleY) ? scaleX : scaleY;

    int newWidth = static_cast<int>(srcWidth * params.scale + 0.5f);
    int newHeight = static_cast<int>(srcHeight * params.scale + 0.5f);
    if (newWidth < 1) newWidth = 1;
    if (newHeight < 1) newHeight = 1;
    if (newWidth > dstWidth) newWidth = dstWidth;
    if (newHeight > dstHeight) newHeight = dstHeight;

    float dw = (dstWidth - newWidth) * 0.5f;
    float dh = (dstHeight - newHeight) * 0.5f;
    params.padX = static_cast<int>(dw - 0.1f + 0.5f);
    params.padY = static_cast<int>(dh - 0.1f + 0.5f);

    return params;
}

bool cudaLetterboxAndPreprocess(
    cudaArray_t srcArray,
    float* dstBuffer,
    int dstWidth,
    int dstHeight,
    cudaStream_t stream
) {
    cudaChannelFormatDesc desc;
    cudaExtent extent;
    cudaError_t err = cudaArrayGetInfo(&desc, &extent, nullptr, srcArray);
    if (err != cudaSuccess) {
        return false;
    }

    int srcWidth = static_cast<int>(extent.width);
    int srcHeight = static_cast<int>(extent.height);

    LetterboxParams params = calculateLetterboxParams(srcWidth, srcHeight, dstWidth, dstHeight);
    return cudaLetterboxAndPreprocessWithParams(srcArray, dstBuffer, params, stream);
}

bool cudaLetterboxAndPreprocessCrop(
    cudaArray_t srcArray,
    float* dstBuffer,
    int fullWidth,
    int fullHeight,
    int cropX,
    int cropY,
    int cropW,
    int cropH,
    int dstWidth,
    int dstHeight,
    cudaStream_t stream
) {
    (void)fullWidth;
    (void)fullHeight;
    if (cropW <= 0 || cropH <= 0 || !dstBuffer) return false;

    LetterboxParams params = calculateLetterboxParams(cropW, cropH, dstWidth, dstHeight);
    return cudaLetterboxAndPreprocessWithParamsCrop(srcArray, dstBuffer, params, cropX, cropY, stream);
}

bool cudaLetterboxAndPreprocessWithParams(
    cudaArray_t srcArray,
    float* dstBuffer,
    const LetterboxParams& params,
    cudaStream_t stream
) {
    return cudaLetterboxAndPreprocessWithParamsCrop(srcArray, dstBuffer, params, 0, 0, stream);
}

bool cudaLetterboxAndPreprocessWithParamsCrop(
    cudaArray_t srcArray,
    float* dstBuffer,
    const LetterboxParams& params,
    int cropX,
    int cropY,
    cudaStream_t stream
) {
    if (!dstBuffer || params.dstWidth <= 0 || params.dstHeight <= 0) return false;

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = srcArray;

    cudaSurfaceObject_t srcSurface = 0;
    cudaError_t err = cudaCreateSurfaceObject(&srcSurface, &resDesc);
    if (err != cudaSuccess) {
        return false;
    }

    int channelSize = params.dstWidth * params.dstHeight;
    float* dstR = dstBuffer;
    float* dstG = dstBuffer + channelSize;
    float* dstB = dstBuffer + channelSize * 2;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (params.dstWidth + blockSize.x - 1) / blockSize.x,
        (params.dstHeight + blockSize.y - 1) / blockSize.y
    );

    letterboxKernelCrop<<<gridSize, blockSize, 0, stream>>>(
        srcSurface,
        dstR, dstG, dstB,
        cropX, cropY,
        params.srcWidth, params.srcHeight,
        params.dstWidth, params.dstHeight,
        params.scale,
        params.padX, params.padY
    );

    err = cudaGetLastError();
    cudaDestroySurfaceObject(srcSurface);

    return (err == cudaSuccess);
}

bool cudaPreprocessBGRA(
    const unsigned char* srcBGRA,
    int srcWidth,
    int srcHeight,
    int srcPitch,
    float* dstBuffer,
    int dstWidth,
    int dstHeight,
    cudaStream_t stream
) {
    if (!srcBGRA || !dstBuffer) return false;

    LetterboxParams params = calculateLetterboxParams(srcWidth, srcHeight, dstWidth, dstHeight);

    int channelSize = dstWidth * dstHeight;
    float* dstR = dstBuffer;
    float* dstG = dstBuffer + channelSize;
    float* dstB = dstBuffer + channelSize * 2;

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (dstWidth + blockSize.x - 1) / blockSize.x,
        (dstHeight + blockSize.y - 1) / blockSize.y
    );

    preprocessBGRAKernel<<<gridSize, blockSize, 0, stream>>>(
        srcBGRA,
        dstR, dstG, dstB,
        srcWidth, srcHeight, srcPitch,
        dstWidth, dstHeight,
        params.scale,
        params.padX, params.padY
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess);
}

bool cudaPreprocessBGRAHost(
    const unsigned char* hostBGRA,
    int srcWidth,
    int srcHeight,
    int srcPitch,
    float* dstBuffer,
    int dstWidth,
    int dstHeight,
    cudaStream_t stream,
    unsigned char** ioDeviceBgra,
    size_t* ioDeviceBgraBytes
) {
    if (!hostBGRA || !dstBuffer || !ioDeviceBgra || !ioDeviceBgraBytes) return false;
    if (srcWidth <= 0 || srcHeight <= 0 || srcPitch < srcWidth * 4) return false;

    size_t needed = static_cast<size_t>(srcPitch) * static_cast<size_t>(srcHeight);
    if (*ioDeviceBgra == nullptr || *ioDeviceBgraBytes < needed) {
        if (*ioDeviceBgra) {
            cudaFree(*ioDeviceBgra);
            *ioDeviceBgra = nullptr;
            *ioDeviceBgraBytes = 0;
        }
        if (cudaMalloc(ioDeviceBgra, needed) != cudaSuccess) {
            *ioDeviceBgra = nullptr;
            *ioDeviceBgraBytes = 0;
            return false;
        }
        *ioDeviceBgraBytes = needed;
    }

    if (cudaMemcpyAsync(*ioDeviceBgra, hostBGRA, needed, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }

    return cudaPreprocessBGRA(
        *ioDeviceBgra, srcWidth, srcHeight, srcPitch,
        dstBuffer, dstWidth, dstHeight, stream
    );
}
