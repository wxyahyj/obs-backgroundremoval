#include "CudaPreprocessor.cuh"
#include <device_launch_parameters.h>
#include <cuda_surface_types.h>
#include <surface_functions.h>

__constant__ float cPadValue[4] = {114.0f / 255.0f, 114.0f / 255.0f, 114.0f / 255.0f, 1.0f};

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
    LetterboxParams params;
    params.srcWidth = srcWidth;
    params.srcHeight = srcHeight;
    params.dstWidth = dstWidth;
    params.dstHeight = dstHeight;
    
    float scaleX = static_cast<float>(dstWidth) / srcWidth;
    float scaleY = static_cast<float>(dstHeight) / srcHeight;
    params.scale = (scaleX < scaleY) ? scaleX : scaleY;
    
    int newWidth = static_cast<int>(srcWidth * params.scale);
    int newHeight = static_cast<int>(srcHeight * params.scale);
    
    params.padX = (dstWidth - newWidth) / 2;
    params.padY = (dstHeight - newHeight) / 2;
    
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

bool cudaLetterboxAndPreprocessWithParams(
    cudaArray_t srcArray,
    float* dstBuffer,
    const LetterboxParams& params,
    cudaStream_t stream
) {
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
    
    letterboxKernel<<<gridSize, blockSize, 0, stream>>>(
        srcSurface,
        dstR, dstG, dstB,
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
