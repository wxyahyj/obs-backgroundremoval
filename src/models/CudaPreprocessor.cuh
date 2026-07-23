#ifndef CUDA_PREPROCESSOR_CUH
#define CUDA_PREPROCESSOR_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

struct LetterboxParams {
    int srcWidth;
    int srcHeight;
    int dstWidth;
    int dstHeight;
    float scale;
    int padX;
    int padY;
};

// Full-texture letterbox (crop = full surface).
bool cudaLetterboxAndPreprocess(
    cudaArray_t srcArray,
    float* dstBuffer,
    int dstWidth,
    int dstHeight,
    cudaStream_t stream
);

// Letterbox a crop rectangle inside a full surface (ROI support).
// cropX/cropY/cropW/cropH are in full-texture pixel coordinates.
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
);

bool cudaLetterboxAndPreprocessWithParams(
    cudaArray_t srcArray,
    float* dstBuffer,
    const LetterboxParams& params,
    cudaStream_t stream
);

// Same as WithParams but samples from (cropX + srcX, cropY + srcY) on the surface.
bool cudaLetterboxAndPreprocessWithParamsCrop(
    cudaArray_t srcArray,
    float* dstBuffer,
    const LetterboxParams& params,
    int cropX,
    int cropY,
    cudaStream_t stream
);

// Device-pointer BGRA letterbox (srcBGRA must already be on device).
bool cudaPreprocessBGRA(
    const unsigned char* srcBGRA,
    int srcWidth,
    int srcHeight,
    int srcPitch,
    float* dstBuffer,
    int dstWidth,
    int dstHeight,
    cudaStream_t stream
);

// Host BGRA → H2D → letterbox → device float CHW. Allocates/reuses temp device BGRA.
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
);

LetterboxParams calculateLetterboxParams(
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight
);

#ifdef __cplusplus
}
#endif

#endif
