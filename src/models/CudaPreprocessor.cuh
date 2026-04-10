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

bool cudaLetterboxAndPreprocess(
    cudaArray_t srcArray,
    float* dstBuffer,
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
