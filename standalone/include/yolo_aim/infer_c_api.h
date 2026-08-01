#pragma once

#include "types.h"

#ifdef _WIN32
#  ifdef YALO_INFER_EXPORTS
#    define YINF_API __declspec(dllexport)
#  else
#    define YINF_API __declspec(dllimport)
#  endif
#else
#  define YINF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* YoloInferHandle;

/* device: "cpu" | "dml" | "cuda" | "tensorrt" */
YINF_API YoloInferHandle yolo_infer_create(const char* model_path, const char* device);
YINF_API void yolo_infer_destroy(YoloInferHandle h);
YINF_API int yolo_infer_set_thresholds(YoloInferHandle h, float conf, float nms);
YINF_API int yolo_infer_bgr(YoloInferHandle h,
                            const unsigned char* bgr, int w, int h, int stride,
                            YoloDet* out, int max_out);
YINF_API const char* yolo_infer_last_error(YoloInferHandle h);

#ifdef __cplusplus
}
#endif
