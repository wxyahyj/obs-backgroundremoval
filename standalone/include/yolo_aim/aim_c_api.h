#pragma once

#include "types.h"

#ifdef _WIN32
#  ifdef YALO_AIM_EXPORTS
#    define YAIM_API __declspec(dllexport)
#  else
#    define YAIM_API __declspec(dllimport)
#  endif
#else
#  define YAIM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* YoloAimHandle;

YAIM_API YoloAimHandle yolo_aim_create(void);
YAIM_API void yolo_aim_destroy(YoloAimHandle h);
YAIM_API int yolo_aim_set_config_json(YoloAimHandle h, const char* json);
YAIM_API int yolo_aim_tick(YoloAimHandle h,
                           const YoloDet* dets, int n,
                           const AimFrameMeta* meta,
                           AimDebug* debug_out);
YAIM_API const char* yolo_aim_last_error(YoloAimHandle h);

#ifdef __cplusplus
}
#endif
