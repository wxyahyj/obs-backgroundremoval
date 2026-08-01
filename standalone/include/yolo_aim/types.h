#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct YoloDet {
    int32_t class_id;
    float confidence;
    float x, y, w, h; /* normalized 0..1 */
    float cx, cy;
    int32_t track_id;
} YoloDet;

typedef struct AimFrameMeta {
    int32_t origin_x;
    int32_t origin_y;
    int32_t frame_w;
    int32_t frame_h;
    int32_t screen_w;
    int32_t screen_h;
    int64_t pts_ns;
    int32_t hotkey_down; /* 1 if aim hotkey held */
} AimFrameMeta;

typedef struct AimDebug {
    float error_x, error_y;
    float out_x, out_y;
    float kp, ki, kd;
    int32_t det_count;
    int32_t target_track_id;
    double capture_fps;
    double infer_ms;
} AimDebug;

typedef struct EngineSnapshot {
    int32_t running;
    int32_t capture_ok;
    int32_t infer_ok;
    int32_t aim_ok;
    int32_t aim_hotkey_down;
    int32_t aim_moved;
    double capture_fps;
    double infer_fps;
    double infer_ms;
    int32_t last_det_count;
    float aim_err_x;
    float aim_err_y;
    float aim_out_x;
    float aim_out_y;
    char backend[32];
    char last_error[256];
    /* runtime truth (updated by engine thread) */
    char capture_mode[16];   /* center | region */
    char device_req[16];     /* configured device */
    char device_act[16];     /* actual after load/fallback */
    int32_t capture_w;
    int32_t capture_h;
    int32_t region_x;
    int32_t region_y;
    int32_t origin_x;
    int32_t origin_y;
    char model_path[512];
} EngineSnapshot;

#ifdef __cplusplus
}
#endif
