#include <obs-module.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *yolo_filter_getname(void *unused);
void *yolo_filter_create(obs_data_t *settings, obs_source_t *source);
void yolo_filter_destroy(void *data);
void yolo_filter_defaults(obs_data_t *settings);
obs_properties_t *yolo_filter_properties(void *data);
void yolo_filter_update(void *data, obs_data_t *settings);
void yolo_filter_activate(void *data);
void yolo_filter_deactivate(void *data);
void yolo_filter_video_tick(void *data, float seconds);
void yolo_filter_video_render(void *data, gs_effect_t *_effect);

#ifdef __cplusplus
}
#endif