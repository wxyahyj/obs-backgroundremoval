#include <obs-module.h>
#include "yolo-filter.h"

static const char *yolo_filter_get_name(void *type_data)
{
    UNUSED_PARAMETER(type_data);
    return obs_module_text("YoloDaWang");
}

static void *yolo_filter_create(obs_data_t *settings, obs_source_t *source)
{
    return yolo_filter_create(settings, source);
}

static void yolo_filter_destroy(void *data)
{
    yolo_filter_destroy(data);
}

static void yolo_filter_get_defaults(obs_data_t *settings)
{
    yolo_filter_defaults(settings);
}

static obs_properties_t *yolo_filter_get_properties(void *data)
{
    return yolo_filter_properties(data);
}

static void yolo_filter_update(void *data, obs_data_t *settings)
{
    yolo_filter_update(data, settings);
}

static void yolo_filter_activate(void *data)
{
    yolo_filter_activate(data);
}

static void yolo_filter_deactivate(void *data)
{
    yolo_filter_deactivate(data);
}

static void yolo_filter_video_tick(void *data, float seconds)
{
    yolo_filter_video_tick(data, seconds);
}

static void yolo_filter_video_render(void *data, gs_effect_t *effect)
{
    yolo_filter_video_render(data, effect);
}

struct obs_source_info yolo_filter_info = {
    .id = "yolo_da_wang_filter",
    .type = OBS_SOURCE_TYPE_FILTER,
    .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC,
    .get_name = yolo_filter_get_name,
    .create = yolo_filter_create,
    .destroy = yolo_filter_destroy,
    .get_defaults = yolo_filter_get_defaults,
    .get_properties = yolo_filter_get_properties,
    .update = yolo_filter_update,
    .activate = yolo_filter_activate,
    .deactivate = yolo_filter_deactivate,
    .video_tick = yolo_filter_video_tick,
    .video_render = yolo_filter_video_render,
};