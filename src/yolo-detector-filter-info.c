#include <obs-module.h>
#include "yolo-detector-filter.h"

struct obs_source_info yolo_detector_filter_info = {
	.id = "yolo-detector-filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = yolo_detector_filter_getname,
	.create = yolo_detector_filter_create,
	.destroy = yolo_detector_filter_destroy,
	.get_defaults = yolo_detector_filter_defaults,
	.get_properties = yolo_detector_filter_properties,
	.update = yolo_detector_filter_update,
	.activate = yolo_detector_filter_activate,
	.deactivate = yolo_detector_filter_deactivate,
	.video_tick = yolo_detector_filter_video_tick,
	.video_render = yolo_detector_filter_video_render,
};
