/*
OBS YOLO Detector Filter Plugin
Based on OBS Background Removal Filter Plugin
Copyright (C) 2021 Roy Shilkrot roy.shil@gmail.com

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>

#include "plugin-support.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "zh-CN")

MODULE_EXPORT const char *obs_module_description(void)
{
	return obs_module_text("YOLODetectorPlugin");
}

extern struct obs_source_info yolo_detector_filter_info;

#ifdef ENABLE_QT
#include <obs-frontend-api.h>

#ifdef __cplusplus
extern "C" {
#endif

void YoloAimSettingsDialog_Show();

#ifdef __cplusplus
}
#endif

static void on_tools_menu_clicked(void *data)
{
	UNUSED_PARAMETER(data);
	YoloAimSettingsDialog_Show();
}

#endif

bool obs_module_load(void)
{
	obs_register_source(&yolo_detector_filter_info);
	
#ifdef ENABLE_QT
	obs_frontend_add_tools_menu_item(obs_module_text("YOLO自瞄设置"), on_tools_menu_clicked, NULL);
#endif
	
	obs_log(LOG_INFO, "YOLO Detector Plugin loaded successfully (version %s)", PLUGIN_VERSION);

	return true;
}

void obs_module_unload()
{
	obs_log(LOG_INFO, "YOLO Detector Plugin unloaded");
}
