#include "obs-module.h"
#include "plugin-support.h"
#include "util/Log.hpp"

#include <cstdio>

const char *PLUGIN_NAME = "yolo-aim-standalone";
const char *PLUGIN_VERSION = "0.1.0";

// 主仓 OBS API 日志 stub → standalone 日志通道(文件 + stderr)
void obs_log(int log_level, const char *format, ...)
{
	const char *lvl = "INFO";
	if (log_level <= LOG_ERROR)
		lvl = "ERROR";
	else if (log_level <= LOG_WARNING)
		lvl = "WARN";
	else if (log_level >= LOG_DEBUG)
		lvl = "DEBUG";

	va_list args;
	va_start(args, format);
	ya::util::log_write(lvl, format, args);
	va_end(args);
}
