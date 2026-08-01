#include "obs-module.h"
#include "plugin-support.h"

#include <cstdio>

const char *PLUGIN_NAME = "yolo-aim-standalone";
const char *PLUGIN_VERSION = "0.1.0";

void obs_log(int log_level, const char *format, ...)
{
	const char *lvl = "INFO";
	if (log_level <= LOG_ERROR)
		lvl = "ERROR";
	else if (log_level <= LOG_WARNING)
		lvl = "WARN";
	else if (log_level >= LOG_DEBUG)
		lvl = "DEBUG";

	std::fprintf(stderr, "[%s] ", lvl);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fprintf(stderr, "\n");
}
