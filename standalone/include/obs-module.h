#pragma once
/*
 * Standalone stub replacing OBS headers so parent ModelYOLO / Model.h compile
 * without linking libobs. Only the symbols actually referenced by inference
 * sources are provided.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#ifndef LOG_ERROR
#define LOG_ERROR   100
#define LOG_WARNING 200
#define LOG_INFO    300
#define LOG_DEBUG   400
#endif

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(v) ((void)(v))
#endif

/* bfree is used by Model::getModelFilepath — map to free */
static inline void bfree(void *p) { free(p); }

/* Model path lookup is unused by ModelYOLO::loadModel(path); return NULL. */
static inline char *obs_module_file(const char *file)
{
	(void)file;
	return NULL;
}

/* RecoilPatternManager etc. — writable config path under ./config/ */
static inline char *obs_module_config_path(const char *file)
{
	const char *name = file ? file : "config.json";
	size_t n = strlen(name) + 16;
	char *buf = (char *)malloc(n);
	if (!buf)
		return NULL;
	snprintf(buf, n, "config/%s", name);
	return buf;
}

void obs_log(int log_level, const char *format, ...);

/* parent sources sometimes call libobs blog() directly */
#ifndef blog
#define blog obs_log
#endif

#ifdef __cplusplus
}
#endif
