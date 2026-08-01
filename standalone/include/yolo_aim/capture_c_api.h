#pragma once

#include <stdint.h>

#ifdef _WIN32
#  if defined(YCAP_STATIC_LINK)
#    define YCAP_API
#  elif defined(YALO_CAPTURE_EXPORTS)
#    define YCAP_API __declspec(dllexport)
#  else
#    define YCAP_API __declspec(dllimport)
#  endif
#else
#  define YCAP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* YCapHandle;

typedef enum YCapBackend {
    YCAP_GDI = 0,
    YCAP_DXGI = 1,
    YCAP_WGC = 2
} YCapBackend;

YCAP_API YCapHandle ycap_create(YCapBackend backend);
YCAP_API int ycap_init_center(YCapHandle h, int width, int height);
YCAP_API int ycap_init_region(YCapHandle h, int x, int y, int width, int height);
YCAP_API int ycap_set_window(YCapHandle h, void* hwnd);
YCAP_API int ycap_set_region(YCapHandle h, int x, int y, int width, int height);

/* Returns internal BGR buffer (valid until next capture/release). */
YCAP_API const unsigned char* ycap_capture_bgr(YCapHandle h);
YCAP_API int ycap_width(YCapHandle h);
YCAP_API int ycap_height(YCapHandle h);
YCAP_API void ycap_release(YCapHandle h);

#ifdef __cplusplus
}
#endif
