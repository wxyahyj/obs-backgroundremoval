#ifdef _WIN32

#include "LogiDriverMouseController.hpp"
#include "logi_driver.h"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windows.h>

LogiDriverMouseController::LogiDriverMouseController(int type)
    : AbstractMouseController()
    , driverType(type)
    , deviceConnected(false)
{
    logi_driver_init();
    if (device_open2(type)) {
        deviceConnected = true;
        obs_log(LOG_INFO, "[LogiDriver] 连接成功，驱动类型: %d", get_driver_type());
    } else {
        obs_log(LOG_WARNING, "[LogiDriver] 连接失败，驱动子类型: %d", type);
    }
}

LogiDriverMouseController::~LogiDriverMouseController()
{
    device_close();
    logi_driver_cleanup();
}

bool LogiDriverMouseController::ensureConnected()
{
    if (deviceConnected) return true;
    if (device_open2(driverType)) {
        deviceConnected = true;
        obs_log(LOG_INFO, "[LogiDriver] 重连成功，驱动类型: %d", get_driver_type());
        return true;
    }
    return false;
}

void LogiDriverMouseController::moveMouse(int dx, int dy)
{
    if (!ensureConnected()) return;
    if (!moveR(dx, dy)) {
        deviceConnected = false;
    }
}

void LogiDriverMouseController::performClickDown()
{
    if (!ensureConnected()) return;
    if (!mouse_down(1)) {
        deviceConnected = false;
    }
}

void LogiDriverMouseController::performClickUp()
{
    if (!ensureConnected()) return;
    if (!mouse_up(1)) {
        deviceConnected = false;
    }
}

bool LogiDriverMouseController::checkFiring()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
}

#endif
