#ifdef _WIN32

#include "MouseController.hpp"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windowsx.h>

MouseController::MouseController()
    : AbstractMouseController()
{
}

MouseController::~MouseController()
{
}

void MouseController::updateConfig(const MouseControllerConfig& config)
{
    AbstractMouseController::updateConfig(config);
    screenWidth_ = config.screenWidth > 0 ? config.screenWidth : GetSystemMetrics(SM_CXSCREEN);
    screenHeight_ = config.screenHeight > 0 ? config.screenHeight : GetSystemMetrics(SM_CYSCREEN);
}

void MouseController::moveMouse(int dx, int dy)
{
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dx = dx;
    input.mi.dy = dy;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.time = 0;
    input.mi.dwExtraInfo = 0;
    SendInput(1, &input, sizeof(INPUT));
}

void MouseController::performClickDown()
{
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));
}

void MouseController::performClickUp()
{
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));
}

bool MouseController::checkFiring()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
}

#endif
