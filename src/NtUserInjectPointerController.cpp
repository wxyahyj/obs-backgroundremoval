#ifdef _WIN32

#include "NtUserInjectPointerController.hpp"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windows.h>
#include <winuser.h>

// 函数指针类型
typedef HSYNTHETICPOINTERDEVICE(WINAPI* FN_CreateSyntheticPointerDevice)(
    POINTER_INPUT_TYPE pointerType,
    ULONG maxCount,
    POINTER_FEEDBACK_MODE mode
);

typedef BOOL(WINAPI* FN_InjectSyntheticPointerInput)(
    HSYNTHETICPOINTERDEVICE device,
    const POINTER_TYPE_INFO* pointerInfo,
    UINT32 count
);

typedef VOID(WINAPI* FN_DestroySyntheticPointerDevice)(
    HSYNTHETICPOINTERDEVICE device
);

NtUserInjectPointerController::NtUserInjectPointerController()
    : AbstractMouseController()
    , functionAvailable_(false)
    , hUser32_(nullptr)
    , device_(nullptr)
    , pfnCreateSyntheticPointerDevice_(nullptr)
    , pfnInjectSyntheticPointerInput_(nullptr)
    , pfnDestroySyntheticPointerDevice_(nullptr)
{
    if (resolveFunctions()) {
        functionAvailable_ = initDevice();
    }
}

NtUserInjectPointerController::~NtUserInjectPointerController()
{
    cleanupDevice();
    if (hUser32_) {
        FreeLibrary(hUser32_);
        hUser32_ = nullptr;
    }
}

bool NtUserInjectPointerController::resolveFunctions()
{
    hUser32_ = LoadLibraryA("user32.dll");
    if (!hUser32_) {
        obs_log(LOG_WARNING, "[NtUserInjectPtr] Failed to load user32.dll");
        return false;
    }

    pfnCreateSyntheticPointerDevice_ = GetProcAddress(hUser32_, "CreateSyntheticPointerDevice");
    pfnInjectSyntheticPointerInput_ = GetProcAddress(hUser32_, "InjectSyntheticPointerInput");
    pfnDestroySyntheticPointerDevice_ = GetProcAddress(hUser32_, "DestroySyntheticPointerDevice");

    if (!pfnCreateSyntheticPointerDevice_ || !pfnInjectSyntheticPointerInput_ || !pfnDestroySyntheticPointerDevice_) {
        obs_log(LOG_WARNING, "[NtUserInjectPtr] Failed to resolve synthetic pointer API (requires Win10 RS5+)");
        return false;
    }

    obs_log(LOG_INFO, "[NtUserInjectPtr] Synthetic pointer API resolved successfully");
    return true;
}

bool NtUserInjectPointerController::initDevice()
{
    if (!pfnCreateSyntheticPointerDevice_) return false;

    FN_CreateSyntheticPointerDevice pfn = (FN_CreateSyntheticPointerDevice)pfnCreateSyntheticPointerDevice_;
    device_ = pfn(PT_TOUCH, 1, POINTER_FEEDBACK_DEFAULT);

    if (!device_) {
        DWORD err = GetLastError();
        obs_log(LOG_WARNING, "[NtUserInjectPtr] CreateSyntheticPointerDevice failed: %lu", err);
        return false;
    }

    obs_log(LOG_INFO, "[NtUserInjectPtr] Synthetic touch device created successfully");
    return true;
}

void NtUserInjectPointerController::cleanupDevice()
{
    if (device_ && pfnDestroySyntheticPointerDevice_) {
        FN_DestroySyntheticPointerDevice pfn = (FN_DestroySyntheticPointerDevice)pfnDestroySyntheticPointerDevice_;
        pfn(device_);
        device_ = nullptr;
    }
}

void NtUserInjectPointerController::injectTouch(int x, int y, DWORD pointerFlags)
{
    if (!device_ || !pfnInjectSyntheticPointerInput_) return;

    POINTER_TYPE_INFO pti = {};
    pti.type = PT_TOUCH;
    pti.touchInfo.pointerInfo.pointerType = PT_TOUCH;
    pti.touchInfo.pointerInfo.pointerId = 0;
    pti.touchInfo.pointerInfo.pointerFlags = (POINTER_FLAGS)pointerFlags;
    pti.touchInfo.pointerInfo.ptPixelLocation.x = x;
    pti.touchInfo.pointerInfo.ptPixelLocation.y = y;
    pti.touchInfo.pointerInfo.ptPixelLocationRaw.x = x;
    pti.touchInfo.pointerInfo.ptPixelLocationRaw.y = y;
    pti.touchInfo.touchFlags = TOUCH_FLAG_NONE;
    pti.touchInfo.touchMask = TOUCH_MASK_NONE;
    pti.touchInfo.rcContact.left = x - 2;
    pti.touchInfo.rcContact.top = y - 2;
    pti.touchInfo.rcContact.right = x + 2;
    pti.touchInfo.rcContact.bottom = y + 2;
    pti.touchInfo.rcContactRaw = pti.touchInfo.rcContact;

    FN_InjectSyntheticPointerInput pfn = (FN_InjectSyntheticPointerInput)pfnInjectSyntheticPointerInput_;
    pfn(device_, &pti, 1);
}

void NtUserInjectPointerController::moveMouse(int dx, int dy)
{
    if (!isAvailable()) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        input.mi.dx = dx;
        input.mi.dy = dy;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    // 获取当前鼠标位置，计算新位置
    POINT curPos;
    GetCursorPos(&curPos);
    int newX = curPos.x + dx;
    int newY = curPos.y + dy;

    // 限制在屏幕范围内
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    if (newX < 0) newX = 0;
    if (newY < 0) newY = 0;
    if (newX >= screenW) newX = screenW - 1;
    if (newY >= screenH) newY = screenH - 1;

    // 触摸更新
    injectTouch(newX, newY, POINTER_FLAG_INRANGE | POINTER_FLAG_UPDATE);

    // 同时移动真实鼠标光标
    SetCursorPos(newX, newY);
}

void NtUserInjectPointerController::performClickDown()
{
    if (!isAvailable()) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    POINT curPos;
    GetCursorPos(&curPos);
    injectTouch(curPos.x, curPos.y, POINTER_FLAG_INRANGE | POINTER_FLAG_INCONTACT | POINTER_FLAG_DOWN);
}

void NtUserInjectPointerController::performClickUp()
{
    if (!isAvailable()) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    POINT curPos;
    GetCursorPos(&curPos);
    injectTouch(curPos.x, curPos.y, POINTER_FLAG_INRANGE | POINTER_FLAG_UP);
}

bool NtUserInjectPointerController::checkFiring()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
}

#endif
