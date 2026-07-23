#ifdef _WIN32

#include "NtUserInjectMouseController.hpp"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windows.h>

// 注入鼠标输入结构（根据探测结果，大小为 32 字节）
// MOUSEINPUT 正常 28 字节，这里对齐后是 32 字节
// 注意：参数顺序为 (data, count)，不是 (count, data)
typedef struct INJECT_MOUSE_INPUT {
    LONG dx;
    LONG dy;
    DWORD mouseData;
    DWORD dwFlags;
    DWORD time;
    ULONG_PTR dwExtraInfo;
} INJECT_MOUSE_INPUT;

// 函数指针类型：参数顺序 (data, count)
typedef BOOL(WINAPI* FN_InjectMouseInput)(
    const INJECT_MOUSE_INPUT* data,
    UINT count
);

NtUserInjectMouseController::NtUserInjectMouseController()
    : AbstractMouseController()
    , functionAvailable_(false)
    , hModule_(nullptr)
    , pfnInjectMouseInput_(nullptr)
{
    functionAvailable_ = resolveFunctions();
    if (functionAvailable_) {
        obs_log(LOG_INFO, "[NtUserInjectMouse] InjectMouseInput resolved successfully");
    } else {
        obs_log(LOG_WARNING, "[NtUserInjectMouse] Failed to resolve InjectMouseInput");
    }
}

NtUserInjectMouseController::~NtUserInjectMouseController()
{
    if (hModule_) {
        FreeLibrary(hModule_);
        hModule_ = nullptr;
    }
}

bool NtUserInjectMouseController::resolveFunctions()
{
    // 优先从 win32u.dll 查找
    hModule_ = LoadLibraryA("win32u.dll");
    if (hModule_) {
        pfnInjectMouseInput_ = GetProcAddress(hModule_, "NtUserInjectMouseInput");
        if (pfnInjectMouseInput_) {
            obs_log(LOG_INFO, "[NtUserInjectMouse] Function resolved from win32u.dll (NtUserInjectMouseInput)");
            return true;
        }
    }

    // 回退到 user32.dll
    HMODULE hUser32 = LoadLibraryA("user32.dll");
    if (hUser32) {
        pfnInjectMouseInput_ = GetProcAddress(hUser32, "InjectMouseInput");
        if (pfnInjectMouseInput_) {
            obs_log(LOG_INFO, "[NtUserInjectMouse] Function resolved from user32.dll (InjectMouseInput)");
            return true;
        }
    }

    return false;
}

void NtUserInjectMouseController::moveMouse(int dx, int dy)
{
    if (!functionAvailable_ || !pfnInjectMouseInput_) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        input.mi.dx = dx;
        input.mi.dy = dy;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    INJECT_MOUSE_INPUT data = {};
    data.dx = dx;
    data.dy = dy;
    data.dwFlags = MOUSEEVENTF_MOVE;

    FN_InjectMouseInput pfn = (FN_InjectMouseInput)pfnInjectMouseInput_;
    // 参数顺序：(data, count)
    pfn(&data, 1);
}

void NtUserInjectMouseController::performClickDown()
{
    if (!functionAvailable_ || !pfnInjectMouseInput_) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    INJECT_MOUSE_INPUT data = {};
    data.dwFlags = MOUSEEVENTF_LEFTDOWN;

    FN_InjectMouseInput pfn = (FN_InjectMouseInput)pfnInjectMouseInput_;
    pfn(&data, 1);
}

void NtUserInjectMouseController::performClickUp()
{
    if (!functionAvailable_ || !pfnInjectMouseInput_) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    INJECT_MOUSE_INPUT data = {};
    data.dwFlags = MOUSEEVENTF_LEFTUP;

    FN_InjectMouseInput pfn = (FN_InjectMouseInput)pfnInjectMouseInput_;
    pfn(&data, 1);
}

bool NtUserInjectMouseController::checkFiring()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
}

#endif
