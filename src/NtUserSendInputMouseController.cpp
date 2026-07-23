#ifdef _WIN32

#include "NtUserSendInputMouseController.hpp"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windows.h>

// NtUserSendInput 函数指针类型定义
// 注意：NtUserSendInput 的签名与 SendInput 相同
typedef UINT(WINAPI* FN_NtUserSendInput)(UINT cInputs, LPINPUT pInputs, int cbSize);

NtUserSendInputMouseController::NtUserSendInputMouseController()
    : AbstractMouseController()
    , initialized_(false)
    , functionAvailable_(false)
    , pfnNtUserSendInput_(nullptr)
{
    if (resolveFunction()) {
        functionAvailable_ = true;
        obs_log(LOG_INFO, "[NtUserSI] NtUserSendInput resolved successfully");
    } else {
        obs_log(LOG_WARNING, "[NtUserSI] Failed to resolve NtUserSendInput");
    }
    initialized_ = true;
}

NtUserSendInputMouseController::~NtUserSendInputMouseController()
{
}

bool NtUserSendInputMouseController::resolveFunction()
{
    // 优先从 win32u.dll 查找（Windows 10+ 的用户态 win32k 前导库）
    HMODULE hWin32u = LoadLibraryA("win32u.dll");
    if (hWin32u) {
        pfnNtUserSendInput_ = GetProcAddress(hWin32u, "NtUserSendInput");
        if (pfnNtUserSendInput_) {
            obs_log(LOG_INFO, "[NtUserSI] Resolved from win32u.dll");
            return true;
        }
    }

    // 回退到 user32.dll
    HMODULE hUser32 = LoadLibraryA("user32.dll");
    if (hUser32) {
        pfnNtUserSendInput_ = GetProcAddress(hUser32, "NtUserSendInput");
        if (pfnNtUserSendInput_) {
            obs_log(LOG_INFO, "[NtUserSI] Resolved from user32.dll");
            return true;
        }
    }

    // 如果两个地方都找不到，说明系统不支持直接导出
    // 可以尝试通过 SSDT 号直接系统调用，但不同 Windows 版本号不同，不稳定
    // 这里只使用导出函数的方式
    return false;
}

void NtUserSendInputMouseController::moveMouse(int dx, int dy)
{
    if (!functionAvailable_ || !pfnNtUserSendInput_) {
        // 回退到 SendInput
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        input.mi.dx = dx;
        input.mi.dy = dy;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.dx = dx;
    input.mi.dy = dy;

    FN_NtUserSendInput pfn = (FN_NtUserSendInput)pfnNtUserSendInput_;
    UINT result = pfn(1, &input, sizeof(INPUT));
    if (result == 0) {
        DWORD err = GetLastError();
        if (err != 0) {
            obs_log(LOG_WARNING, "[NtUserSI] NtUserSendInput move failed: %lu", err);
        }
    }
}

void NtUserSendInputMouseController::performClickDown()
{
    if (!functionAvailable_ || !pfnNtUserSendInput_) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;

    FN_NtUserSendInput pfn = (FN_NtUserSendInput)pfnNtUserSendInput_;
    pfn(1, &input, sizeof(INPUT));
}

void NtUserSendInputMouseController::performClickUp()
{
    if (!functionAvailable_ || !pfnNtUserSendInput_) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
        return;
    }

    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;

    FN_NtUserSendInput pfn = (FN_NtUserSendInput)pfnNtUserSendInput_;
    pfn(1, &input, sizeof(INPUT));
}

bool NtUserSendInputMouseController::checkFiring()
{
    return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
}

#endif
