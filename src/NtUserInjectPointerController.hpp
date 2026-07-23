#ifndef NTUSER_INJECT_POINTER_CONTROLLER_HPP
#define NTUSER_INJECT_POINTER_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"
#include <string>
#include <windows.h>

// NtUserInjectPointer 合成指针设备注入控制器
// 使用 CreateSyntheticPointerDevice 创建合成触摸设备
// 通过 InjectSyntheticPointerInput 注入触摸输入来模拟鼠标移动
// 完全文档化的 Win10 RS5+ API，走独立的指针输入栈
class NtUserInjectPointerController : public AbstractMouseController {
private:
    bool functionAvailable_;
    HMODULE hUser32_;
    HSYNTHETICPOINTERDEVICE device_;

    FARPROC pfnCreateSyntheticPointerDevice_;
    FARPROC pfnInjectSyntheticPointerInput_;
    FARPROC pfnDestroySyntheticPointerDevice_;

    bool resolveFunctions();
    bool initDevice();
    void cleanupDevice();
    void injectTouch(int x, int y, DWORD flags);

public:
    NtUserInjectPointerController();
    ~NtUserInjectPointerController();

    ControllerType getControllerType() const override { return ControllerType::NtUserInjectPointer; }
    bool isAvailable() const { return functionAvailable_ && device_ != nullptr; }

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    const char* getLogPrefix() const override { return "NtUserInjectPtr"; }
};

#endif
#endif
