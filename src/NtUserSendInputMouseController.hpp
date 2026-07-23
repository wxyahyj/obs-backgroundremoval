#ifndef NTUSER_SENDINPUT_MOUSE_CONTROLLER_HPP
#define NTUSER_SENDINPUT_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"
#include <string>

// NtUserSendInput 直接调用控制器
// 通过动态获取 win32u.dll / user32.dll 中的 NtUserSendInput 函数地址，
// 绕过 user32!SendInput 封装层直接调用 Native API
class NtUserSendInputMouseController : public AbstractMouseController {
private:
    bool initialized_;
    bool functionAvailable_;
    FARPROC pfnNtUserSendInput_;

    bool resolveFunction();

public:
    NtUserSendInputMouseController();
    ~NtUserSendInputMouseController();

    ControllerType getControllerType() const override { return ControllerType::NtUserSendInput; }
    bool isAvailable() const { return functionAvailable_; }

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    const char* getLogPrefix() const override { return "NtUserSI"; }
};

#endif
#endif
