#ifndef NTUSER_INJECT_MOUSE_CONTROLLER_HPP
#define NTUSER_INJECT_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"
#include <string>

// NtUserInjectMouse 注入控制器
// 通过 InjectMouseInput 直接注入鼠标输入
// 与 SendInput 不同路径的系统级注入
class NtUserInjectMouseController : public AbstractMouseController {
private:
    bool functionAvailable_;
    HMODULE hModule_;
    FARPROC pfnInjectMouseInput_;

    bool resolveFunctions();

public:
    NtUserInjectMouseController();
    ~NtUserInjectMouseController();

    ControllerType getControllerType() const override { return ControllerType::NtUserInjectMouse; }
    bool isAvailable() const { return functionAvailable_; }

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    const char* getLogPrefix() const override { return "NtUserInjectMouse"; }
};

#endif
#endif
