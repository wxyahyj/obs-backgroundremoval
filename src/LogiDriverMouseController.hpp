#ifndef LOGI_DRIVER_MOUSE_CONTROLLER_HPP
#define LOGI_DRIVER_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"

// 罗技G HUB / LGS / 雷蛇 Synapse 驱动控制器
// 通过内核级IOCTL直接与驱动通信，绕过系统鼠标输入
class LogiDriverMouseController : public AbstractMouseController {
private:
    int driverType;         // 0=自动检测, 1=GHUB, 2=LGS, 3=Razer
    bool deviceConnected;

    bool ensureConnected();

public:
    LogiDriverMouseController(int type = 0);
    ~LogiDriverMouseController();

    ControllerType getControllerType() const override { return ControllerType::LogiDriver; }

    bool isConnected() const { return deviceConnected; }

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;

    const char* getLogPrefix() const override { return "LogiDriver"; }
};

#endif

#endif
