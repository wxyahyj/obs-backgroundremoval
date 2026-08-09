#ifndef MAKCU_MOUSE_CONTROLLER_HPP
#define MAKCU_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"
#include <atomic>
#include <string>

#include "makcu/include/makcu.h"

/**
 * MAKCU 硬件鼠标控制器 (SDK 版).
 *
 * 双机拓扑: MAKCU 鼠标直通孔插主机 (玩家操作), 电脑口插辅机 (OBS+插件).
 * 固件旁路监控主机鼠标按键并经串口上报, 辅机 SDK 收到按键回调
 * (enableButtonMonitoring + setMouseButtonCallback), 热键不再依赖本机 GetAsyncKeyState.
 */
class MAKCUMouseController : public AbstractMouseController {
private:
    makcu::Device device_;
    std::atomic<bool> connected_{false};
    std::string portName;
    int baudRate;

    // 主机鼠标按键状态: LEFT RIGHT MIDDLE SIDE1 SIDE2 (SDK 回调更新)
    std::atomic<bool> buttonStates_[5];
    int moveCount_ = 0;     // 成功发送计数 (诊断日志采样)
    int moveFailCount_ = 0; // 发送失败/未连接计数

    void onButtonCallback(makcu::MouseButton button, bool pressed);

public:
    MAKCUMouseController();
    MAKCUMouseController(const std::string& port, int baud = 115200);
    ~MAKCUMouseController();

    ControllerType getControllerType() const override { return ControllerType::MAKCU; }

    bool isConnected() const { return connected_.load(std::memory_order_acquire); }
    bool testCommunication();

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    bool isPhysicalButtonPressed(int vk) override;

    const char* getLogPrefix() const override { return "MAKCU"; }
};

#endif

#endif
