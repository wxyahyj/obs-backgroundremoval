#ifdef _WIN32

#include "MAKCUMouseController.hpp"
#include <obs-module.h>
#include <plugin-support.h>

#define NOMINMAX
#include <windows.h>

MAKCUMouseController::MAKCUMouseController()
    : AbstractMouseController()
    , portName("COM3")
    , baudRate(115200)
{
    for (auto &b : buttonStates_) {
        b.store(false);
    }
}

MAKCUMouseController::MAKCUMouseController(const std::string& port, int baud)
    : AbstractMouseController()
    , portName(port)
    , baudRate(baud)
{
    for (auto &b : buttonStates_) {
        b.store(false);
    }

    try {
        // MAKCU 固件旁路按键监控: 主机的鼠标按键经串口上报到本机 (辅机)
        device_.setMouseButtonCallback([this](makcu::MouseButton button, bool pressed) {
            onButtonCallback(button, pressed);
        });
        device_.enableButtonMonitoring(true);

        if (!device_.connect(portName)) {
            obs_log(LOG_ERROR, "[MAKCU] 连接失败: %s", portName.c_str());
            connected_.store(false);
            return;
        }

        // 波特率: SDK 默认 115200 连接, 再切换到用户配置 (MAKCU 高帧率模式 4000000)
        if (baudRate > 0 && baudRate != 115200) {
            if (!device_.setBaudRate(static_cast<uint32_t>(baudRate), true)) {
                obs_log(LOG_WARNING, "[MAKCU] 切换波特率 %d 失败, 保持 115200", baudRate);
            }
        }
        device_.enableHighPerformanceMode(true);

        connected_.store(true);
        obs_log(LOG_INFO, "[MAKCU] Connected: %s @ %d baud (按键监控已启用)", portName.c_str(), baudRate);
    } catch (const std::exception &e) {
        obs_log(LOG_ERROR, "[MAKCU] 初始化异常: %s", e.what());
        connected_.store(false);
    }
}

MAKCUMouseController::~MAKCUMouseController()
{
    try {
        device_.disconnect();
    } catch (...) {
    }
    connected_.store(false);
}

void MAKCUMouseController::onButtonCallback(makcu::MouseButton button, bool pressed)
{
    int idx = static_cast<int>(button);
    if (idx >= 0 && idx < 5) {
        buttonStates_[idx].store(pressed, std::memory_order_release);
    }
}

// VK 虚拟键码 -> MAKCU 鼠标按键 映射表 (仅鼠标键, 键盘键 MAKCU 不报)
static const int kMakcuVkTable[5] = {VK_LBUTTON, VK_RBUTTON, VK_MBUTTON, VK_XBUTTON1, VK_XBUTTON2};

bool MAKCUMouseController::isPhysicalButtonPressed(int vk)
{
    // 鼠标键: 走 MAKCU 硬件上报 (双机场景主机按键, 单机同样可用)
    for (int i = 0; i < 5; i++) {
        if (vk == kMakcuVkTable[i]) {
            return buttonStates_[i].load(std::memory_order_acquire);
        }
    }
    // 键盘键: MAKCU 不上报, 回退本机键盘状态 (仅单机场景有效; 双机时主机键盘本机查不到)
    if (vk > 0) {
        return (GetAsyncKeyState(vk) & 0x8000) != 0;
    }
    return false;
}

bool MAKCUMouseController::checkFiring()
{
    // 左键射击状态来自 MAKCU 硬件上报 (不是本机 GetAsyncKeyState)
    return buttonStates_[static_cast<int>(makcu::MouseButton::LEFT)].load(std::memory_order_acquire);
}

void MAKCUMouseController::moveMouse(int dx, int dy)
{
    if (!connected_.load(std::memory_order_acquire)) {
        moveFailCount_++;
        if (moveFailCount_ == 1 || moveFailCount_ % 300 == 0) {
            obs_log(LOG_ERROR, "[MAKCU] moveMouse 未连接 (port=%s), 已丢 %d 次移动", portName.c_str(), moveFailCount_);
        }
        return;
    }
    try {
        bool ok = device_.mouseMove(dx, dy);
        moveCount_++;
        if (!ok) {
            moveFailCount_++;
            if (moveFailCount_ % 60 == 1) {
                obs_log(LOG_ERROR, "[MAKCU] mouseMove 发送失败 (dx=%d dy=%d), 累计失败 %d", dx, dy, moveFailCount_);
            }
        } else if (moveCount_ % 120 == 1) {
            obs_log(LOG_INFO, "[MAKCU] move 采样 dx=%d dy=%d (第 %d 次)", dx, dy, moveCount_);
        }
    } catch (const std::exception &e) {
        obs_log(LOG_ERROR, "[MAKCU] mouseMove 异常: %s", e.what());
        connected_.store(false);
    }
}

void MAKCUMouseController::performClickDown()
{
    if (!connected_.load(std::memory_order_acquire)) {
        return;
    }
    try {
        device_.mouseDown(makcu::MouseButton::LEFT);
    } catch (...) {
        connected_.store(false);
    }
}

void MAKCUMouseController::performClickUp()
{
    if (!connected_.load(std::memory_order_acquire)) {
        return;
    }
    try {
        device_.mouseUp(makcu::MouseButton::LEFT);
    } catch (...) {
        connected_.store(false);
    }
}

bool MAKCUMouseController::testCommunication()
{
    if (!connected_.load(std::memory_order_acquire)) {
        if (!device_.connect(portName)) {
            return false;
        }
        connected_.store(true);
    }
    try {
        std::string version = device_.getVersion();
        obs_log(LOG_INFO, "[MAKCU] 版本: %s", version.c_str());
        return !version.empty();
    } catch (...) {
        return false;
    }
}

#endif