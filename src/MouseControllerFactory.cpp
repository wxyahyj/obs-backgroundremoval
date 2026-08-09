#ifdef _WIN32

#include "MouseControllerFactory.hpp"
#include "hide/anticeat.h"
#include <obs-module.h>
#include <plugin-support.h>

std::unique_ptr<MouseControllerInterface> MouseControllerFactory::createController(ControllerType type, const std::string& makcuPort, int makcuBaudRate, int logiDriverType)
{
    obs_log(LOG_INFO, "[ControllerFactory] create type=%d port=%s baud=%d logiType=%d",
            (int)type, makcuPort.c_str(), makcuBaudRate, logiDriverType);

    /* 顶级反作弊环境 (EAC/BattlEye/Vanguard 等) 下 LogiDriver 的驱动 IOCTL
     * 流量 + syscall 特征会被运行时扫描, 自动降级到更隐蔽/物理的后端。 */
    if (type == ControllerType::LogiDriver && AntiCheat::detectTop()) {
        obs_log(LOG_WARNING, "[ControllerFactory] 检测到顶级反作弊环境, LogiDriver 自动降级");
        if (!makcuPort.empty()) {
            obs_log(LOG_INFO, "[ControllerFactory] 降级到 MAKCU (硬件串口, 物理隔离)");
            return std::make_unique<MAKCUMouseController>(makcuPort, makcuBaudRate);
        }
        obs_log(LOG_INFO, "[ControllerFactory] 降级到 GvInput (HID 虚拟设备)");
        return std::make_unique<GvInputMouseController>();
    }

    switch (type) {
        case ControllerType::MAKCU:
            return std::make_unique<MAKCUMouseController>(makcuPort, makcuBaudRate);
        case ControllerType::GvInput:
            return std::make_unique<GvInputMouseController>();
        case ControllerType::TencInput:
            return std::make_unique<TencInputMouseController>();
        case ControllerType::NtUserSendInput:
            return std::make_unique<NtUserSendInputMouseController>();
        case ControllerType::NtUserInjectMouse:
            return std::make_unique<NtUserInjectMouseController>();
        case ControllerType::NtUserInjectPointer:
            return std::make_unique<NtUserInjectPointerController>();
        case ControllerType::LogiDriver:
            return std::make_unique<LogiDriverMouseController>(logiDriverType);
        case ControllerType::WindowsAPI:
        default:
            return std::make_unique<MouseController>();
    }
}

#endif
