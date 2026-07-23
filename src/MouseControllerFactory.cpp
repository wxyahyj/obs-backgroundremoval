#ifdef _WIN32

#include "MouseControllerFactory.hpp"

std::unique_ptr<MouseControllerInterface> MouseControllerFactory::createController(ControllerType type, const std::string& makcuPort, int makcuBaudRate, int logiDriverType)
{
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
