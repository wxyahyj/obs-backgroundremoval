#ifdef _WIN32

#include "MouseControllerFactory.hpp"

std::unique_ptr<MouseControllerInterface> MouseControllerFactory::createController(ControllerType type, const std::string& makcuPort, int makcuBaudRate, int logiDriverType)
{
    switch (type) {
        case ControllerType::MAKCU:
            return std::make_unique<MAKCUMouseController>(makcuPort, makcuBaudRate);
        case ControllerType::LogiDriver:
            return std::make_unique<LogiDriverMouseController>(logiDriverType);
        case ControllerType::WindowsAPI:
        default:
            return std::make_unique<MouseController>();
    }
}

#endif
