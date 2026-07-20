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
        case ControllerType::LogiDriver:
            return std::make_unique<LogiDriverMouseController>(logiDriverType);
        case ControllerType::WindowsAPI:
        default:
            return std::make_unique<MouseController>();
    }
}

#endif
