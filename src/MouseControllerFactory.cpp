#ifdef _WIN32

#include "MouseControllerFactory.hpp"

std::unique_ptr<MouseControllerInterface> MouseControllerFactory::createController(ControllerType type, const std::string& makcuPort, int makcuBaudRate)
{
    switch (type) {
        case ControllerType::MAKCU:
            return std::make_unique<MAKCUMouseController>(makcuPort, makcuBaudRate);
        case ControllerType::WindowsAPI:
        default:
            return std::make_unique<MouseController>();
    }
}

#endif
