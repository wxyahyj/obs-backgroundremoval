#ifndef MOUSE_CONTROLLER_FACTORY_HPP
#define MOUSE_CONTROLLER_FACTORY_HPP

#ifdef _WIN32

#include <memory>
#include "MouseControllerInterface.hpp"
#include "MouseController.hpp"
#include "MAKCUMouseController.hpp"

class MouseControllerFactory {
public:
    static std::unique_ptr<MouseControllerInterface> createController(ControllerType type, const std::string& makcuPort = "COM5", int makcuBaudRate = 40000);
};

#endif

#endif
