#ifndef MOUSE_CONTROLLER_FACTORY_HPP
#define MOUSE_CONTROLLER_FACTORY_HPP

#ifdef _WIN32

#include <memory>
#include "MouseControllerInterface.hpp"
#include "MouseController.hpp"
#include "MAKCUMouseController.hpp"
#include "LogiDriverMouseController.hpp"
#include "GvInputMouseController.hpp"
#include "NtUserSendInputMouseController.hpp"
#include "NtUserInjectMouseController.hpp"
#include "NtUserInjectPointerController.hpp"

class MouseControllerFactory {
public:
    static std::unique_ptr<MouseControllerInterface> createController(ControllerType type, const std::string& makcuPort = "COM5", int makcuBaudRate = 40000, int logiDriverType = 0);
};

#endif

#endif
