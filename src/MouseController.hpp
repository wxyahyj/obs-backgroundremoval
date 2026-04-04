#ifndef MOUSE_CONTROLLER_HPP
#define MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"

class MouseController : public AbstractMouseController {
public:
    MouseController();
    ~MouseController();

    ControllerType getControllerType() const override { return ControllerType::WindowsAPI; }

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    
    const char* getLogPrefix() const override { return ""; }
};

#endif

#endif
