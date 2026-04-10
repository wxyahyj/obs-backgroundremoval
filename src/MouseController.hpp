#ifndef MOUSE_CONTROLLER_HPP
#define MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"

class MouseController : public AbstractMouseController {
public:
    MouseController();
    ~MouseController();

    ControllerType getControllerType() const override { return ControllerType::WindowsAPI; }
    void updateConfig(const MouseControllerConfig& config) override;

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    
    const char* getLogPrefix() const override { return ""; }

private:
    bool useAbsoluteMove_ = false;
    int screenWidth_ = 1920;
    int screenHeight_ = 1080;
};

#endif

#endif
