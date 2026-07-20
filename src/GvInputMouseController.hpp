#ifndef GVINPUT_MOUSE_CONTROLLER_HPP
#define GVINPUT_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"
#include <string>

// UU Remote gvInput HID driver controller
class GvInputMouseController : public AbstractMouseController {
private:
    HANDLE hDevice;
    bool deviceConnected;
    std::wstring devicePath;

    bool openDevice();
    void closeDevice();

public:
    GvInputMouseController();
    ~GvInputMouseController();

    ControllerType getControllerType() const override { return ControllerType::GvInput; }
    bool isConnected() const { return deviceConnected; }

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    const char* getLogPrefix() const override { return "GvInput"; }
};

// Tencent MyAppsHidBus HID controller (CyberRemote / TX Meeting / TX XianFeng)
class TencInputMouseController : public AbstractMouseController {
private:
    HANDLE hDevice;
    bool deviceConnected;
    std::wstring devicePath;
    int reportSize;
    WORD reportMagic;

    bool openDevice();
    void closeDevice();

public:
    TencInputMouseController();
    ~TencInputMouseController();

    ControllerType getControllerType() const override { return ControllerType::TencInput; }
    bool isConnected() const { return deviceConnected; }

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    const char* getLogPrefix() const override { return "TencInput"; }
};

#endif
#endif