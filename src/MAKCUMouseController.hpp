#ifndef MAKCU_MOUSE_CONTROLLER_HPP
#define MAKCU_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include "AbstractMouseController.hpp"
#include <string>

class MAKCUMouseController : public AbstractMouseController {
private:
    HANDLE hSerial;
    bool serialConnected;
    std::string portName;
    int baudRate;

    bool connectSerial();
    void disconnectSerial();
    bool sendSerialCommand(const std::string& command);

public:
    MAKCUMouseController();
    MAKCUMouseController(const std::string& port, int baud = 115200);
    ~MAKCUMouseController();

    ControllerType getControllerType() const override { return ControllerType::MAKCU; }
    
    bool isConnected() const { return serialConnected; }
    bool testCommunication();

protected:
    void moveMouse(int dx, int dy) override;
    void performClickDown() override;
    void performClickUp() override;
    bool checkFiring() override;
    
    const char* getLogPrefix() const override { return "MAKCU"; }
    
private:
    void move(int dx, int dy);
    void moveTo(int x, int y);
    void clickDown();
    void clickUp();
};

#endif

#endif
