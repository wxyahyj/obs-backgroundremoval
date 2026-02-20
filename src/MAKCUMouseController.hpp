#ifndef MAKCU_MOUSE_CONTROLLER_HPP
#define MAKCU_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include <windows.h>
#include <vector>
#include <mutex>
#include <string>
#include "MouseControllerInterface.hpp"

class MAKCUMouseController : public MouseControllerInterface {
public:
    MAKCUMouseController();
    MAKCUMouseController(const std::string& port, int baud);
    ~MAKCUMouseController();

    void updateConfig(const MouseControllerConfig& config) override;
    
    void setDetections(const std::vector<Detection>& detections) override;
    
    void tick() override;
    
    bool isConnected();
    bool testCommunication();

private:
    std::mutex mutex;
    MouseControllerConfig config;
    std::vector<Detection> currentDetections;
    
    HANDLE hSerial;
    bool serialConnected;
    std::string portName;
    int baudRate;
    
    bool isMoving;
    float currentVelocityX;
    float currentVelocityY;
    float currentAccelerationX;
    float currentAccelerationY;
    float previousMoveX;
    float previousMoveY;
    float pidPreviousErrorX;
    float pidPreviousErrorY;
    float filteredDeltaErrorX;
    float filteredDeltaErrorY;
    
    bool connectSerial();
    void disconnectSerial();
    bool sendSerialCommand(const std::string& command);
    
    void move(int dx, int dy);
    void moveTo(int x, int y);
    void click(bool left = true);
    void wheel(int delta);
    
    float calculateDynamicP(float distance);
    Detection* selectTarget();
    POINT convertToScreenCoordinates(const Detection& det);
    void resetPidState();
    void resetMotionState();
};

#endif

#endif
