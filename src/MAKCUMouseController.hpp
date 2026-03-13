#ifndef MAKCU_MOUSE_CONTROLLER_HPP
#define MAKCU_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include <windows.h>
#include <vector>
#include <mutex>
#include <string>
#include <chrono>
#include <random>
#include "MouseControllerInterface.hpp"
#include "DerivativePredictor.hpp"

class MAKCUMouseController : public MouseControllerInterface {
public:
    MAKCUMouseController();
    MAKCUMouseController(const std::string& port, int baud);
    ~MAKCUMouseController();

    void updateConfig(const MouseControllerConfig& config) override;
    
    void setDetections(const std::vector<Detection>& detections) override;

    void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) override;
    
    void tick() override;
    
    void setCurrentWeapon(const std::string& weaponName) override;
    std::string getCurrentWeapon() const override;
    
    ControllerType getControllerType() const override { return ControllerType::MAKCU; }
    
    bool isConnected();
    bool testCommunication();

private:
    mutable std::mutex mutex;
    MouseControllerConfig config;
    std::vector<Detection> currentDetections;
    
    int cachedScreenWidth;
    int cachedScreenHeight;
    
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
    float previousErrorX;
    float previousErrorY;
    int errorSignChangeCount;
    std::chrono::steady_clock::time_point lastSignChangeTime;
    
    // 积分控制相关
    float integralX;
    float integralY;
    
    // 运动预测器
    DerivativePredictor predictor;
    
    // 时间步长自适应
    std::chrono::steady_clock::time_point lastTickTime;
    float deltaTime;
    
    bool connectSerial();
    void disconnectSerial();
    bool sendSerialCommand(const std::string& command);
    
    void move(int dx, int dy);
    void moveTo(int x, int y);
    void click(bool left = true);
    void clickDown();
    void clickUp();
    void wheel(int delta);
    
    float calculateDynamicP(float distance);
    float calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor);
    float calculateIntegral(float error, float& integral, float deltaTime);
    Detection* selectTarget();
    POINT convertToScreenCoordinates(const Detection& det);
    void resetPidState();
    void resetMotionState();
    void performAutoClick();
    void releaseAutoTrigger();
    int getRandomDelay();
    int getRandomDuration();
    float getCurrentPGain();
    
    std::chrono::steady_clock::time_point hotkeyPressStartTime;
    bool yUnlockActive;
    std::chrono::steady_clock::time_point lastAutoTriggerTime;
    std::chrono::steady_clock::time_point autoTriggerFireStartTime;
    std::chrono::steady_clock::time_point autoTriggerDelayStartTime;
    bool autoTriggerHolding;
    bool autoTriggerWaitingForDelay;
    int currentFireDuration;
    std::mt19937 randomGenerator;
    
    int currentTargetTrackId;
    std::chrono::steady_clock::time_point targetLockStartTime;
    float currentTargetDistance;
    
    std::string currentWeapon_;
};

#endif

#endif
