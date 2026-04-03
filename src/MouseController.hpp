#ifndef MOUSE_CONTROLLER_HPP
#define MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include <windows.h>
#include <vector>
#include <mutex>
#include <random>
#include <chrono>
#include "MouseControllerInterface.hpp"
#include "DerivativePredictor.hpp"
#include "KalmanFilter.hpp"
#include "TargetTracker.hpp"

class MouseController : public MouseControllerInterface {
public:
    MouseController();
    ~MouseController();

    void updateConfig(const MouseControllerConfig& config) override;
    
    void setDetections(const std::vector<Detection>& detections) override;

    void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) override;
    
    void tick() override;
    
    void setCurrentWeapon(const std::string& weaponName);
    
    std::string getCurrentWeapon() const;
    
    ControllerType getControllerType() const override { return ControllerType::WindowsAPI; }

    bool getKalmanPrediction(float& predX, float& predY) const override;

private:
    mutable std::mutex mutex;
    MouseControllerConfig config;
    std::vector<Detection> currentDetections;
    
    int cachedScreenWidth;
    int cachedScreenHeight;
    
    bool isMoving;
    POINT startPos;
    POINT targetPos;
    
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
    
    // 目标速度计算
    float previousTargetX;
    float previousTargetY;
    float targetVelocityX;  // 目标X速度（像素/帧）
    float targetVelocityY;  // 目标Y速度（像素/帧）

    // 积分控制相关
    float integralX;
    float integralY;
    float integralGainX;  // 高级PID积分增益
    float integralGainY;
    
    // 标准PID状态变量
    float stdIntegralX;
    float stdIntegralY;
    float stdIntegralGainX;
    float stdIntegralGainY;
    float stdLastErrorX;
    float stdLastErrorY;
    float stdFilteredDeltaErrorX;  // 标准PID微分滤波状态
    float stdFilteredDeltaErrorY;
    float stdPreviousMoveX;  // 标准PID平滑状态
    float stdPreviousMoveY;
    
    // 自动压枪相关
    std::chrono::steady_clock::time_point lastRecoilTime;
    bool isFiring;
    
    // 运动预测器
    DerivativePredictor predictor;

    // 卡尔曼滤波器
    KalmanFilter kalmanFilter;
    bool kalmanFilterInitialized;

    // 增强的目标跟踪器
    TargetTracker targetTracker;

    // 时间步长自适应
    std::chrono::steady_clock::time_point lastTickTime;
    float deltaTime;
    
    float calculateDynamicP(float distance);
    float calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor);
    float calculateIntegral(float error, float& integral, float& integralGain, float lastError, float deltaTime);
    bool adjustIntegralGain(float error, float lastError, float& integralGain);
    float calculateStandardPID(float error, float& integral, float& integralGain, 
                               float& lastError, float& filteredDeltaError, float deltaTime);
    bool adjustStandardIntegral(float error, float lastError, float& integralGain);
    Detection* selectTarget();
    POINT convertToScreenCoordinates(const Detection& det);
    void moveMouseTo(const POINT& pos);
    void startMouseMovement(const POINT& target);
    void resetPidState();
    void resetMotionState();
    void performAutoClick();
    void releaseAutoTrigger();
    int getRandomDelay();
    int getRandomDuration();
    float getCurrentPGain();
    
    void setPidDataCallback(PidDataCallback callback) override {
        pidDataCallback_ = callback;
    }
    
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
    
    std::string currentWeapon;
    
    PidDataCallback pidDataCallback_;
};

#endif

#endif
