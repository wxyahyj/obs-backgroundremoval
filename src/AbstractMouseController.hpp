#ifndef ABSTRACT_MOUSE_CONTROLLER_HPP
#define ABSTRACT_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#define NOMINMAX
#include <windows.h>
#include <vector>
#include <mutex>
#include <random>
#include <chrono>
#include <string>
#include "MouseControllerInterface.hpp"
#include "DerivativePredictor.hpp"
#include "curve.hpp"
#include "../libs/pid/pid.h"

class AbstractMouseController : public MouseControllerInterface {
protected:
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
    
    float previousTargetX;
    float previousTargetY;
    float targetVelocityX;
    float targetVelocityY;

    float integralX;
    float integralY;
    float integralGainX;
    float integralGainY;
    
    int lockedTrackId;
    
    std::chrono::steady_clock::time_point lastRecoilTime;
    bool isFiring;
    
    DerivativePredictor predictor;

    PidController externalPidX;  // 外部PID X轴控制器
    PidController externalPidY;  // 外部PID Y轴控制器
    bool externalPidInitialized_; // 外部PID是否已初始化

    std::chrono::steady_clock::time_point lastTickTime;
    float deltaTime;
    
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
    
    // 延迟转火相关
    int pendingTargetTrackId;
    std::chrono::steady_clock::time_point pendingTargetStartTime;
    float pendingTargetScore;
    float currentTargetScore;
    
    std::string currentWeapon;
    
    float bezierPhase;
    
    PidDataCallback pidDataCallback_;
    
    // 神经网络轨迹生成器
    MMousePredictor neuralPathPredictor_;
    bool enableNeuralPath_;
    bool neuralPathInitialized_;
    bool enableNeuralPathDebug_;  // 神经网络调试日志开关
    std::vector<std::pair<double, double>> neuralPathPoints_;
    size_t neuralPathIndex_;
    float lastNeuralTargetX_;  // 上一个目标X坐标
    float lastNeuralTargetY_;  // 上一个目标Y坐标

    // 准星位置（瞄准起点），-1表示使用画面中心
    float aimOriginX_ = -1.0f;
    float aimOriginY_ = -1.0f;

    virtual void moveMouse(int dx, int dy) = 0;
    virtual void performClickDown() = 0;
    virtual void performClickUp() = 0;
    virtual bool checkFiring() = 0;
    
    // 神经网络轨迹初始化
    void initializeNeuralPathIfNeeded();
    
    float calculateDynamicP(float distance);
    float calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor);
    Detection* selectTarget();
    POINT convertToScreenCoordinates(const Detection& det);
    void resetPidState();
    void resetMotionState();
    void performAutoClick();
    void releaseAutoTrigger();
    int getRandomDelay();
    int getRandomDuration();
    float getCurrentPGain();
    
    virtual const char* getLogPrefix() const;

public:
    AbstractMouseController();
    virtual ~AbstractMouseController() = default;

    void updateConfig(const MouseControllerConfig& config) override;
    MouseControllerConfig getConfig() const override;
    void setDetections(const std::vector<Detection>& detections) override;
    void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) override;
    void tick() override;
    void setCurrentWeapon(const std::string& weaponName) override;
    std::string getCurrentWeapon() const override;
    void setPidDataCallback(PidDataCallback callback) override;
    void setAimOrigin(float x, float y) override;
};

#endif

#endif
