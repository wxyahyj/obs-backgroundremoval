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
#include "ChrisPIDController.hpp"
#include "DynamicPIDController.hpp"
#include "AdaptivePIDController.hpp"
#include "IncrementalPIDController.hpp"
#include "OneEuroFilter.hpp"
#include "MotionSimulator.hpp"
#include "curve.hpp"
#include "TargetStabilityAnalyzer.hpp"

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
    
    float stdIntegralX;
    float stdIntegralY;
    float stdIntegralGainX;
    float stdIntegralGainY;
    float stdLastErrorX;
    float stdLastErrorY;
    float stdFilteredDeltaErrorX;
    float stdFilteredDeltaErrorY;
    float stdPreviousMoveX;
    float stdPreviousMoveY;
    
    bool advHasReachedX;
    bool advHasReachedY;
    int advStableCountX;
    int advStableCountY;
    float advPreviousOutputX;
    float advPreviousOutputY;
    
    int lockedTrackId;
    
    std::chrono::steady_clock::time_point lastRecoilTime;
    bool isFiring;
    
    DerivativePredictor predictor;

    ChrisAimController chrisController_;

    DynamicPIDAxis dynamicPidX;  // 动态PID X轴控制器
    DynamicPIDAxis dynamicPidY;  // 动态PID Y轴控制器

    AdaptiveAimController adaptiveController_;  // 自适应PID控制器

    IncrementalPIDAdapter incrementalController_;  // 增量式PID控制器

    OneEuroFilter oneEuroX;  // 高级PID一欧元滤波器X轴
    OneEuroFilter oneEuroY;  // 高级PID一欧元滤波器Y轴

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
    
    // MotionSimulator 人类行为模拟器
    MotionSimulator motionSimulator_;
    bool enableMotionSimulator_;
    bool motionSimInitialized_;
    
    // 神经网络轨迹生成器
    MMousePredictor neuralPathPredictor_;
    bool enableNeuralPath_;
    bool neuralPathInitialized_;
    bool enableNeuralPathDebug_;  // 神经网络调试日志开关
    std::vector<std::pair<double, double>> neuralPathPoints_;
    size_t neuralPathIndex_;
    float lastNeuralTargetX_;  // 上一个目标X坐标
    float lastNeuralTargetY_;  // 上一个目标Y坐标

    // 目标稳定性分析器
    TargetStabilityAnalyzer stabilityAnalyzer_;
    int currentFrameWidth_ = 0;
    int currentFrameHeight_ = 0;

    virtual void moveMouse(int dx, int dy) = 0;
    virtual void performClickDown() = 0;
    virtual void performClickUp() = 0;
    virtual bool checkFiring() = 0;
    
    // 神经网络轨迹初始化
    void initializeNeuralPathIfNeeded();
    
    float calculateDynamicP(float distance);
    float calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor);
    float calculateIntegral(float error, float& integral, float& integralGain, float lastError, float deltaTime);
    bool adjustIntegralGain(float error, float lastError, float& integralGain);
    float calculateStandardPID(float error, float& integral, float& integralGain, 
                               float& lastError, float& filteredDeltaError, float deltaTime);
    bool adjustStandardIntegral(float error, float lastError, float& integralGain);
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
};

#endif

#endif
