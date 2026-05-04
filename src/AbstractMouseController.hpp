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
#include <cmath>
#include "MouseControllerInterface.hpp"
#include "DerivativePredictor.hpp"
#include "curve.hpp"
#include "../libs/pid/pid.h"

// 简单1D卡尔曼滤波器（专业PID风格，成员变量公开）
class SimpleKalmanFilter {
public:
    SimpleKalmanFilter(float q = 0.1f, float r = 1.0f, float x0 = 0.0f, float p0 = 1.0f)
        : Q_(q), R_(r), x_(x0), P_(p0) {}

    void init(float x0) { x_ = x0; P_ = 1.0f; }
    
    float update(float measurement) {
        float P_pred = P_ + Q_;
        float K = P_pred / (P_pred + R_);
        float innov = measurement - x_;
        x_ = x_ + K * innov;
        P_ = (1.0f - K) * P_pred;
        return x_;
    }
    
    void reset() { x_ = 0.0f; P_ = 1.0f; }
    void setQ(float q) { Q_ = q; }
    void setR(float r) { R_ = r; }
    float getState() const { return x_; }

    // 专业PID需要访问这些成员
    float Q_, R_, x_, P_;
};

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
    
    // 自适应增益状态
    float adaptivePGainX;
    float adaptivePGainY;
    float adaptiveIGainX;
    float adaptiveIGainY;
    
    // 卡尔曼滤波器（专业PID风格：kf2 + kf3 两级级联）
    SimpleKalmanFilter kf2X;      // X轴kf2卡尔曼（D项第一级滤波）
    SimpleKalmanFilter kf2Y;      // Y轴kf2卡尔曼
    SimpleKalmanFilter kalmanOutputX;
    SimpleKalmanFilter kalmanOutputY;
    // kf3卡尔曼状态（第二级滤波）
    float kf3X_x;                 // X轴kf3状态
    float kf3X_P;                 // X轴kf3协方差
    float kf3Y_x;                 // Y轴kf3状态
    float kf3Y_P;                 // Y轴kf3协方差
    
    // 上一帧输出（专业PID的D项需要）
    float lastOutputX;
    float lastOutputY;
    
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
