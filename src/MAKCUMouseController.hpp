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
#include "KalmanFilter.hpp"

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

    bool getKalmanPrediction(float& predX, float& predY) const override;

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
    
    // 目标锁定相关
    int lockedTrackId;  // 当前锁定的目标trackId，-1表示未锁定
    
    // 自动压枪相关
    std::chrono::steady_clock::time_point lastRecoilTime;
    bool isFiring;
    
    // 运动预测器
    DerivativePredictor predictor;

    // 卡尔曼滤波器
    KalmanFilter kalmanFilter;
    bool kalmanFilterInitialized;

    // 贝塞尔曲线移动状态
    struct BezierState {
        bool active = false;
        float startX, startY;
        float endX, endY;
        float controlX, controlY;
        float progress = 0.0f;
        std::chrono::steady_clock::time_point startTime;
        float duration = 0.0f;
    };
    BezierState bezierState;

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
    
    // 贝塞尔曲线相关函数
    void initBezierMovement(float startX, float startY, float endX, float endY);
    void getQuadraticBezierPoint(float t, float& x, float& y);
    float calculateBezierDuration(float distance);
    
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
    
    std::string currentWeapon_;
    
    PidDataCallback pidDataCallback_;
};

#endif

#endif
