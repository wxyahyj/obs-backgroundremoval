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
#include "SmithPredictor.hpp"
#include "SlewRateController.hpp"
#include "AdaptivePIDController.hpp"
#include "IMMFilter.hpp"
#include "VariationalBayesFilter.hpp"
#include "shuwu_pid.hpp"
#include "MotionSimulator.h"
#include "OneEuroFilter.hpp"
#include "curve.hpp"
#include "mpid.hpp"
#include "utils/GhostTracker.hpp"
#include "aim_controller.hpp"

// 切换到逆向重构的 PidController（来自 mpid.hpp 的 mist::reconstructed 命名空间）
using mist::reconstructed::PidController;

// 简单1D卡尔曼滤波器（专业PID风格，成员变量公开）
class SimpleKalmanFilter {
public:
    SimpleKalmanFilter(float q = 0.1f, float r = 1.0f, float x0 = 0.0f, float p0 = 1.0f)
        : Q_(q), R_(r), x_(x0), P_(p0) {}

    void init(float x0) { x_ = x0; P_ = 1.0f; }
    
    float update(float measurement) {
        float P_pred = P_ + Q_;
        float denom = P_pred + R_;
        float K = (denom > 1e-10f) ? P_pred / denom : 0.0f;
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
    // 目标丢失恢复后的预测衰减计数：冻结期 IMM 速度估计仍指向旧方向，
    // 恢复后前 N 帧压低预测权重，防止旧速度持续注入导致"一直往一个方向跑"
    int freezeRecoverFrames_ = 0;
    
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
    // 锁目标短暂丢失宽限（关联闪断/遮挡不立刻转火）
    int lockMissCount_ = 0;
    static constexpr int kMaxLockMissFrames = 8;
    
    std::chrono::steady_clock::time_point lastRecoilTime;
    bool isFiring;
    
    DerivativePredictor predictor;
    
    IMMFilter immFilter;
    bool immInitialized_;

    VariationalBayesFilter vbFilter;

    // 目标切换保护：换目标帧强制关预测 + 平滑重置（预测状态属于旧目标）
    int lastPredictionTrackId_ = -1;
    int smoothedTrackId_ = -1;

    OneEuroFilter oneEuroX_;
    OneEuroFilter oneEuroY_;
    int oneEuroLockedTrackId_ = -1;

    SmithPredictor smithPredictor;

    // avgInferenceTimeMs  推理延迟(毫秒)，Smith 自动 tau 使用
    float avgInferenceTimeMs_;

    // GhostTracker曲线轨迹生成器
    GhostTracker ghostTracker;

    PidController externalPidX;  // 外部PID X轴控制器
    PidController externalPidY;  // 外部PID Y轴控制器
    bool externalPidInitialized_; // 外部PID是否已初始化

    // aim 控制器（增量式PID+运动预测+柏林噪声，完整版）
    aim::AimController aimController_;

    // SlewRate控制器（限速平滑趋近）
    slewrate::SlewControllerRuntime slewRuntime_;
    bool slewRateInitialized_ = false;

    // 自适应PID控制器（位置式+自适应积分增益）
    AdaptivePIDController adaptivePidX_;
    AdaptivePIDController adaptivePidY_;

    // 书屋控制器（AiMod 完整移植：MotionSimulator 拟人仿真 + P_PID）
    shuwu::ShuWuPid shuwuPidX_;
    shuwu::ShuWuPid shuwuPidY_;
    MotionSimulator shuwuMotionSim_;

    AlgorithmType lastAppliedAlgorithm_ = AlgorithmType::AdvancedPID;  // 上次应用的算法类型，用于检测算法切换

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
    // 首次锁定确认：新目标连续 N 帧出现且是 bestTarget 才锁定（防单帧误检抢锁）
    int pendingLockTrackId = -1;
    int pendingLockFrames = 0;
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

    // 原 static 局部变量（改为成员变量支持多实例）
    bool wasHotkeyPressed_ = false;
    int targetFrameCount_ = 0;
    float lastCenterX_ = 0.0f;
    float lastCenterY_ = 0.0f;
    float maxCenterDelta_ = 0.0f;
    int neuralLogCount_ = 0;
    int frameCount_ = 0;
    int moveFrameCount_ = 0;
    int deadZoneFrameCount_ = 0;
    int logCounter_ = 0;
    int externalLogCounter_ = 0;

    // 检测结果新鲜度: 双机/UDP 检测帧率低时, 无新结果期间移动输出指数衰减,
    // 抑制"同一旧位置持续推"造成的来回过冲振荡 (单机高检测帧率不受影响).
    std::chrono::steady_clock::time_point lastDetectionsUpdate_ = std::chrono::steady_clock::now();

    // 移动输出 EMA 平滑状态 (aim_smoothing): 抑制低分辨率/双机画面检测框像素抖动造成的大幅突跳
    // 目标中心平滑状态 (低分辨率/双机画面检测框每帧跳几十像素)
    float smoothedTargetX_ = -1.0f;
    float smoothedTargetY_ = -1.0f;

    // 子像素累积器：SendInput 需要整数 mickey，但 PID/SlewRate 等输出常为 <1px 浮点数；
    // 游戏 Raw Input 下小数被 static_cast<int> 截断为 0 → 视觉上"不动"。此处累积余数凑够 1 再发。
    float subpixelAccumX_ = 0.0f;
    float subpixelAccumY_ = 0.0f;

    virtual void moveMouse(int dx, int dy) = 0;
    virtual void performClickDown() = 0;
    virtual void performClickUp() = 0;
    virtual bool checkFiring() = 0;

    // 热键物理按键状态。默认查本机键盘 (GetAsyncKeyState);
    // MAKCU 等硬件控制器覆写为设备上报的按键状态 (双机场景: 主机的按键经 MAKCU 固件上报到辅机).
    bool isPhysicalButtonPressed(int vk) override;
    
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
    void setDetectionsWithFrameSize(std::vector<Detection>&& detections, int frameWidth, int frameHeight, int cropX, int cropY) override;
    void tick() override;
    void setCurrentWeapon(const std::string& weaponName) override;
    std::string getCurrentWeapon() const override;
    void setPidDataCallback(PidDataCallback callback) override;
    void setAimOrigin(float x, float y) override;
    void setInferenceTimeMs(float ms) override;
};

#endif

#endif
