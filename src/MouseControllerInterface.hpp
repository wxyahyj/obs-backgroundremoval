#ifndef MOUSE_CONTROLLER_INTERFACE_HPP
#define MOUSE_CONTROLLER_INTERFACE_HPP

#include <vector>
#include <string>
#include <functional>
#include "models/Detection.h"

enum class ControllerType {
    WindowsAPI,
    MAKCU
};

enum class AlgorithmType {
    AdvancedPID,      // 高级PID（当前使用的自适应PID）
    StandardPID,      // 标准PID（经典PID）
    ChrisPID,         // ChrisPID（克里斯控制器）
    DynamicPID        // 动态PID（动态阈值状态机）
};

// PID数据回调函数类型
struct PidDebugData {
    // === 基础数据（原有） ===
    float errorX;
    float errorY;
    float outputX;
    float outputY;
    float targetX;
    float targetY;
    float targetVelocityX;  // 目标X速度（像素/帧）
    float targetVelocityY;  // 目标Y速度（像素/帧）
    float currentKp;        // 当前使用的Kp
    float currentKi;        // 当前使用的Ki
    float currentKd;        // 当前使用的Kd

    // === 新增：P/I/D 分项输出 ===
    float pTermX = 0;       // X轴 P项贡献
    float pTermY = 0;       // Y轴 P项贡献
    float iTermX = 0;       // X轴 I项贡献
    float iTermY = 0;       // Y轴 I项贡献
    float dTermX = 0;       // X轴 D项贡献
    float dTermY = 0;       // Y轴 D项贡献

    // === 新增：积分健康状态 ===
    float integralAbsX = 0;         // 积分绝对值 X
    float integralAbsY = 0;         // 积分绝对值 Y
    float integralLimitX = 1.0f;    // X积分限幅值
    float integralLimitY = 1.0f;    // Y积分限幅值
    float integralRatioX = 0;       // X积分占用率 (0~1)
    float integralRatioY = 0;       // Y积分占用率 (0~1)

    // === 新增：控制模式自动诊断 ===
    int controlMode = 0;            // 0=IDLE(无目标) 1=TRACKING 2=LOCKED(锁定) 3=I_SATURATION(饱和告警) 4=OSCILLATING(振荡) 5=PREDICTING(预测主导)
    int algorithmType = 0;          // 0=AdvancedPID 1=StandardPID 2=ChrisPID

    // === 新增：额外诊断信息 ===
    bool isFiring = false;          // 是否正在射击（压枪状态）
    float smoothingFactorX = 0;     // 当前X平滑系数
    float smoothingFactorY = 0;     // 当前Y平滑系数
};
using PidDataCallback = std::function<void(const PidDebugData&)>;

struct MouseControllerConfig {
    bool enableMouseControl = false;
    int hotkeyVirtualKey = 0;
    int fovRadiusPixels = 100;
    float sourceCanvasPosX = 0.0f;
    float sourceCanvasPosY = 0.0f;
    float sourceCanvasScaleX = 1.0f;
    float sourceCanvasScaleY = 1.0f;
    int sourceWidth = 1920;
    int sourceHeight = 1080;
    int inferenceFrameWidth = 0;
    int inferenceFrameHeight = 0;
    int cropOffsetX = 0;
    int cropOffsetY = 0;
    int screenOffsetX = 0;
    int screenOffsetY = 0;
    int screenWidth = 0;
    int screenHeight = 0;
    
    // 算法选择
    AlgorithmType algorithmType = AlgorithmType::AdvancedPID;
    
    // 高级PID参数（当前使用的算法）
    float pidPMin = 0.15f;
    float pidPMax = 0.6f;
    float pidPSlope = 1.0f;
    float pidD = 0.007f;
    float pidI = 0.01f;  // 积分系数（和标准PID的stdKi一致）
    float aimSmoothingX = 0.7f;
    float aimSmoothingY = 0.5f;
    float maxPixelMove = 128.0f;
    float deadZonePixels = 5.0f;
    float targetYOffset = 0.0f;
    float derivativeFilterAlpha = 0.2f;
    
    // 高级PID增强参数（状态机+动态阈值）
    float advTargetThreshold = 10.0f;         // 达标误差阈值
    float advMinCoefficient = 1.5f;           // 动态阈值最小系数
    float advMaxCoefficient = 2.5f;           // 动态阈值最大系数
    float advTransitionSharpness = 5.0f;      // Sigmoid过渡锐度
    float advTransitionMidpoint = 0.3f;       // Sigmoid过渡中点
    float advOutputSmoothing = 0.7f;          // 输出平滑系数（EMA）
    float advSpeedFactor = 0.5f;              // 未达标时速度因子（半速）
    
    // 一欧元滤波器参数（高级PID可选输出平滑）
    bool useOneEuroFilter = false;            // 是否使用一欧元滤波器替代EMA
    float oneEuroMinCutoff = 1.0f;            // 最小截止频率
    float oneEuroBeta = 0.0f;                 // 速度因子（β*|速度|动态调整）
    float oneEuroDCutoff = 1.0f;              // 微分截止频率（速度平滑）
    
    ControllerType controllerType = ControllerType::WindowsAPI;
    std::string makcuPort;
    int makcuBaudRate = 115200;
    int yUnlockDelayMs = 300;
    bool yUnlockEnabled = false;
    bool autoTriggerEnabled = false;
    int autoTriggerRadius = 5;
    int autoTriggerCooldownMs = 200;
    int autoTriggerFireDelay = 0;
    int autoTriggerFireDuration = 50;
    int autoTriggerInterval = 50;
    bool autoTriggerDelayRandomEnabled = false;
    int autoTriggerDelayRandomMin = 0;
    int autoTriggerDelayRandomMax = 0;
    bool autoTriggerDurationRandomEnabled = false;
    int autoTriggerDurationRandomMin = 0;
    int autoTriggerDurationRandomMax = 0;
    int autoTriggerMoveCompensation = 0;
    int targetSwitchDelayMs = 500;
    float targetSwitchTolerance = 0.15f;
    float integralLimit = 100.0f;
    float integralSeparationThreshold = 50.0f;
    float integralDeadZone = 5.0f;
    float integralRate = 0.015f;  // 积分增益变化率（和标准PID一致）
    float pGainRampInitialScale = 0.6f;
    float pGainRampDuration = 0.5f;
    // DerivativePredictor配置
    bool useDerivativePredictor = true;    // 是否启用导数预测器
    float predictionWeightX = 0.5f;
    float predictionWeightY = 0.1f;
    float maxPredictionTime = 0.1f;        // 最大预测时间(秒)

    // 标准PID参数（经典PID算法）
    float stdKp = 0.3f;           // 比例系数
    float stdKi = 0.01f;          // 积分系数
    float stdKd = 0.005f;         // 微分系数
    float stdOutputLimit = 50.0f; // 输出限幅
    float stdDeadZone = 0.3f;     // 死区
    float stdIntegralLimit = 100.0f;      // 积分限幅
    float stdIntegralDeadzone = 1.0f;     // 积分死区
    float stdIntegralThreshold = 50.0f;   // 积分分离阈值
    float stdIntegralRate = 0.015f;       // 积分增益变化率
    float stdDerivativeFilterAlpha = 0.2f; // 微分滤波系数（和高级PID一致）
    float stdSmoothingX = 0.7f;   // 标准PID输出X轴平滑系数
    float stdSmoothingY = 0.5f;   // 标准PID输出Y轴平滑系数
    
    // 持续自瞄和自动压枪配置
    bool continuousAimEnabled = false;
    bool autoRecoilControlEnabled = false;
    float recoilStrength = 5.0f;
    int recoilSpeed = 16;
    float recoilPidGainScale = 0.3f;
    
    // 贝塞尔曲线移动参数
    bool enableBezierMovement = false;
    float bezierCurvature = 0.3f;
    float bezierRandomness = 0.2f;
    
    // ChrisPID参数（完全复刻克里斯控制器）
    float chrisKp = 0.45f;
    float chrisKi = 0.02f;
    float chrisKd = 0.04f;
    float chrisPredWeightX = 0.5f;
    float chrisPredWeightY = 0.1f;
    float chrisInitScale = 0.6f;
    float chrisRampTime = 0.5f;
    float chrisOutputMax = 150.0f;
    float chrisIMax = 100.0f;
    float chrisDFilterAlpha = 0.3f;  // D项滤波系数
    
    // 动态PID参数（动态阈值状态机控制器）
    float dynamicKp = 0.5f;
    float dynamicKi = 0.1f;
    float dynamicKd = 0.05f;
    float dynamicTargetThreshold = 4.0f;       // 达标误差阈值
    float dynamicSpeedMultiplier = 1.0f;       // 速度倍率
    float dynamicMinCoefficient = 1.6f;        // 最小系数
    float dynamicMaxCoefficient = 2.7f;        // 最大系数
    float dynamicTransitionSharpness = 5.0f;   // 过渡锐度
    float dynamicTransitionMidpoint = 0.0f;    // 动态过渡中点
    int   dynamicMinDataPoints = 2;            // 最小数据量
    float dynamicErrorTolerance = 3.0f;        // 误差变化容限
    float dynamicSmoothingFactor = 0.8f;       // 平滑因子
    
    // 多指标融合追踪权重
    float trackingWeightIou = 0.4f;       // IoU距离权重
    float trackingWeightCenter = 0.3f;    // 中心点距离权重
    float trackingWeightAspect = 0.15f;   // 宽高比距离权重
    float trackingWeightArea = 0.15f;     // 面积距离权重
    
    // MotionSimulator 人类行为模拟器配置
    bool enableMotionSimulator = false;        // 是否启用人类行为模拟
    bool motionSimRandomPos = true;            // 随机落点
    bool motionSimOvershoot = true;            // 过冲
    bool motionSimMicroOvershoot = true;       // 微过冲
    bool motionSimInertia = true;              // 惯性停止
    bool motionSimLeftBtnAdaptive = true;      // 左键自适应
    bool motionSimSprayMode = true;            // 连射模式
    bool motionSimTapPause = true;             // 点击暂停
    bool motionSimRetry = true;                // 重试
    int motionSimMaxRetry = 2;                 // 最大重试次数
    int motionSimDelayMs = 80;                 // 目标延迟(毫秒)
    float motionSimDirectProb = 0.85f;         // 直线移动概率
    float motionSimOvershootProb = 0.10f;      // 过冲概率
    float motionSimMicroOvshootProb = 0.05f;   // 微过冲概率
    
    // 神经网络轨迹生成器配置
    bool enableNeuralPath = false;             // 是否启用神经网络轨迹
    int neuralPathPoints = 25;                 // 轨迹点数量
    double neuralMouseStepSize = 4.0;          // 鼠标步长
    int neuralTargetRadius = 8;                // 目标半径（到达判定）
    bool enableNeuralPathDebug = false;        // 是否输出神经网络调试日志

    // 目标稳定性检测配置
    bool enableStabilityCheck = false;         // 是否启用稳定性检测
    int stabilityRequiredFrames = 3;           // 需要连续稳定的帧数
    float stabilityPositionThreshold = 5.0f;   // 位置稳定性阈值（像素）
    float stabilitySizeThreshold = 0.1f;       // 尺寸稳定性阈值（相对变化）
};

class MouseControllerInterface {
public:
    virtual ~MouseControllerInterface() = default;

    virtual void updateConfig(const MouseControllerConfig& config) = 0;
    
    virtual MouseControllerConfig getConfig() const = 0;

    virtual void setDetections(const std::vector<Detection>& detections) = 0;

    virtual void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) = 0;

    virtual void tick() = 0;
    
    virtual void setCurrentWeapon(const std::string& weaponName) = 0;
    
    virtual std::string getCurrentWeapon() const = 0;
    
    virtual ControllerType getControllerType() const = 0;

    // 设置PID数据回调函数（用于调试可视化）
    virtual void setPidDataCallback(PidDataCallback callback) = 0;

    // 线程控制接口（异步鼠标控制）
    virtual void start() = 0;   // 启动控制线程
    virtual void stop() = 0;    // 停止控制线程
    virtual bool isRunning() const = 0;  // 检查线程状态
};

#endif
