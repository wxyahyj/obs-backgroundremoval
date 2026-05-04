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
    AdvancedPID,      // 高级PID（精简版）
    ExternalPID       // 外部PID库（pid_x64.lib）
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
    
    // 高级PID参数（精简版）
    float pidPMin = 0.15f;
    float pidPMax = 0.6f;
    float pidPSlope = 1.0f;
    float pidD = 0.007f;
    float pidI = 0.01f;
    float maxPixelMove = 128.0f;
    float deadZonePixels = 5.0f;
    float targetYOffset = 0.0f;
    float derivativeFilterAlpha = 0.2f;
    // 高级PID可调参数（对应专业PID的predict和rate）
    float adaptivePGainRate = 0.03f;      // 自适应P增益变化率（对应专业PID的predict）
    float dTermScale = 0.3f;              // D项缩放因子（对应专业PID的rate）
    // 注：自适应增益、卡尔曼滤波、跳变检测(30像素)、软限幅(atan2)始终启用，像专业PID一样自动运行
    
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
    float integralRate = 1.0f;             // 积分速率系数
    float pGainRampInitialScale = 0.6f;
    float pGainRampDuration = 0.5f;
    // DerivativePredictor配置
    bool useDerivativePredictor = true;    // 是否启用导数预测器
    float predictionWeightX = 0.5f;
    float predictionWeightY = 0.1f;
    float maxPredictionTime = 0.1f;        // 最大预测时间(秒)
    
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
    
    // GhostTracker曲线轨迹参数
    bool enableGhostTracker = false;        // 是否启用曲线轨迹
    float ghostCurvature = 0.5f;            // 曲线强度 (0.0 - 1.0)
    float ghostNoiseIntensity = 12.0f;      // 最大噪声像素
    float ghostVerticalSnapRatio = 3.0f;    // 垂直吸附比例阈值
    float ghostNoiseFreq = 0.8f;            // Perlin噪声频率
    
    // 外部PID参数（pid_x64.lib）
    float externalKpX = 1.5f;                  // X轴比例系数
    float externalKiX = 0.0f;                  // X轴积分系数
    float externalKdX = 1.5f;                  // X轴微分系数
    float externalKpY = 1.5f;                  // Y轴比例系数
    float externalKiY = 0.0f;                  // Y轴积分系数
    float externalKdY = 1.5f;                  // Y轴微分系数
    float externalPredictX = 1.0f;             // X轴预测参数
    float externalPredictY = 1.0f;             // Y轴预测参数
    float externalRateX = 0.3f;                // X轴采样率
    float externalRateY = 0.3f;                // Y轴采样率
    float externalKiMode = 1.0f;               // 积分模式
    float externalKpLimit = 9900.0f;           // P项限幅
    float externalKiLimit = 9900.0f;           // I项限幅
    float externalKdLimit = 9900.0f;           // D项限幅
    float externalOutputLimit = 0.0f;          // 输出限幅（0=不限幅）
    float externalKiRate = 0.05f;              // 积分速率
    float externalKiDeadband = 0.5f;           // 积分死区
    
    // 多指标融合追踪权重
    float trackingWeightIou = 0.4f;       // IoU距离权重
    float trackingWeightCenter = 0.3f;    // 中心点距离权重
    float trackingWeightAspect = 0.15f;   // 宽高比距离权重
    float trackingWeightArea = 0.15f;     // 面积距离权重
    
    // 神经网络轨迹生成器配置
    bool enableNeuralPath = false;             // 是否启用神经网络轨迹
    int neuralPathPoints = 25;                 // 轨迹点数量
    double neuralMouseStepSize = 8.0;          // 鼠标步长
    int neuralTargetRadius = 8;                // 目标半径（到达判定）
    int neuralConsumePerFrame = 2;             // 每帧消费路径点数（加速执行）
    bool enableNeuralPathDebug = false;        // 是否输出神经网络调试日志

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

    // 设置瞄准起点（准星位置），替代默认的画面中心
    // x/y 为像素坐标，设为-1表示使用画面中心
    virtual void setAimOrigin(float x, float y) = 0;
};

#endif
