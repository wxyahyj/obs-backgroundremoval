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
    AdvancedPID,  // 高级PID（当前使用的自适应PID）
    StandardPID   // 标准PID（经典PID）
};

// PID数据回调函数类型
struct PidDebugData {
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
    float predictionWeightX = 0.5f;
    float predictionWeightY = 0.1f;

    // 卡尔曼滤波器配置
    bool useKalmanFilter = true;           // 是否启用卡尔曼滤波
    float kalmanProcessNoise = 0.01f;      // 过程噪声
    float kalmanMeasurementNoise = 1.0f;   // 测量噪声
    float kalmanConfidenceScale = 1.0f;    // 置信度缩放因子
    float kalmanPredictionWeightX = 0.2f;  // 卡尔曼预测X轴权重
    float kalmanPredictionWeightY = 0.1f;  // 卡尔曼预测Y轴权重

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
};

class MouseControllerInterface {
public:
    virtual ~MouseControllerInterface() = default;

    virtual void updateConfig(const MouseControllerConfig& config) = 0;

    virtual void setDetections(const std::vector<Detection>& detections) = 0;

    virtual void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) = 0;

    virtual void tick() = 0;
    
    virtual void setCurrentWeapon(const std::string& weaponName) = 0;
    
    virtual std::string getCurrentWeapon() const = 0;
    
    virtual ControllerType getControllerType() const = 0;

    // 获取卡尔曼滤波器预测的目标位置（绝对坐标， 用于可视化）
    // 返回值：是否成功获取预测位置
    virtual bool getKalmanPrediction(float& predX, float& predY) const = 0;
    
    // 设置PID数据回调函数（用于调试可视化）
    virtual void setPidDataCallback(PidDataCallback callback) = 0;
};

#endif
