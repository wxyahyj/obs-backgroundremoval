#ifndef SLEWRATE_CONTROLLER_HPP
#define SLEWRATE_CONTROLLER_HPP

#include <algorithm>
#include <cmath>

namespace slewrate {

// 轴状态（内部指令、平滑趋近项、误差历史）
struct SlewAxisState {
    float command = 0.0f;          // 施加最终输出增益之前的内部指令值
    float filteredApproach = 0.0f; // 经过平滑处理的趋近/制动项
    float previousError = 0.0f;    // 上一帧的轴误差
    float errorRate = 0.0f;        // 误差变化速率
};

// 轴遥测数据（调试/分析用）
struct SlewAxisTelemetry {
    float proportional = 0.0f;
    float slewResidual = 0.0f;
    float filteredTerm = 0.0f;
};

// 控制器可调参数
struct SlewControllerParameters {
    float outputGain = 0.25f;         // 输出增益
    float responseSmoothing = 0.0008f; // 响应平滑系数
    float approachDamping = 5.0f;      // 趋近阻尼
    float updateIntervalMs = 5.0f;     // 更新间隔（毫秒）
    float normalizationScale = 5.0f;   // 归一化缩放系数
};

// 运行时状态（每实例一份）
struct SlewControllerRuntime {
    SlewAxisState x;
    SlewAxisState y;
    SlewAxisTelemetry xTelemetry;
    SlewAxisTelemetry yTelemetry;
};

// 输出结果
struct SlewAimOutput {
    float dx = 0.0f;
    float dy = 0.0f;
};

/// 更新单个轴的控制器状态并返回输出位移
/// @param state 轴状态（读写）
/// @param telemetry 遥测输出
/// @param error 当前误差（像素）
/// @param elapsedMs 帧间隔（毫秒）
/// @param parameters 控制器参数
/// @param updateState 是否更新内部状态（true=正常帧，false=仅计算不更新）
float updateAxis(
    SlewAxisState& state,
    SlewAxisTelemetry& telemetry,
    float error,
    float elapsedMs,
    const SlewControllerParameters& parameters,
    bool updateState);

/// 核心控制器——两轴同时更新
SlewAimOutput updateControllerCore(
    SlewControllerRuntime& runtime,
    float errorX,
    float errorY,
    float elapsedMs,
    const SlewControllerParameters& parameters,
    bool updateState,
    bool closeToTarget);

} // namespace slewrate

#endif // SLEWRATE_CONTROLLER_HPP