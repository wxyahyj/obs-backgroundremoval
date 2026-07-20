#include "SlewRateController.hpp"

namespace slewrate {

template <typename T>
static T clamp(T value, T low, T high) {
    return std::max(low, std::min(high, value));
}

float updateAxis(
    SlewAxisState& state,
    SlewAxisTelemetry& telemetry,
    float error,
    float elapsedMs,
    const SlewControllerParameters& parameters,
    bool updateState)
{
    // 帧间隔过短时不做任何处理，直接返回零
    if (elapsedMs < 0.5f) {
        if (updateState) {
            state.previousError = error;
        }
        telemetry = {};
        return 0.0f;
    }

    if (updateState) {
        const float previousError = state.previousError;
        const float updateTime = std::max(1.0f, parameters.updateIntervalMs);
        const float normalization =
            std::max(1.0f, parameters.normalizationScale);

        // 每次更新的目标误差变化量，限幅以过滤突变跳跃
        const float errorRate = clamp(
            (error - state.previousError) / updateTime,
            -10.0f,
            10.0f);

        state.errorRate = errorRate;
        state.previousError = error;

        // 基础比例项，经过归一化和硬限幅处理
        const float normalizedError = clamp(
            (error * parameters.outputGain) / normalization,
            -3.0f,
            3.0f);

        const float direction = error < 0.0f ? -1.0f : 1.0f;
        const float absoluteError = std::fabs(error);
        const float normalizedMagnitude =
            clamp(absoluteError / normalization, 0.0f, 3.0f);

        // 靠近目标时逐渐降低响应强度
        const float nearTargetFactor = std::min(absoluteError / 20.0f, 1.0f);

        // 若当前误差变化速率已足够快速收敛，则不再继续加速趋近目标
        const float availableApproach = clamp(
            normalizedMagnitude -
                std::max(-errorRate * direction, 0.0f),
            0.0f,
            3.0f);
        const float desiredApproach = availableApproach * direction;

        // 对趋近项进行低通平滑，平滑系数随误差增大而增大
        const float smoothing = clamp(
            normalization * parameters.responseSmoothing * nearTargetFactor,
            0.0f,
            1.0f);
        state.filteredApproach +=
            (desiredApproach - state.filteredApproach) * smoothing;

        // 越过目标时立即衰减已积累的运动量
        if (state.filteredApproach * error < 0.0f) {
            state.filteredApproach *= 0.75f;
        }

        float requestedCommand = clamp(
            state.filteredApproach + normalizedError,
            -3.0f,
            3.0f);

        const float proportional = normalizedError * normalization;
        float brakingResidual = 0.0f;

        // 若误差在保持同号的情况下持续缩小，则主动削减输出以抑制超调
        const float previousMagnitude = std::fabs(previousError);
        if (previousMagnitude > 1.0f &&
            previousMagnitude > absoluteError &&
            previousError * error > 0.0f) {

            const float dampingAmount =
                (1.0f - absoluteError / previousMagnitude) *
                (parameters.approachDamping * 0.2f);
            const float dampingMultiplier =
                clamp(1.0f - dampingAmount, 0.0f, 1.0f);

            const float beforeDamping = requestedCommand;
            requestedCommand *= dampingMultiplier;
            brakingResidual =
                (requestedCommand - beforeDamping) * normalization;
        }

        // 指令与误差符号相反时（已越过目标），进一步衰减指令和趋近项
        if (requestedCommand * error < 0.0f) {
            requestedCommand *= 0.75f;
            state.filteredApproach *= 0.75f;
        }

        // 限制每帧的指令变化量（slew rate 限制）
        const float requestedDelta = requestedCommand - state.command;
        const float appliedDelta = clamp(requestedDelta, -2.4000001f, 2.4000001f);
        state.command += appliedDelta;

        telemetry.proportional = proportional;
        telemetry.slewResidual =
            (appliedDelta - requestedDelta) * normalization + brakingResidual;
        telemetry.filteredTerm = state.filteredApproach * normalization;
    }

    return clamp(
        elapsedMs * state.command,
        -100.0f,
        100.0f);
}

SlewAimOutput updateControllerCore(
    SlewControllerRuntime& runtime,
    float errorX,
    float errorY,
    float elapsedMs,
    const SlewControllerParameters& parameters,
    bool updateState,
    bool closeToTarget)
{
    SlewAimOutput output{
        updateAxis(
            runtime.x,
            runtime.xTelemetry,
            errorX,
            elapsedMs,
            parameters,
            updateState),
        updateAxis(
            runtime.y,
            runtime.yTelemetry,
            errorY,
            elapsedMs,
            parameters,
            updateState),
    };

    // 接近选定目标时缩小输出
    if (closeToTarget) {
        output.dx *= 0.60000002f;
        output.dy *= 0.60000002f;
    }

    // 输出极小时归零，避免亚像素抖动
    if (std::fabs(output.dx) < 0.000001f) {
        output.dx = 0.0f;
    }
    if (std::fabs(output.dy) < 0.000001f) {
        output.dy = 0.0f;
    }

    return output;
}

} // namespace slewrate