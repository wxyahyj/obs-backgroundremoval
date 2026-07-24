#ifdef _WIN32

// NOTE: DynamicPIDAxis is dt-aware and compiled, but NOT wired into AlgorithmType dispatch.
// Current UI "dynamic_kp/ki/kd" settings are not consumed by any active controller path.
// Do not use for production until connected via MouseControllerInterface enum.

#include "DynamicPIDController.hpp"
#include <cmath>
#include <algorithm>

DynamicPIDAxis::DynamicPIDAxis()
    : kp(0.5f)
    , ki(0.1f)
    , kd(0.05f)
{}

float DynamicPIDAxis::controlLoop(float currentError, float timeInterval, float recentTargetWidth, float imageSize)
{
    ++frameCount;

    // 动态阈值计算
    float widthRatio = recentTargetWidth / imageSize;
    float dynamicCoeff = minCoefficient + (maxCoefficient - minCoefficient) /
        (1.0f + std::exp(-transitionSharpness * (widthRatio - dynamicTransitionMidpoint)));

    // 输入死区
    if (std::abs(currentError) < dynamicCoeff * dynamicThresholdBase) {
        currentError = 0.0f;
    }

    // 积分计算（时间域）
    integral += currentError * timeInterval;
    integral = std::clamp(integral, -integralLimit, integralLimit);

    // 微分计算（时间域）
    float derivative = (currentError - lastError) / timeInterval;

    float P = kp * currentError;
    float I = ki * integral;
    float D = kd * derivative;

    lastError = currentError;

    // 输出限幅
    float output = P + I + D;
    output = std::clamp(output, -outputLimit, outputLimit);

    return output;
}

void DynamicPIDAxis::reset()
{
    integral = 0.0f;
    lastError = 0.0f;
    frameCount = 0;
}

void DynamicPIDAxis::setGains(float p, float i, float d)
{
    kp = p;
    ki = i;
    kd = d;
}

void DynamicPIDAxis::setThresholdParams(float baseThreshold, float minCoef, float maxCoef, float sharpness, float midpoint)
{
    dynamicThresholdBase = baseThreshold;
    minCoefficient = minCoef;
    maxCoefficient = maxCoef;
    transitionSharpness = sharpness;
    dynamicTransitionMidpoint = midpoint;
}

void DynamicPIDAxis::setLimits(float integralLim, float outputLim)
{
    integralLimit = integralLim;
    outputLimit = outputLim;
}

#endif
