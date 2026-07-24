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
    // P1-2 safety: protect against division by zero
    if (timeInterval <= 1e-6f) return 0.0f;
    
    ++frameCount;

    // 动态阈值计算
    float widthRatio = recentTargetWidth / imageSize;
    float dynamicCoeff = minCoefficient + (maxCoefficient - minCoefficient) /
        (1.0f + std::exp(-transitionSharpness * (widthRatio - dynamicTransitionMidpoint)));

    // 输入死区
    if (std::abs(currentError) < dynamicCoeff * targetThreshold) {
        currentError = 0.0f;
    }

    // 积分计算（时间域）
    integralAccum += currentError * timeInterval;
    integralAccum = std::clamp(integralAccum, -100.0f, 100.0f);

    // 微分计算（时间域）
    float derivative = (currentError - previousError) / timeInterval;

    float P = kp * currentError;
    float I = ki * integralAccum;
    float D = kd * derivative;

    previousError = currentError;

    // 输出限幅
    float output = P + I + D;
    output = std::clamp(output, -50.0f, 50.0f);

    return output;
}

void DynamicPIDAxis::reset()
{
    integralAccum = 0.0f;
    previousError = 0.0f;
    frameCount = 0;
}

void DynamicPIDAxis::updateParams(float p, float i, float d)
{
    kp = p;
    ki = i;
    kd = d;
}

void DynamicPIDAxis::setBottomParams(float targetThresh, float speedMult, float minCoef, float maxCoef, 
                                     float sharpness, float midpoint, int minData, float errorTol)
{
    targetThreshold = targetThresh;
    speedMultiplier = speedMult;
    minCoefficient = minCoef;
    maxCoefficient = maxCoef;
    transitionSharpness = sharpness;
    dynamicTransitionMidpoint = midpoint;
    minDataPoints = minData;
    errorChangeTolerance = errorTol;
}

void DynamicPIDAxis::setSmoothingFactor(float alpha)
{
    smoothingFactor = std::clamp(alpha, 0.0f, 1.0f);
}

float DynamicPIDAxis::getVelocity() const
{
    return velocity;
}

#endif
