#ifdef _WIN32

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
    dynamicJudgmentThreshold = dynamicCoeff * recentTargetWidth;

    // 1. 正常达标入口（误差极小）
    if (!hasReached && std::abs(currentError) < targetThreshold) {
        hasReached = true;
    }
    // 2. 严重偏离目标：直接退出积分状态
    else if (std::abs(currentError) >= dynamicJudgmentThreshold) {
        hasReached = false;
        integralAccum = 0.0f;
        stableCount = 0;
    }
    // 3. 模糊区域：可能已经"差不多稳定"但误差还大
    else if (!hasReached && std::abs(currentError) >= targetThreshold && std::abs(currentError) <= dynamicJudgmentThreshold) {
        float diff = std::abs(currentError - previousError);

        if (diff < errorChangeTolerance) {
            stableCount++;
        } else {
            stableCount = 0;
        }

        if (stableCount >= minDataPoints) {
            hasReached = true;
            stableCount = 0;
            integralAccum = 0.0f;
        }
    }

    errorChangeRate = (currentError - previousError) / timeInterval;

    if (hasReached) {
        integralAccum += currentError * timeInterval;
        integralTerm = ki * integralAccum;
        proportionalTerm = kp * currentError;
        derivativeTerm = kd * errorChangeRate;
        derivativeOutput = derivativeTerm;
    } else {
        integralAccum += (currentError * 0.5f) * timeInterval;
        integralTerm = ki * integralAccum;
        proportionalTerm = (kp * 0.5f) * currentError;
        derivativeTerm = kd * errorChangeRate;
        derivativeOutput = derivativeTerm;
    }

    // 原始 PID 输出
    float rawOutput = proportionalTerm + integralTerm + derivativeOutput;

    // 输出平滑处理
    totalOutput = smoothingFactor * rawOutput;
    previousSmoothedOutput = totalOutput;

    if (hasReached) {
        float newVelocity = ((currentError - previousError) / timeInterval) + (rawOutput / timeInterval) * speedMultiplier;
        currentVelocity = newVelocity;
        previousVelocity = newVelocity;
    } else {
        currentVelocity = 0.0f;
        previousVelocity = 0.0f;
    }

    previousError = currentError;

    return totalOutput;
}

void DynamicPIDAxis::setBottomParams(float targetThreshold_, float speedMultiplier_, float minCoeff_, float maxCoeff_,
                                     float sharpness_, float midpoint_, int minDataPoints_, float errorTolerance_)
{
    targetThreshold = targetThreshold_;
    speedMultiplier = speedMultiplier_;
    minCoefficient = minCoeff_;
    maxCoefficient = maxCoeff_;
    transitionSharpness = sharpness_;
    dynamicTransitionMidpoint = midpoint_;
    minDataPoints = minDataPoints_;
    errorChangeTolerance = errorTolerance_;
}

void DynamicPIDAxis::updateParams(float kp_, float ki_, float kd_)
{
    kp = kp_;
    ki = ki_;
    kd = kd_;
}

void DynamicPIDAxis::setSmoothingFactor(float alpha)
{
    smoothingFactor = std::clamp(alpha, 0.0f, 1.0f);
}

float DynamicPIDAxis::getVelocity() const
{
    return currentVelocity;
}

void DynamicPIDAxis::reset()
{
    totalOutput = 0.0f;
    previousError = 0.0f;
    previousSmoothedOutput = 0.0f;
    integralAccum = 0.0f;
    frameCount = 0;
    currentVelocity = 0.0f;
    errorChangeRate = 0.0f;
    previousVelocity = 0.0f;
    hasReached = false;
    dynamicJudgmentThreshold = 0.0f;
    stableCount = 0;
}

#endif // _WIN32
