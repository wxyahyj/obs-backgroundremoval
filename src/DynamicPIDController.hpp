#ifndef DYNAMIC_PID_CONTROLLER_HPP
#define DYNAMIC_PID_CONTROLLER_HPP

#ifdef _WIN32

#include <cmath>
#include <algorithm>

struct DynamicPIDAxisConfig {
    float kp = 0.5f;
    float ki = 0.1f;
    float kd = 0.05f;
    float targetThreshold = 4.0f;
    float speedMultiplier = 1.0f;
    float minCoefficient = 1.6f;
    float maxCoefficient = 2.7f;
    float transitionSharpness = 5.0f;
    float dynamicTransitionMidpoint = 0.0f;
    int minDataPoints = 2;
    float errorChangeTolerance = 3.0f;
    float smoothingFactor = 0.8f;
};

class DynamicPIDAxis {
public:
    DynamicPIDAxis();

    float controlLoop(float currentError, float timeInterval, float recentTargetWidth, float imageSize);
    float getVelocity() const;
    void reset();
    void setBottomParams(float targetThreshold, float speedMultiplier, float minCoeff, float maxCoeff,
                        float sharpness, float midpoint, int minDataPoints, float errorTolerance);
    void updateParams(float kp, float ki, float kd);
    void setSmoothingFactor(float alpha);

    // 获取调试信息
    float getLastProportional() const { return proportionalTerm; }
    float getLastIntegral() const { return integralTerm; }
    float getLastDerivative() const { return derivativeOutput; }
    bool getIsReached() const { return hasReached; }

private:
    float kp;
    float ki;
    float kd;

    float proportionalTerm;
    float integralTerm;
    float derivativeTerm;
    float derivativeOutput;
    int stableCount = 0;

    float previousError = 0.0f;
    float errorChangeRate = 0.0f;
    float integralAccum = 0.0f;

    float velocity = 0.0f;
    float currentVelocity = 0.0f;
    float previousVelocity = 0.0f;
    unsigned int frameCount = 0;
    float speedMultiplier = 1.0f;

    float previousSmoothedOutput = 0.0f;
    float smoothingFactor = 1.0f;
    float totalOutput = 0.0f;

    bool hasReached = false;
    float targetThreshold = 4.0f;
    float dynamicJudgmentThreshold = 0.5f;
    float minCoefficient = 1.6f;
    float maxCoefficient = 2.7f;
    float transitionSharpness = 5.0f;
    float dynamicTransitionMidpoint = 0.0f;
    int minDataPoints = 2;
    float errorChangeTolerance = 3.0f;
};

#endif // _WIN32

#endif // DYNAMIC_PID_CONTROLLER_HPP
