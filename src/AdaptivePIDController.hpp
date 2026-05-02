#ifndef ADAPTIVE_PID_CONTROLLER_HPP
#define ADAPTIVE_PID_CONTROLLER_HPP

#ifdef _WIN32

#include <cmath>
#include <algorithm>
#include "DerivativePredictor.hpp"

struct AdaptivePIDConfig {
    float baseKp = 0.5f;
    float baseKi = 0.1f;
    float baseKd = 0.05f;
    
    float integralGainThreshold = 5.0f;
    float kpGainThreshold = 5.0f;
    float integralGainRate = 0.1f;
    float kpGainRate = 0.1f;
    float largeErrorRate = 0.1f;
    
    float maxOutput = 1000.0f;
    float maxIntegral = 1000.0f;
    
    bool usePredictor = true;
    float predWeightX = 0.5f;
    float predWeightY = 0.1f;
    float maxPredTime = 0.1f;
    
    float outputSmoothing = 0.7f;
    float derivativeFilterAlpha = 0.3f;
};

class AdaptivePIDAxis {
public:
    AdaptivePIDAxis();
    
    float calculate(float error, float dt);
    void adjustIntegralGain(float error);
    void adjustKpGain(float error);
    
    void setConfig(const AdaptivePIDConfig& config);
    void setMaxOutput(float max);
    void setMaxIntegral(float max);
    
    void reset();
    
    float getLastProportional() const { return lastProportional_; }
    float getLastIntegral() const { return lastIntegral_; }
    float getLastDerivative() const { return lastDerivative_; }
    float getCurrentKpGain() const { return kpGain_; }
    float getCurrentIntegralGain() const { return integralGain_; }
    
private:
    float baseKp_ = 0.5f;
    float baseKi_ = 0.1f;
    float baseKd_ = 0.05f;
    
    float integralGainThreshold_ = 5.0f;
    float kpGainThreshold_ = 5.0f;
    float integralGainRate_ = 0.1f;
    float kpGainRate_ = 0.1f;
    float largeErrorRate_ = 0.1f;
    
    float integralGain_ = 1.0f;
    float kpGain_ = 1.0f;
    
    float integralSum_ = 0.0f;
    float prevError_ = 0.0f;
    float derivativeFiltered_ = 0.0f;
    
    float maxOutput_ = 1000.0f;
    float maxIntegral_ = 1000.0f;
    
    float lastProportional_ = 0.0f;
    float lastIntegral_ = 0.0f;
    float lastDerivative_ = 0.0f;
};

class AdaptiveAimController {
public:
    AdaptiveAimController();
    
    void update(float rawDx, float rawDy, double currentTime, float& outX, float& outY);
    void setConfig(const AdaptivePIDConfig& config);
    void reset();
    void resetPredictor();
    
    struct DebugTerms {
        float pTermX = 0, pTermY = 0;
        float iTermX = 0, iTermY = 0;
        float dTermX = 0, dTermY = 0;
        float kpGainX = 0, kpGainY = 0;
        float iGainX = 0, iGainY = 0;
        float predX = 0, predY = 0;
        float fusionErrorX = 0, fusionErrorY = 0;
    };
    DebugTerms getLastDebugTerms() const { return lastDebugTerms_; }
    
private:
    AdaptivePIDAxis axisX_;
    AdaptivePIDAxis axisY_;
    DerivativePredictor predictor_;
    AdaptivePIDConfig config_;
    
    float lastOutputX_ = 0.0f;
    float lastOutputY_ = 0.0f;
    double lastTime_ = 0.0;
    
    DebugTerms lastDebugTerms_;
};

#endif

#endif
