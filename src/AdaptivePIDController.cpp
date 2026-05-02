#ifdef _WIN32

#include "AdaptivePIDController.hpp"
#include <obs-module.h>
#include <cmath>
#include <algorithm>

AdaptivePIDAxis::AdaptivePIDAxis()
    : baseKp_(0.5f)
    , baseKi_(0.1f)
    , baseKd_(0.05f)
    , integralGainThreshold_(5.0f)
    , kpGainThreshold_(5.0f)
    , integralGainRate_(0.1f)
    , kpGainRate_(0.1f)
    , largeErrorRate_(0.1f)
    , integralGain_(1.0f)
    , kpGain_(1.0f)
    , integralSum_(0.0f)
    , prevError_(0.0f)
    , derivativeFiltered_(0.0f)
    , maxOutput_(1000.0f)
    , maxIntegral_(1000.0f)
    , lastProportional_(0.0f)
    , lastIntegral_(0.0f)
    , lastDerivative_(0.0f)
{
}

void AdaptivePIDAxis::setConfig(const AdaptivePIDConfig& config)
{
    baseKp_ = config.baseKp;
    baseKi_ = config.baseKi;
    baseKd_ = config.baseKd;
    integralGainThreshold_ = config.integralGainThreshold;
    kpGainThreshold_ = config.kpGainThreshold;
    integralGainRate_ = config.integralGainRate;
    kpGainRate_ = config.kpGainRate;
    largeErrorRate_ = config.largeErrorRate;
    maxOutput_ = config.maxOutput;
    maxIntegral_ = config.maxIntegral;
}

void AdaptivePIDAxis::setMaxOutput(float max)
{
    maxOutput_ = max;
}

void AdaptivePIDAxis::setMaxIntegral(float max)
{
    maxIntegral_ = max;
}

void AdaptivePIDAxis::adjustIntegralGain(float error)
{
    float absError = std::fabs(error);
    float ratio;
    
    if (absError < integralGainThreshold_) {
        ratio = 1.0f - (absError / integralGainThreshold_);
        integralGain_ += (ratio - integralGain_) * integralGainRate_;
    } else {
        ratio = integralGainThreshold_ / absError;
        integralGain_ += (ratio * integralGain_ - integralGain_) * largeErrorRate_;
    }
    
    integralGain_ = std::clamp(integralGain_, 0.0f, 1.0f);
}

void AdaptivePIDAxis::adjustKpGain(float error)
{
    float absError = std::fabs(error);
    float ratio;
    
    if (absError < kpGainThreshold_) {
        ratio = 1.0f - (absError / kpGainThreshold_);
        kpGain_ += (ratio - kpGain_) * kpGainRate_;
    } else {
        ratio = kpGainThreshold_ / absError;
        kpGain_ += (ratio * kpGain_ - kpGain_) * largeErrorRate_;
    }
    
    kpGain_ = std::clamp(kpGain_, 0.0f, 1.0f);
}

float AdaptivePIDAxis::calculate(float error, float dt)
{
    if (dt <= 1e-6f) {
        dt = 0.01f;
    }
    
    adjustIntegralGain(error);
    adjustKpGain(error);
    
    float p = baseKp_ * kpGain_ * error;
    lastProportional_ = p;
    
    integralSum_ += baseKi_ * integralGain_ * error * dt;
    integralSum_ = std::clamp(integralSum_, -maxIntegral_, maxIntegral_);
    lastIntegral_ = integralSum_;
    
    float derivative = (error - prevError_) / dt;
    derivativeFiltered_ = derivativeFiltered_ * 0.7f + derivative * 0.3f;
    float d = baseKd_ * derivativeFiltered_;
    lastDerivative_ = d;
    
    float output = p + integralSum_ + d;
    output = std::clamp(output, -maxOutput_, maxOutput_);
    
    prevError_ = error;
    
    return output;
}

void AdaptivePIDAxis::reset()
{
    integralSum_ = 0.0f;
    prevError_ = 0.0f;
    derivativeFiltered_ = 0.0f;
    integralGain_ = 1.0f;
    kpGain_ = 1.0f;
    lastProportional_ = 0.0f;
    lastIntegral_ = 0.0f;
    lastDerivative_ = 0.0f;
}

AdaptiveAimController::AdaptiveAimController()
    : lastOutputX_(0.0f)
    , lastOutputY_(0.0f)
    , lastTime_(0.0)
{
}

void AdaptiveAimController::setConfig(const AdaptivePIDConfig& config)
{
    config_ = config;
    axisX_.setConfig(config);
    axisY_.setConfig(config);
    predictor_.setMaxPredictionTime(config.maxPredTime);
}

void AdaptiveAimController::update(float rawDx, float rawDy, double currentTime, float& outX, float& outY)
{
    float dt = 0.01f;
    double rawDt = 0.0;
    if (lastTime_ > 0.0) {
        rawDt = currentTime - lastTime_;
        dt = static_cast<float>(rawDt);
    }
    dt = std::clamp(dt, 0.001f, 0.05f);
    lastTime_ = currentTime;
    
    float predX = 0.0f, predY = 0.0f;
    if (config_.usePredictor) {
        predictor_.update(rawDx, rawDy, lastOutputX_, lastOutputY_, dt);
        predictor_.predict(dt, predX, predY);
    }
    
    float fusionErrorX = rawDx + predX * config_.predWeightX;
    float fusionErrorY = rawDy + predY * config_.predWeightY;
    
    float rawOutputX = axisX_.calculate(fusionErrorX, dt);
    float rawOutputY = axisY_.calculate(fusionErrorY, dt);
    
    outX = lastOutputX_ * (1.0f - config_.outputSmoothing) + rawOutputX * config_.outputSmoothing;
    outY = lastOutputY_ * (1.0f - config_.outputSmoothing) + rawOutputY * config_.outputSmoothing;
    
    lastOutputX_ = outX;
    lastOutputY_ = outY;
    
    lastDebugTerms_.pTermX = axisX_.getLastProportional();
    lastDebugTerms_.pTermY = axisY_.getLastProportional();
    lastDebugTerms_.iTermX = axisX_.getLastIntegral();
    lastDebugTerms_.iTermY = axisY_.getLastIntegral();
    lastDebugTerms_.dTermX = axisX_.getLastDerivative();
    lastDebugTerms_.dTermY = axisY_.getLastDerivative();
    lastDebugTerms_.kpGainX = axisX_.getCurrentKpGain();
    lastDebugTerms_.kpGainY = axisY_.getCurrentKpGain();
    lastDebugTerms_.iGainX = axisX_.getCurrentIntegralGain();
    lastDebugTerms_.iGainY = axisY_.getCurrentIntegralGain();
    lastDebugTerms_.predX = predX;
    lastDebugTerms_.predY = predY;
    lastDebugTerms_.fusionErrorX = fusionErrorX;
    lastDebugTerms_.fusionErrorY = fusionErrorY;
    
    static int logCounter = 0;
    if (++logCounter >= 30) {
        logCounter = 0;
        blog(LOG_INFO, "[AdaptivePID] === DT DEBUG === currentTime=%.3f lastTime=%.3f rawDt=%.4f clampedDt=%.4f",
             currentTime, lastTime_ - rawDt, rawDt, dt);
        blog(LOG_INFO, "[AdaptivePID] INPUT: rawDx=%.1f rawDy=%.1f | predX=%.1f predY=%.1f",
             rawDx, rawDy, predX, predY);
        blog(LOG_INFO, "[AdaptivePID] FUSION: fusionX=%.1f fusionY=%.1f | rawOutX=%.1f rawOutY=%.1f",
             fusionErrorX, fusionErrorY, rawOutputX, rawOutputY);
        blog(LOG_INFO, "[AdaptivePID] OUTPUT: lastOutX=%.1f lastOutY=%.1f | smooth=%.2f | finalX=%.1f finalY=%.1f",
             lastOutputX_ - outX * config_.outputSmoothing / (1.0f - config_.outputSmoothing + 0.001f),
             lastOutputY_ - outY * config_.outputSmoothing / (1.0f - config_.outputSmoothing + 0.001f),
             config_.outputSmoothing, outX, outY);
        blog(LOG_INFO, "[AdaptivePID] GAINS: kpGain=(%.3f,%.3f) iGain=(%.3f,%.3f)",
             lastDebugTerms_.kpGainX, lastDebugTerms_.kpGainY,
             lastDebugTerms_.iGainX, lastDebugTerms_.iGainY);
    }
}

void AdaptiveAimController::reset()
{
    axisX_.reset();
    axisY_.reset();
    predictor_.reset();
    lastOutputX_ = 0.0f;
    lastOutputY_ = 0.0f;
    lastTime_ = 0.0;
}

void AdaptiveAimController::resetPredictor()
{
    predictor_.reset();
}

#endif // _WIN32
