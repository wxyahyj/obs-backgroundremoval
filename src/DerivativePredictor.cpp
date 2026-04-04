#ifdef _WIN32

#include "DerivativePredictor.hpp"
#include <algorithm>
#include <cmath>

DerivativePredictor::DerivativePredictor()
    : velocityX(0.0f)
    , velocityY(0.0f)
    , accelerationX(0.0f)
    , accelerationY(0.0f)
    , velocitySmoothFactor(0.15f)
    , accelerationSmoothFactor(0.15f)
    , maxPredictionTime(0.1f)
    , previousErrorX(0.0f)
    , previousErrorY(0.0f)
    , previousVelocityX(0.0f)
    , previousVelocityY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
{}

void DerivativePredictor::update(float errorX, float errorY, float prevMoveX, float prevMoveY, float deltaTime)
{
    if (deltaTime <= 0.0f) return;
    
    // 速度计算：考虑上次移动量（更准确的速度估计）
    float rawVelX = ((errorX - previousErrorX) + prevMoveX) / deltaTime;
    float rawVelY = ((errorY - previousErrorY) + prevMoveY) / deltaTime;
    
    // 方向过滤：如果速度方向与误差方向相反，置零
    if ((rawVelX > 0 && errorX < 0) || (rawVelX < 0 && errorX > 0)) {
        rawVelX = 0.0f;
    }
    if ((rawVelY > 0 && errorY < 0) || (rawVelY < 0 && errorY > 0)) {
        rawVelY = 0.0f;
    }
    
    // 指数平滑
    velocityX = velocitySmoothFactor * rawVelX + (1.0f - velocitySmoothFactor) * velocityX;
    velocityY = velocitySmoothFactor * rawVelY + (1.0f - velocitySmoothFactor) * velocityY;
    
    // 加速度计算
    float rawAccX = (velocityX - previousVelocityX) / deltaTime;
    float rawAccY = (velocityY - previousVelocityY) / deltaTime;
    
    // 方向过滤：如果加速度方向与误差方向相反，置零
    if ((rawAccX > 0 && errorX < 0) || (rawAccX < 0 && errorX > 0)) {
        rawAccX = 0.0f;
    }
    if ((rawAccY > 0 && errorY < 0) || (rawAccY < 0 && errorY > 0)) {
        rawAccY = 0.0f;
    }
    
    // 指数平滑
    accelerationX = accelerationSmoothFactor * rawAccX + (1.0f - accelerationSmoothFactor) * accelerationX;
    accelerationY = accelerationSmoothFactor * rawAccY + (1.0f - accelerationSmoothFactor) * accelerationY;
    
    // 保存状态
    previousErrorX = errorX;
    previousErrorY = errorY;
    previousVelocityX = velocityX;
    previousVelocityY = velocityY;
    previousMoveX = prevMoveX;
    previousMoveY = prevMoveY;
}

void DerivativePredictor::predict(float predictionTime, float& predictedX, float& predictedY)
{
    predictionTime = std::min(predictionTime, maxPredictionTime);
    
    predictedX = velocityX * predictionTime + 0.5f * accelerationX * predictionTime * predictionTime;
    predictedY = velocityY * predictionTime + 0.5f * accelerationY * predictionTime * predictionTime;
}

void DerivativePredictor::reset()
{
    velocityX = 0.0f;
    velocityY = 0.0f;
    accelerationX = 0.0f;
    accelerationY = 0.0f;
    previousErrorX = 0.0f;
    previousErrorY = 0.0f;
    previousVelocityX = 0.0f;
    previousVelocityY = 0.0f;
    previousMoveX = 0.0f;
    previousMoveY = 0.0f;
}

void DerivativePredictor::setSmoothFactors(float velocitySmooth, float accelerationSmooth)
{
    velocitySmoothFactor = velocitySmooth;
    accelerationSmoothFactor = accelerationSmooth;
}

void DerivativePredictor::setMaxPredictionTime(float maxTime)
{
    maxPredictionTime = maxTime;
}

#endif
