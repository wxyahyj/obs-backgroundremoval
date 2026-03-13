#ifdef _WIN32

#include "DerivativePredictor.hpp"
#include <algorithm>

DerivativePredictor::DerivativePredictor()
    : velocityX(0.0f)
    , velocityY(0.0f)
    , accelerationX(0.0f)
    , accelerationY(0.0f)
    , velocitySmoothFactor(0.7f)
    , accelerationSmoothFactor(0.5f)
    , maxPredictionTime(0.1f)
    , previousErrorX(0.0f)
    , previousErrorY(0.0f)
    , previousVelocityX(0.0f)
    , previousVelocityY(0.0f)
{}

void DerivativePredictor::update(float errorX, float errorY, float deltaTime)
{
    if (deltaTime <= 0.0f) return;
    
    float newVelocityX = (errorX - previousErrorX) / deltaTime;
    float newVelocityY = (errorY - previousErrorY) / deltaTime;
    
    velocityX = velocitySmoothFactor * velocityX + (1.0f - velocitySmoothFactor) * newVelocityX;
    velocityY = velocitySmoothFactor * velocityY + (1.0f - velocitySmoothFactor) * newVelocityY;
    
    float newAccelerationX = (velocityX - previousVelocityX) / deltaTime;
    float newAccelerationY = (velocityY - previousVelocityY) / deltaTime;
    
    accelerationX = accelerationSmoothFactor * accelerationX + (1.0f - accelerationSmoothFactor) * newAccelerationX;
    accelerationY = accelerationSmoothFactor * accelerationY + (1.0f - accelerationSmoothFactor) * newAccelerationY;
    
    previousErrorX = errorX;
    previousErrorY = errorY;
    previousVelocityX = velocityX;
    previousVelocityY = velocityY;
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
}

#endif
