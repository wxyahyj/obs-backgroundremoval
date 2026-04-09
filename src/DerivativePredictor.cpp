#ifdef _WIN32

#include "DerivativePredictor.hpp"
#include <algorithm>
#include <cmath>

DerivativePredictor::DerivativePredictor()
    : velocityX(0.0f)
    , velocityY(0.0f)
    , accelerationX(0.0f)
    , accelerationY(0.0f)
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
    
    float rawVelX = ((errorX - previousErrorX) + prevMoveX) / deltaTime;
    float rawVelY = ((errorY - previousErrorY) + prevMoveY) / deltaTime;
    
    if ((rawVelX > 0 && errorX < 0) || (rawVelX < 0 && errorX > 0)) {
        rawVelX = 0.0f;
    }
    if ((rawVelY > 0 && errorY < 0) || (rawVelY < 0 && errorY > 0)) {
        rawVelY = 0.0f;
    }
    
    velocityX = ALPHA_VEL * rawVelX + (1.0f - ALPHA_VEL) * velocityX;
    velocityY = ALPHA_VEL * rawVelY + (1.0f - ALPHA_VEL) * velocityY;
    
    float rawAccX = (velocityX - previousVelocityX) / deltaTime;
    float rawAccY = (velocityY - previousVelocityY) / deltaTime;
    
    if ((rawAccX > 0 && errorX < 0) || (rawAccX < 0 && errorX > 0)) {
        rawAccX = 0.0f;
    }
    if ((rawAccY > 0 && errorY < 0) || (rawAccY < 0 && errorY > 0)) {
        rawAccY = 0.0f;
    }
    
    accelerationX = ALPHA_ACC * rawAccX + (1.0f - ALPHA_ACC) * accelerationX;
    accelerationY = ALPHA_ACC * rawAccY + (1.0f - ALPHA_ACC) * accelerationY;
    
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

void DerivativePredictor::setMaxPredictionTime(float maxTime)
{
    maxPredictionTime = maxTime;
}

#endif
