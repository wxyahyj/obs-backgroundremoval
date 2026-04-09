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
    if (deltaTime <= 1e-6f) return;
    
    float rawVelX = ((errorX - previousErrorX) + prevMoveX) / deltaTime;
    float rawVelY = ((errorY - previousErrorY) + prevMoveY) / deltaTime;
    
    rawVelX = std::clamp(rawVelX, -MAX_VEL, MAX_VEL);
    rawVelY = std::clamp(rawVelY, -MAX_VEL, MAX_VEL);
    
    for (int axis = 0; axis < 2; ++axis) {
        float& vel = (axis == 0) ? rawVelX : rawVelY;
        float err = (axis == 0) ? errorX : errorY;
        
        if (std::abs(err) > 5.0f) {
            if (std::signbit(vel) != std::signbit(err)) {
                vel *= 0.1f;
            }
        }
    }
    
    float adjAlphaVel = 1.0f - static_cast<float>(std::exp(-deltaTime / 0.01));
    adjAlphaVel = std::clamp(adjAlphaVel, 0.05f, 0.8f);
    
    float prevVelX = velocityX;
    float prevVelY = velocityY;
    
    velocityX = adjAlphaVel * rawVelX + (1.0f - adjAlphaVel) * velocityX;
    velocityY = adjAlphaVel * rawVelY + (1.0f - adjAlphaVel) * velocityY;
    
    float rawAccX = (velocityX - prevVelX) / deltaTime;
    float rawAccY = (velocityY - prevVelY) / deltaTime;
    
    rawAccX = std::clamp(rawAccX, -MAX_ACC, MAX_ACC);
    rawAccY = std::clamp(rawAccY, -MAX_ACC, MAX_ACC);
    
    for (int axis = 0; axis < 2; ++axis) {
        float& acc = (axis == 0) ? rawAccX : rawAccY;
        float err = (axis == 0) ? errorX : errorY;
        
        if (std::abs(err) > 5.0f) {
            if (std::signbit(acc) != std::signbit(err)) {
                acc *= 0.1f;
            }
        }
    }
    
    float adjAlphaAcc = 1.0f - static_cast<float>(std::exp(-deltaTime / 0.01));
    adjAlphaAcc = std::clamp(adjAlphaAcc, 0.05f, 0.8f);
    
    accelerationX = adjAlphaAcc * rawAccX + (1.0f - adjAlphaAcc) * accelerationX;
    accelerationY = adjAlphaAcc * rawAccY + (1.0f - adjAlphaAcc) * accelerationY;
    
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
