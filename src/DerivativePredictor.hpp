#ifndef DERIVATIVE_PREDICTOR_HPP
#define DERIVATIVE_PREDICTOR_HPP

class DerivativePredictor {
private:
    float velocityX;
    float velocityY;
    float accelerationX;
    float accelerationY;
    float velocitySmoothFactor;
    float accelerationSmoothFactor;
    float maxPredictionTime;
    float previousErrorX;
    float previousErrorY;
    float previousVelocityX;
    float previousVelocityY;

public:
    DerivativePredictor();
    void update(float errorX, float errorY, float deltaTime);
    void predict(float predictionTime, float& predictedX, float& predictedY);
    void reset();

    // 动态更新参数
    void setVelocitySmoothFactor(float factor) { velocitySmoothFactor = factor; }
    void setAccelerationSmoothFactor(float factor) { accelerationSmoothFactor = factor; }
    void setMaxPredictionTime(float time) { maxPredictionTime = time; }
};

#endif
