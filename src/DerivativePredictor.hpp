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
    float previousMoveX;
    float previousMoveY;

public:
    DerivativePredictor();
    void update(float errorX, float errorY, float previousMoveX, float previousMoveY, float deltaTime);
    void predict(float predictionTime, float& predictedX, float& predictedY);
    void reset();
};

#endif
