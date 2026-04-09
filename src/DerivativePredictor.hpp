#ifndef DERIVATIVE_PREDICTOR_HPP
#define DERIVATIVE_PREDICTOR_HPP

class DerivativePredictor {
private:
    float velocityX;
    float velocityY;
    float accelerationX;
    float accelerationY;
    float maxPredictionTime;
    float previousErrorX;
    float previousErrorY;
    float previousVelocityX;
    float previousVelocityY;
    float previousMoveX;
    float previousMoveY;

    static constexpr float ALPHA_VEL = 0.25f;
    static constexpr float ALPHA_ACC = 0.15f;
    static constexpr float MAX_VEL = 3000.0f;
    static constexpr float MAX_ACC = 5000.0f;

public:
    DerivativePredictor();
    void update(float errorX, float errorY, float previousMoveX, float previousMoveY, float deltaTime);
    void predict(float predictionTime, float& predictedX, float& predictedY);
    void reset();
    void setMaxPredictionTime(float maxTime);
};

#endif
