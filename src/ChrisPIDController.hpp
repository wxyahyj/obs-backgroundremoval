#ifndef CHRIS_PID_CONTROLLER_HPP
#define CHRIS_PID_CONTROLLER_HPP

#ifdef _WIN32

#include <array>
#include <cmath>
#include <chrono>
#include <algorithm>

struct ChrisPIDConfig {
    float kp = 0.45f;
    float ki = 0.02f;
    float kd = 0.04f;
    float predWeightX = 0.5f;
    float predWeightY = 0.1f;
    float initScale = 0.6f;
    float rampTime = 0.5f;
    float outputMax = 150.0f;
    float iMax = 100.0f;
};

class ChrisDerivativePredictor {
public:
    static constexpr float ALPHA_VEL = 0.25f;
    static constexpr float ALPHA_ACC = 0.15f;
    static constexpr float MAX_VEL = 3000.0f;
    static constexpr float MAX_ACC = 5000.0f;

    ChrisDerivativePredictor();
    
    std::array<float, 2> predict(
        const std::array<float, 2>& curr_e,
        const std::array<float, 2>& prev_e,
        const std::array<float, 2>& prev_m,
        float dt
    );
    
    void reset();

private:
    std::array<float, 2> smoothed_vel_;
    std::array<float, 2> smoothed_acc_;
    std::array<float, 2> prev_smoothed_vel_;
};

class ChrisAimController {
public:
    ChrisAimController();
    
    void setConfig(const ChrisPIDConfig& config);
    void update(float raw_dx, float raw_dy, double current_time, float& out_x, float& out_y);
    void reset();

private:
    ChrisPIDConfig config_;
    ChrisDerivativePredictor predictor_;
    
    std::array<float, 2> i_term_;
    std::array<float, 2> last_error_;
    std::array<float, 2> last_raw_error_;
    std::array<float, 2> last_output_;
    
    double last_time_;
    double lock_start_time_;
};

#endif // _WIN32

#endif // CHRIS_PID_CONTROLLER_HPP
