#ifndef DOPA_PID_CONTROLLER_HPP
#define DOPA_PID_CONTROLLER_HPP

#ifdef _WIN32

#include <array>
#include <cmath>
#include <deque>
#include <chrono>
#include <algorithm>

struct DopaPIDConfig {
    float kpX = 0.8f;
    float kpY = 0.6f;
    float kiX = 0.005f;
    float kiY = 0.004f;
    float kdX = 0.025f;
    float kdY = 0.03f;
    float windupGuardX = 50.0f;
    float windupGuardY = 50.0f;
    float outputLimitMinX = -50.0f;
    float outputLimitMaxX = 50.0f;
    float outputLimitMinY = -50.0f;
    float outputLimitMaxY = 50.0f;
    float predWeight = 0.8f;
    int gameFps = 60;
    float smoothX = 0.0f;
    float smoothY = 0.0f;
    float smoothDeadzone = 2.0f;
    float smoothAlgorithm = 1.0f;
    std::string antiWindupMode = "freeze";
    float backcalcGainX = 0.0f;
    float backcalcGainY = 0.0f;
};

class DopaDerivativePredictor {
public:
    static constexpr float ALPHA_VEL = 0.15f;
    static constexpr float ALPHA_ACC = 0.15f;

    DopaDerivativePredictor();
    
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

class DopaDualAxisPID {
public:
    DopaDualAxisPID();
    
    void setConfig(const DopaPIDConfig& config);
    void compute(float target_x, float target_y, float crosshair_x, float crosshair_y,
                 float& out_x, float& out_y);
    void reset();
    
    std::array<float, 3> getComponents(const std::string& axis) const;

private:
    DopaPIDConfig config_;
    DopaDerivativePredictor predictor_;
    
    float pred_dt_;
    
    std::array<float, 2> last_raw_error_;
    std::array<float, 2> last_pid_output_;
    std::array<float, 2> last_error_;
    std::array<float, 2> p_term_;
    std::array<float, 2> i_term_;
    std::array<float, 2> d_term_;
    std::array<float, 2> last_integral_increment_;
    
    std::deque<std::array<float, 2>> error_history_;
    std::deque<double> time_history_;
    
    static constexpr size_t HISTORY_SIZE = 20;
    static constexpr float UNIFORM_THRESHOLD = 1.5f;
    static constexpr float MIN_VELOCITY_THRESHOLD = 10.0f;
    static constexpr float MAX_VELOCITY_THRESHOLD = 100.0f;
    static constexpr float COMPENSATION_FACTOR = 2.0f;
    
    float calculateOutput(int axis, float error, float dt);
    void applySmoothing(float& x_output, float& y_output, float error_x, float error_y);
    float applyLimitsAndAntiWindup(int axis, float unsat_value);
    bool isUniformMotion();
};

#endif // _WIN32

#endif // DOPA_PID_CONTROLLER_HPP
