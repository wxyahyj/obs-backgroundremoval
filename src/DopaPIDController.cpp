#ifdef _WIN32

#include "DopaPIDController.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <obs-module.h>

DopaDerivativePredictor::DopaDerivativePredictor()
    : smoothed_vel_({0.0f, 0.0f})
    , smoothed_acc_({0.0f, 0.0f})
    , prev_smoothed_vel_({0.0f, 0.0f})
{
}

std::array<float, 2> DopaDerivativePredictor::predict(
    const std::array<float, 2>& curr_e,
    const std::array<float, 2>& prev_e,
    const std::array<float, 2>& prev_m,
    float dt)
{
    if (dt <= 1e-6f) {
        return {0.0f, 0.0f};
    }
    
    std::array<float, 2> vel_raw = {
        ((curr_e[0] - prev_e[0]) + prev_m[0]) / dt,
        ((curr_e[1] - prev_e[1]) + prev_m[1]) / dt
    };
    
    for (int axis = 0; axis < 2; ++axis) {
        if (std::signbit(vel_raw[axis]) != std::signbit(curr_e[axis]) && curr_e[axis] != 0.0f) {
            vel_raw[axis] = 0.0f;
        }
    }
    
    prev_smoothed_vel_ = smoothed_vel_;
    
    for (int axis = 0; axis < 2; ++axis) {
        smoothed_vel_[axis] = ALPHA_VEL * vel_raw[axis] + (1.0f - ALPHA_VEL) * smoothed_vel_[axis];
    }
    
    std::array<float, 2> acc_raw = {
        (smoothed_vel_[0] - prev_smoothed_vel_[0]) / dt,
        (smoothed_vel_[1] - prev_smoothed_vel_[1]) / dt
    };
    
    for (int axis = 0; axis < 2; ++axis) {
        if (std::signbit(acc_raw[axis]) != std::signbit(curr_e[axis]) && curr_e[axis] != 0.0f) {
            acc_raw[axis] = 0.0f;
        }
    }
    
    for (int axis = 0; axis < 2; ++axis) {
        smoothed_acc_[axis] = ALPHA_ACC * acc_raw[axis] + (1.0f - ALPHA_ACC) * smoothed_acc_[axis];
    }
    
    return {
        smoothed_vel_[0] * dt + 0.5f * smoothed_acc_[0] * dt * dt,
        smoothed_vel_[1] * dt + 0.5f * smoothed_acc_[1] * dt * dt
    };
}

void DopaDerivativePredictor::reset()
{
    smoothed_vel_.fill(0.0f);
    smoothed_acc_.fill(0.0f);
    prev_smoothed_vel_.fill(0.0f);
}

DopaDualAxisPID::DopaDualAxisPID()
    : pred_dt_(1.0f / 60.0f)
{
    last_raw_error_.fill(0.0f);
    last_pid_output_.fill(0.0f);
    last_error_.fill(0.0f);
    p_term_.fill(0.0f);
    i_term_.fill(0.0f);
    d_term_.fill(0.0f);
    last_integral_increment_.fill(0.0f);
}

void DopaDualAxisPID::setConfig(const DopaPIDConfig& config)
{
    config_ = config;
    pred_dt_ = 1.0f / static_cast<float>(config.gameFps);
}

void DopaDualAxisPID::compute(float target_x, float target_y, float crosshair_x, float crosshair_y,
                               float& out_x, float& out_y)
{
    float raw_error_x = target_x - crosshair_x;
    float raw_error_y = target_y - crosshair_y;
    
    std::array<float, 2> curr_raw_error = {raw_error_x, raw_error_y};
    
    std::array<float, 2> pred_displacement = predictor_.predict(
        curr_raw_error,
        last_raw_error_,
        last_pid_output_,
        pred_dt_
    );
    
    float fusion_error_x = raw_error_x + pred_displacement[0] * config_.predWeight;
    float fusion_error_y = raw_error_y + pred_displacement[1] * config_.predWeight;
    
    float max_pred = std::max(std::abs(pred_displacement[0]), std::abs(pred_displacement[1]));
    if (max_pred > 100.0f) {
        fusion_error_x = raw_error_x;
        fusion_error_y = raw_error_y;
        predictor_.reset();
    }
    
    float x_output_unsat = calculateOutput(0, fusion_error_x, pred_dt_);
    float y_output_unsat = calculateOutput(1, fusion_error_y, pred_dt_);
    
    float x_output = applyLimitsAndAntiWindup(0, x_output_unsat);
    float y_output = applyLimitsAndAntiWindup(1, y_output_unsat);
    
    applySmoothing(x_output, y_output, fusion_error_x, fusion_error_y);
    
    float error_magnitude = std::sqrt(fusion_error_x * fusion_error_x + fusion_error_y * fusion_error_y);
    if (error_magnitude < 5.0f) {
        float deadzone_factor = std::max(0.1f, error_magnitude / 5.0f);
        x_output *= deadzone_factor;
        y_output *= deadzone_factor;
    }
    
    last_error_[0] = fusion_error_x;
    last_error_[1] = fusion_error_y;
    last_raw_error_ = curr_raw_error;
    last_pid_output_ = {x_output, y_output};
    
    // 详细日志输出（每30帧输出一次）
    static int logCounter = 0;
    if (++logCounter >= 30) {
        logCounter = 0;
        blog(LOG_INFO, "[DopaPID] dt=%.4f | error=(%.1f,%.1f) | fusion=(%.1f,%.1f) | "
             "P=(%.2f,%.2f) | I=(%.4f,%.4f) | D=(%.2f,%.2f) | "
             "output=(%.2f,%.2f) | kpX=%.3f kpY=%.3f kiX=%.4f kiY=%.4f",
             pred_dt_, raw_error_x, raw_error_y, fusion_error_x, fusion_error_y,
             p_term_[0], p_term_[1], i_term_[0], i_term_[1],
             d_term_[0], d_term_[1], x_output, y_output,
             config_.kpX, config_.kpY, config_.kiX, config_.kiY);
    }
    
    out_x = x_output;
    out_y = y_output;
}

float DopaDualAxisPID::calculateOutput(int axis, float error, float dt)
{
    float kp = (axis == 0) ? config_.kpX : config_.kpY;
    float ki = (axis == 0) ? config_.kiX : config_.kiY;
    float kd = (axis == 0) ? config_.kdX : config_.kdY;
    
    p_term_[axis] = kp * error;
    
    float integral_increment = ki * error * dt;
    i_term_[axis] += integral_increment;
    last_integral_increment_[axis] = integral_increment;
    
    float windup_guard = (axis == 0) ? config_.windupGuardX : config_.windupGuardY;
    if (windup_guard > 0.0f) {
        i_term_[axis] = std::clamp(i_term_[axis], -windup_guard, windup_guard);
    }
    
    if (dt > 1e-6f) {
        d_term_[axis] = kd * ((error - last_error_[axis]) / dt);
    } else {
        d_term_[axis] = 0.0f;
    }
    
    return p_term_[axis] + i_term_[axis] + d_term_[axis];
}

float DopaDualAxisPID::applyLimitsAndAntiWindup(int axis, float unsat_value)
{
    float value = unsat_value;
    bool saturated = false;
    
    float min_out = (axis == 0) ? config_.outputLimitMinX : config_.outputLimitMinY;
    float max_out = (axis == 0) ? config_.outputLimitMaxX : config_.outputLimitMaxY;
    
    if (value > max_out) {
        value = max_out;
        saturated = true;
    } else if (value < min_out) {
        value = min_out;
        saturated = true;
    }
    
    if (saturated) {
        float windup_guard = (axis == 0) ? config_.windupGuardX : config_.windupGuardY;
        float backcalc_gain = (axis == 0) ? config_.backcalcGainX : config_.backcalcGainY;
        
        if (config_.antiWindupMode == "backcalc") {
            i_term_[axis] += backcalc_gain * (value - unsat_value);
        } else {
            i_term_[axis] -= last_integral_increment_[axis];
        }
        
        i_term_[axis] = std::clamp(i_term_[axis], -windup_guard, windup_guard);
    }
    
    return value;
}

void DopaDualAxisPID::applySmoothing(float& x_output, float& y_output, float error_x, float error_y)
{
    float error_distance = std::sqrt(error_x * error_x + error_y * error_y);
    
    if (error_distance <= config_.smoothDeadzone) {
        return;
    }
    
    double current_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    error_history_.push_back({error_x, error_y});
    time_history_.push_back(current_time);
    
    float step_x = x_output;
    float step_y = y_output;
    
    if (isUniformMotion()) {
        float comp = std::max(1.0f, config_.smoothAlgorithm);
        x_output = step_x * comp;
        y_output = step_y * comp;
        return;
    }
    
    float dx_err = 0.0f;
    float dy_err = 0.0f;
    if (error_history_.size() >= 2) {
        auto last_error = error_history_[error_history_.size() - 2];
        dx_err = error_x - last_error[0];
        dy_err = error_y - last_error[1];
    }
    
    float s_x = std::clamp(config_.smoothX, 0.0f, 1.0f);
    float s_y = std::clamp(config_.smoothY, 0.0f, 1.0f);
    
    x_output = s_x * step_x + (1.0f - s_x) * dx_err;
    y_output = s_y * step_y + (1.0f - s_y) * dy_err;
}

bool DopaDualAxisPID::isUniformMotion()
{
    if (error_history_.size() < std::min(static_cast<size_t>(3), HISTORY_SIZE)) {
        return false;
    }
    
    std::vector<float> velocities;
    for (size_t i = 1; i < error_history_.size(); ++i) {
        double dt = time_history_[i] - time_history_[i - 1];
        if (dt <= 0.0) continue;
        
        float dx = error_history_[i][0] - error_history_[i - 1][0];
        float dy = error_history_[i][1] - error_history_[i - 1][1];
        float v = static_cast<float>(std::sqrt((dx / dt) * (dx / dt) + (dy / dt) * (dy / dt)));
        velocities.push_back(v);
    }
    
    if (velocities.empty()) {
        return false;
    }
    
    float avg_v = 0.0f;
    for (float v : velocities) {
        avg_v += v;
    }
    avg_v /= static_cast<float>(velocities.size());
    
    if (avg_v < MIN_VELOCITY_THRESHOLD || avg_v > MAX_VELOCITY_THRESHOLD) {
        return false;
    }
    
    float var_v = 0.0f;
    for (float v : velocities) {
        float diff = v - avg_v;
        var_v += diff * diff;
    }
    var_v /= static_cast<float>(velocities.size());
    
    return var_v < UNIFORM_THRESHOLD;
}

void DopaDualAxisPID::reset()
{
    predictor_.reset();
    last_raw_error_.fill(0.0f);
    last_pid_output_.fill(0.0f);
    last_error_.fill(0.0f);
    p_term_.fill(0.0f);
    i_term_.fill(0.0f);
    d_term_.fill(0.0f);
    last_integral_increment_.fill(0.0f);
    error_history_.clear();
    time_history_.clear();
}

void DopaDualAxisPID::resetPredictor()
{
    // 只重置预测器，不重置积分项
    // 这样在目标暂时丢失时，积分可以继续累积
    predictor_.reset();
    last_raw_error_.fill(0.0f);
    last_pid_output_.fill(0.0f);
    // 不重置 i_term_
}

std::array<float, 3> DopaDualAxisPID::getComponents(const std::string& axis) const
{
    if (axis == "x") {
        return {p_term_[0], i_term_[0], d_term_[0]};
    } else {
        return {p_term_[1], i_term_[1], d_term_[1]};
    }
}

#endif // _WIN32
