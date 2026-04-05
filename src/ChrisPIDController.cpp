#ifdef _WIN32

#include "ChrisPIDController.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <obs-module.h>

ChrisDerivativePredictor::ChrisDerivativePredictor()
    : smoothed_vel_({0.0f, 0.0f})
    , smoothed_acc_({0.0f, 0.0f})
    , prev_smoothed_vel_({0.0f, 0.0f})
{
}

std::array<float, 2> ChrisDerivativePredictor::predict(
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
    
    vel_raw[0] = std::clamp(vel_raw[0], -MAX_VEL, MAX_VEL);
    vel_raw[1] = std::clamp(vel_raw[1], -MAX_VEL, MAX_VEL);
    
    for (int axis = 0; axis < 2; ++axis) {
        if (std::abs(curr_e[axis]) > 5.0f) {
            if (std::signbit(vel_raw[axis]) != std::signbit(curr_e[axis])) {
                vel_raw[axis] *= 0.1f;
            }
        }
    }
    
    float adj_alpha_vel = 1.0f - static_cast<float>(std::pow(1.0 - ALPHA_VEL, dt / 0.01));
    adj_alpha_vel = std::clamp(adj_alpha_vel, 0.05f, 0.8f);
    
    prev_smoothed_vel_ = smoothed_vel_;
    
    for (int axis = 0; axis < 2; ++axis) {
        smoothed_vel_[axis] = adj_alpha_vel * vel_raw[axis] + (1.0f - adj_alpha_vel) * smoothed_vel_[axis];
    }
    
    std::array<float, 2> acc_raw = {
        (smoothed_vel_[0] - prev_smoothed_vel_[0]) / dt,
        (smoothed_vel_[1] - prev_smoothed_vel_[1]) / dt
    };
    
    acc_raw[0] = std::clamp(acc_raw[0], -MAX_ACC, MAX_ACC);
    acc_raw[1] = std::clamp(acc_raw[1], -MAX_ACC, MAX_ACC);
    
    for (int axis = 0; axis < 2; ++axis) {
        if (std::abs(curr_e[axis]) > 5.0f) {
            if (std::signbit(acc_raw[axis]) != std::signbit(curr_e[axis])) {
                acc_raw[axis] *= 0.1f;
            }
        }
    }
    
    float adj_alpha_acc = 1.0f - static_cast<float>(std::pow(1.0 - ALPHA_ACC, dt / 0.01));
    adj_alpha_acc = std::clamp(adj_alpha_acc, 0.05f, 0.8f);
    
    for (int axis = 0; axis < 2; ++axis) {
        smoothed_acc_[axis] = adj_alpha_acc * acc_raw[axis] + (1.0f - adj_alpha_acc) * smoothed_acc_[axis];
    }
    
    return {
        smoothed_vel_[0] * dt + 0.5f * smoothed_acc_[0] * dt * dt,
        smoothed_vel_[1] * dt + 0.5f * smoothed_acc_[1] * dt * dt
    };
}

void ChrisDerivativePredictor::reset()
{
    smoothed_vel_.fill(0.0f);
    smoothed_acc_.fill(0.0f);
    prev_smoothed_vel_.fill(0.0f);
}

ChrisAimController::ChrisAimController()
    : last_time_(0.0)
    , lock_start_time_(0.0)
{
    i_term_.fill(0.0f);
    last_error_.fill(0.0f);
    last_raw_error_.fill(0.0f);
    last_output_.fill(0.0f);
}

void ChrisAimController::setConfig(const ChrisPIDConfig& config)
{
    config_ = config;
}

void ChrisAimController::update(float raw_dx, float raw_dy, double current_time, float& out_x, float& out_y)
{
    float dt = 0.01f;
    if (last_time_ > 0.0) {
        dt = static_cast<float>(current_time - last_time_);
    }
    dt = std::clamp(dt, 0.001f, 0.05f);
    last_time_ = current_time;
    
    std::array<float, 2> curr_raw_error = {raw_dx, raw_dy};
    
    std::array<float, 2> pred_displacement = predictor_.predict(
        curr_raw_error,
        last_raw_error_,
        last_output_,
        dt
    );
    
    float max_pred_allowed_x = std::min(std::max(std::abs(raw_dx) * 1.5f, 30.0f), 60.0f);
    float max_pred_allowed_y = std::min(std::max(std::abs(raw_dy) * 1.5f, 30.0f), 60.0f);
    
    if (std::abs(pred_displacement[0]) > max_pred_allowed_x || 
        std::abs(pred_displacement[1]) > max_pred_allowed_y) {
        pred_displacement[0] = std::clamp(pred_displacement[0], -max_pred_allowed_x, max_pred_allowed_x);
        pred_displacement[1] = std::clamp(pred_displacement[1], -max_pred_allowed_y, max_pred_allowed_y);
        
        if (std::abs(pred_displacement[0]) > 100.0f || std::abs(pred_displacement[1]) > 100.0f) {
            predictor_.reset();
            pred_displacement.fill(0.0f);
        }
    }
    
    float fusion_error_x = raw_dx + pred_displacement[0] * config_.predWeightX;
    float fusion_error_y = raw_dy + pred_displacement[1] * config_.predWeightY;
    
    if (lock_start_time_ <= 0.0) {
        lock_start_time_ = current_time;
    }
    
    float scale = 1.0f;
    double elapsed = current_time - lock_start_time_;
    if (config_.rampTime > 0.0f && elapsed < config_.rampTime) {
        float progress = static_cast<float>(elapsed / config_.rampTime);
        scale = config_.initScale + (1.0f - config_.initScale) * progress;
    }
    
    float real_kp = config_.kp * scale;
    
    float p_term_x = fusion_error_x * real_kp;
    float p_term_y = fusion_error_y * real_kp;
    
    float i_increment_x = fusion_error_x * dt * config_.ki;
    float i_increment_y = fusion_error_y * dt * config_.ki;
    
    i_term_[0] += i_increment_x;
    i_term_[1] += i_increment_y;
    
    i_term_[0] = std::clamp(i_term_[0], -config_.iMax, config_.iMax);
    i_term_[1] = std::clamp(i_term_[1], -config_.iMax, config_.iMax);
    
    float d_term_x = 0.0f;
    float d_term_y = 0.0f;
    if (dt > 1e-6f) {
        d_term_x = (fusion_error_x - last_error_[0]) / dt * config_.kd;
        d_term_y = (fusion_error_y - last_error_[1]) / dt * config_.kd;
    }
    d_term_x = std::clamp(d_term_x, -50.0f, 50.0f);
    d_term_y = std::clamp(d_term_y, -50.0f, 50.0f);
    
    float output_x = p_term_x + i_term_[0] + d_term_x;
    float output_y = p_term_y + i_term_[1] + d_term_y;
    
    output_x = std::clamp(output_x, -config_.outputMax, config_.outputMax);
    output_y = std::clamp(output_y, -config_.outputMax, config_.outputMax);
    
    // 详细日志输出（每30帧输出一次）
    static int logCounter = 0;
    if (++logCounter >= 30) {
        logCounter = 0;
        blog(LOG_INFO, "[ChrisPID] dt=%.4f | error=(%.1f,%.1f) | fusion=(%.1f,%.1f) | "
             "P=(%.2f,%.2f) | I=(%.4f,%.4f) inc=(%.4f,%.4f) | D=(%.2f,%.2f) | "
             "output=(%.2f,%.2f) | kp=%.3f ki=%.4f kd=%.3f scale=%.2f",
             dt, raw_dx, raw_dy, fusion_error_x, fusion_error_y,
             p_term_x, p_term_y, i_term_[0], i_term_[1], i_increment_x, i_increment_y,
             d_term_x, d_term_y, output_x, output_y,
             config_.kp, config_.ki, config_.kd, scale);
    }
    
    last_error_[0] = fusion_error_x;
    last_error_[1] = fusion_error_y;
    last_raw_error_ = curr_raw_error;
    last_output_ = {output_x, output_y};
    
    out_x = output_x;
    out_y = output_y;
}

void ChrisAimController::reset()
{
    predictor_.reset();
    i_term_.fill(0.0f);
    last_error_.fill(0.0f);
    last_raw_error_.fill(0.0f);
    last_output_.fill(0.0f);
    last_time_ = 0.0;
    lock_start_time_ = 0.0;
}

#endif // _WIN32
