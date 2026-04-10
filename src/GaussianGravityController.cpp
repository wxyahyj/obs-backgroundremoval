#ifdef _WIN32

#include "GaussianGravityController.hpp"
#include <obs-module.h>

GaussianGravityController::GaussianGravityController()
    : last_output_({0.0f, 0.0f})
{
}

void GaussianGravityController::setConfig(const GaussianGravityConfig& config)
{
    config_ = config;
}

float GaussianGravityController::calculateGravityForce(float distance)
{
    if (distance >= config_.maxDistance) {
        return 0.0f;
    }
    
    // 线性引力公式：F = G * (1 - distance / maxDistance)
    float force = config_.gravityStrength * (1.0f - distance / config_.maxDistance);
    
    return force;
}

void GaussianGravityController::compute(float target_x, float target_y, float crosshair_x, float crosshair_y,
                                        float target_confidence, float velocity_x, float velocity_y,
                                        float& out_x, float& out_y)
{
    // 计算误差向量
    float error_x = target_x - crosshair_x;
    float error_y = target_y - crosshair_y;
    
    // 计算距离
    float distance = std::sqrt(error_x * error_x + error_y * error_y);
    
    if (distance < 1.0f) {
        // 已经很接近，输出小力度
        out_x = error_x * 0.5f;
        out_y = error_y * 0.5f;
        return;
    }
    
    // 计算引力大小
    float force = calculateGravityForce(distance);
    
    // 按置信度缩放
    if (config_.confidenceScale) {
        force *= target_confidence;
    }
    
    // 添加速度预测（如果目标在移动，增加引力）
    float velocity_magnitude = std::sqrt(velocity_x * velocity_x + velocity_y * velocity_y);
    if (velocity_magnitude > 0.1f) {
        float velocity_factor = 1.0f + config_.predictionWeight * (velocity_magnitude / 100.0f);
        force *= velocity_factor;
    }
    
    // 限制最大力
    force = std::clamp(force, -config_.maxForce, config_.maxForce);
    
    // 计算方向单位向量
    float dir_x = error_x / distance;
    float dir_y = error_y / distance;
    
    // 计算输出
    float raw_out_x = force * dir_x;
    float raw_out_y = force * dir_y;
    
    // 平滑输出
    out_x = config_.smoothingFactor * raw_out_x + (1.0f - config_.smoothingFactor) * last_output_[0];
    out_y = config_.smoothingFactor * raw_out_y + (1.0f - config_.smoothingFactor) * last_output_[1];
    
    // 保存输出
    last_output_[0] = out_x;
    last_output_[1] = out_y;
    
    // 日志输出（每30帧）
    static int log_counter = 0;
    if (++log_counter >= 30) {
        log_counter = 0;
        blog(LOG_INFO, "[GaussianGravity] distance=%.1f | force=%.2f | velocity=%.2f | out=(%.2f,%.2f)",
             distance, force, velocity_magnitude, out_x, out_y);
    }
}

void GaussianGravityController::reset()
{
    last_output_.fill(0.0f);
}

#endif // _WIN32
