#ifdef _WIN32

#include "IncrementalPIDController.hpp"
#include <obs-module.h>

IncrementalPIDAdapter::IncrementalPIDAdapter()
{
    chainConfig_.profile_a.enabled = true;
    chainConfig_.profile_a.hotkey_down = true;
    chainConfig_.profile_a.class_id = 0;
    chainConfig_.profile_a.horizontal_ratio = 0.5f;
    chainConfig_.profile_a.vertical_ratio = 0.5f;
}

void IncrementalPIDAdapter::setConfig(const IncrementalPIDConfig& config)
{
    config_ = config;
    
    chain_.configure_pid(config.kp, config.ki, config.kd);
    
    chainConfig_.speed_x = config.speedX;
    chainConfig_.speed_y = config.speedY;
    chainConfig_.aim_radius = config.aimRadius;
    chainConfig_.jitter_enabled = config.jitterEnabled;
    chainConfig_.pid_enabled = config.pidEnabled;
    chainConfig_.center_x = config.centerX;
    chainConfig_.center_y = config.centerY;
    
    chainConfig_.side_compensation.enabled = config.sideCompEnabled;
    chainConfig_.side_compensation.hotkey_down = config.sideCompEnabled;
    chainConfig_.side_compensation.cap = config.sideCompCap;
    chainConfig_.side_compensation.denominator = config.sideCompDenom;
}

void IncrementalPIDAdapter::update(float errorX, float errorY, float& outX, float& outY)
{
    mist::reconstructed::Detection det;
    det.x = static_cast<int>(-errorX);
    det.y = static_cast<int>(-errorY);
    det.width = 1;
    det.height = 1;
    det.class_id = 0;
    
    std::vector<mist::reconstructed::Detection> detections = {det};
    
    auto result = chain_.process(chainConfig_, detections);
    
    if (result.has_target && result.emitted_move) {
        outX = static_cast<float>(result.output_move.x);
        outY = static_cast<float>(result.output_move.y);
    } else {
        outX = 0.0f;
        outY = 0.0f;
    }
    
    // 计算增量式PID的各项：output += Kp*(e - e1) + Ki*e + Kd*(e - 2*e1 + e2)
    const auto& pid = chain_.pid();
    float e = static_cast<float>(result.selected_error.x);
    float e1 = pid.previous_error();
    float e2 = pid.previous_previous_error();
    
    lastDebugTerms_.pTerm = pid.kp() * (e - e1);
    lastDebugTerms_.iTerm = pid.ki() * e;
    lastDebugTerms_.dTerm = pid.kd() * (e - 2.0f * e1 + e2);
    lastDebugTerms_.pidOutput = pid.output();
    lastDebugTerms_.lastDistance = chain_.last_distance();
    lastDebugTerms_.previousOutputX = chain_.previous_output_x();
    
    static int logCounter = 0;
    if (++logCounter >= 30) {
        logCounter = 0;
        blog(LOG_INFO, "[IncrementalPID] error=(%.1f,%.1f) | out=(%.1f,%.1f) | P=%.2f I=%.2f D=%.2f | dist=%d",
             errorX, errorY, outX, outY, 
             lastDebugTerms_.pTerm, lastDebugTerms_.iTerm, lastDebugTerms_.dTerm,
             lastDebugTerms_.lastDistance);
    }
}

void IncrementalPIDAdapter::reset()
{
    chain_.reset_runtime();
    lastDebugTerms_ = DebugTerms{};
}

#endif
