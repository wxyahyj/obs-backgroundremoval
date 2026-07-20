#ifndef ADAPTIVE_PID_CONTROLLER_HPP
#define ADAPTIVE_PID_CONTROLLER_HPP

#ifdef _WIN32

#include <cmath>
#include <algorithm>

/// 自适应位置式PID控制器
/// 核心特性：
/// 1. 位置式PID：output = Kp*e + Ki*∫e + Kd*de/dt
/// 2. 输入死区：小误差置零，避免微小抖动
/// 3. 积分分离：根据误差大小自适应调整积分增益
/// 4. 积分死区：积分累积小于阈值时忽略
/// 5. 积分限幅 + 输出限幅：双重抗饱和
class AdaptivePIDController {
public:
    struct Config {
        float kp = 1.0f;
        float ki = 0.1f;
        float kd = 0.05f;
        float deadZone = 0.3f;               // 输入死区阈值
        float integralLimit = 100.0f;         // 积分限幅
        float integralDeadzone = 1.0f;        // 积分死区阈值
        float integralGainThreshold = 50.0f;  // 积分自适应阈值
        float integralGainRate = 0.015f;      // 积分自适应速率
        float outputLimit = 10.0f;            // 输出限幅
    };

    AdaptivePIDController() = default;

    void configure(const Config& cfg) {
        cfg_ = cfg;
    }

    void reset() {
        integral_ = 0.0f;
        lastError_ = 0.0f;
        integralGain_ = 0.0f;
    }

    float update(float error) {
        // 输入死区处理
        if (std::abs(error) < cfg_.deadZone) {
            error = 0.0f;
        }

        // 比例项
        float pOut = cfg_.kp * error;

        // 自适应积分增益
        float gain = adjustIntegralGain(error, lastError_);
        if (gain > 0.0f) {
            integral_ += error;
            integral_ = std::clamp(integral_, -cfg_.integralLimit, cfg_.integralLimit);
        } else {
            integral_ = 0.0f;
        }

        // 积分项 + 积分死区
        float iOut = (std::abs(integral_) > cfg_.integralDeadzone)
            ? cfg_.ki * integral_ : 0.0f;

        // 微分项
        float dOut = cfg_.kd * (error - lastError_);

        // 总输出
        float output = pOut + iOut + dOut;

        // 输出限幅
        output = std::clamp(output, -cfg_.outputLimit, cfg_.outputLimit);

        // 状态更新
        lastError_ = error;

        return output;
    }

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    float integral_ = 0.0f;
    float lastError_ = 0.0f;
    float integralGain_ = 0.0f;

    /// 自适应积分增益：接近目标时积分增强，远离时衰减
    float adjustIntegralGain(float error, float lastError) {
        float errorDerivative = std::abs(error - lastError);

        if (std::abs(error) < cfg_.integralGainThreshold) {
            float adaptRate = cfg_.integralGainRate
                * (1.0f - errorDerivative / (cfg_.integralGainThreshold * 2.0f));
            adaptRate = std::clamp(adaptRate, 0.0f, cfg_.integralGainRate);
            integralGain_ = std::min(integralGain_ + adaptRate, 1.0f);
        } else {
            float decay = 0.1f + 0.9f * std::tanh(std::abs(error) / (cfg_.integralGainThreshold * 2.0f));
            integralGain_ *= (1.0f - 0.05f * decay);
        }
        integralGain_ = std::clamp(integralGain_, 0.0f, 1.0f);
        return integralGain_;
    }
};

#endif // _WIN32
#endif // ADAPTIVE_PID_CONTROLLER_HPP