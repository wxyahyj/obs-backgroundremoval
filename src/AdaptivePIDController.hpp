#ifndef ADAPTIVE_PID_CONTROLLER_HPP
#define ADAPTIVE_PID_CONTROLLER_HPP

#ifdef _WIN32

#include <cmath>
#include <algorithm>

/// 自适应位置式PID控制器（dt-aware, 60Hz 兼容增益）
/// 核心特性：
/// 1. 位置式PID：output = Kp*e + Ki*∫e*dt + Kd*de/dt  （内部换算到连续域）
/// 2. 输入死区：小误差置零，避免微小抖动
/// 3. 积分分离：根据误差大小自适应调整积分增益（时间归一化）
/// 4. 积分死区：积分累积小于阈值时忽略
/// 5. 积分限幅 + 输出限幅：双重抗饱和
///
/// UI 中的 Ki/Kd 仍按「60Hz 每帧参数」理解；内部按 kReferenceFps 换算，
/// 使 60FPS 时行为与旧版一致，帧率不稳时保持时间一致性。
class AdaptivePIDController {
public:
    struct Config {
        float kp = 1.0f;
        float ki = 0.1f;   // 60Hz 等效每帧 Ki
        float kd = 0.05f;  // 60Hz 等效每帧 Kd
        float deadZone = 0.3f;
        float integralLimit = 100.0f;        // 60Hz 等效积分限幅（样本单位）
        float integralDeadzone = 1.0f;       // 60Hz 等效积分死区
        float integralGainThreshold = 50.0f;
        float integralGainRate = 0.015f;     // 60Hz 等效每帧上升速率
        float outputLimit = 10.0f;
        // 导数低通时间常数（秒）；0 = 关闭滤波
        float derivativeFilterTimeConstant = 0.0f;
    };

    static constexpr float kReferenceFps = 60.0f;
    static constexpr float kReferenceDt = 1.0f / kReferenceFps;
    static constexpr float kMinDt = 0.001f;
    static constexpr float kMaxDt = 0.05f;

    AdaptivePIDController() = default;

    void configure(const Config& cfg) {
        cfg_ = cfg;
    }

    void reset() {
        integral_ = 0.0f;
        lastError_ = 0.0f;
        integralGain_ = 0.0f;
        filteredDerivative_ = 0.0f;
    }

    float update(float error, float dtSeconds) {
        const float dt = std::clamp(dtSeconds, kMinDt, kMaxDt);

        // 输入死区
        if (std::abs(error) < cfg_.deadZone) {
            error = 0.0f;
        }

        // 比例项
        float pOut = cfg_.kp * error;

        // 自适应积分增益（时间归一化）
        float gain = adjustIntegralGain(error, lastError_, dt);
        // 物理积分 I_phys = sum(e*dt)；UI 限幅按 60Hz 样本单位 → 换算到秒
        const float integralLimitSec = cfg_.integralLimit * kReferenceDt;
        const float integralDeadzoneSec = cfg_.integralDeadzone * kReferenceDt;
        if (gain > 0.0f) {
            integral_ += error * dt;
            integral_ = std::clamp(integral_, -integralLimitSec, integralLimitSec);
        } else {
            integral_ = 0.0f;
        }

        // Ki_continuous = Ki_frame / kReferenceDt  ⇒  iOut = Ki_c * I_phys
        // 在 60Hz 时等价于旧版 Ki * sum(e)
        const float kiContinuous = cfg_.ki / kReferenceDt;
        float iOut = (std::abs(integral_) > integralDeadzoneSec)
            ? kiContinuous * integral_ : 0.0f;

        // 导数：de/dt；Kd_continuous = Kd_frame * kReferenceDt
        const float rawDerivative = (error - lastError_) / dt;
        float derivative = rawDerivative;
        if (cfg_.derivativeFilterTimeConstant > 1e-6f) {
            const float alpha = dt / (cfg_.derivativeFilterTimeConstant + dt);
            filteredDerivative_ += alpha * (rawDerivative - filteredDerivative_);
            derivative = filteredDerivative_;
        } else {
            filteredDerivative_ = rawDerivative;
        }
        const float kdContinuous = cfg_.kd * kReferenceDt;
        float dOut = kdContinuous * derivative;

        float output = pOut + iOut + dOut;
        output = std::clamp(output, -cfg_.outputLimit, cfg_.outputLimit);

        lastError_ = error;
        return output;
    }

    // 兼容旧调用（无 dt）：按 60Hz 一步处理
    float update(float error) {
        return update(error, kReferenceDt);
    }

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
    float integral_ = 0.0f;           // 物理积分（pixel·s）
    float lastError_ = 0.0f;
    float integralGain_ = 0.0f;
    float filteredDerivative_ = 0.0f;

    /// 自适应积分增益：接近目标时积分增强，远离时衰减（按时间）
    float adjustIntegralGain(float error, float lastError, float dt) {
        const float errorSpeed = std::abs(error - lastError) / dt; // pixels/s

        if (std::abs(error) < cfg_.integralGainThreshold) {
            // 旧版每帧 rise = rate * (1 - |Δe|/(threshold*2))
            // 时间化：ratePerSecond = rate * 60，motion 用 errorSpeed 归一
            const float ratePerSecond = cfg_.integralGainRate * kReferenceFps;
            float normalizedMotion = errorSpeed / (cfg_.integralGainThreshold * 2.0f * kReferenceFps);
            normalizedMotion = std::clamp(normalizedMotion, 0.0f, 1.0f);
            float riseRatePerSecond = ratePerSecond * (1.0f - normalizedMotion);
            riseRatePerSecond = std::clamp(riseRatePerSecond, 0.0f, ratePerSecond);
            integralGain_ = std::min(integralGain_ + riseRatePerSecond * dt, 1.0f);
        } else {
            // 旧版每帧: *= (1 - 0.05 * decayShape)
            // 时间化：retention^(dt*60)
            float decayShape =
                0.1f + 0.9f * std::tanh(std::abs(error) / (cfg_.integralGainThreshold * 2.0f));
            constexpr float kLegacyDecayPerFrame = 0.05f;
            const float base = std::max(0.0f, 1.0f - kLegacyDecayPerFrame * decayShape);
            const float retention = std::pow(base, dt * kReferenceFps);
            integralGain_ *= retention;
        }
        integralGain_ = std::clamp(integralGain_, 0.0f, 1.0f);
        return integralGain_;
    }
};

#endif // _WIN32
#endif // ADAPTIVE_PID_CONTROLLER_HPP
