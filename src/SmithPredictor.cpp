#ifdef _WIN32

#include "SmithPredictor.hpp"
#include <algorithm>
#include <cmath>

SmithPredictor::SmithPredictor()
    : modelStateX_(0.0f)
    , modelStateY_(0.0f)
    , modelStateX_delayed_(0.0f)
    , modelStateY_delayed_(0.0f)
    , delayBufCapacity_(0)
    , delayBufHead_(0)
    , delayBufCount_(0)
{
    delayBuf_.reserve(MAX_DELAY_SAMPLES);
}

void SmithPredictor::reset()
{
    modelStateX_ = 0.0f;
    modelStateY_ = 0.0f;
    modelStateX_delayed_ = 0.0f;
    modelStateY_delayed_ = 0.0f;
    delayBuf_.clear();
    delayBufHead_ = 0;
    delayBufCount_ = 0;
}

void SmithPredictor::setConfig(const Config& cfg)
{
    // tau变化小于10ms不触发reset，防止autoTau微小波动导致缓冲区每帧清零
    bool keyChanged = (config_.enabled != cfg.enabled)
                   || (config_.modelGainK != cfg.modelGainK)
                   || (config_.modelTimeConstT != cfg.modelTimeConstT)
                   || (std::abs(config_.delayTau - cfg.delayTau) > 0.01f);

    config_ = cfg;
    size_t newCapacity = static_cast<size_t>(
        std::ceil(config_.delayTau / 0.001f)
    );
    newCapacity = std::min(newCapacity, MAX_DELAY_SAMPLES);

    // 缓冲区容量变化或关键参数变化时才 reset
    if (keyChanged || newCapacity != delayBufCapacity_) {
        delayBufCapacity_ = newCapacity;
        reset();
    }
}

std::pair<float, float> SmithPredictor::correct(
    float pidOutputX, float pidOutputY,
    float measuredX, float measuredY,
    float dt)
{
    if (!config_.enabled || dt <= 1e-6f) {
        return { measuredX, measuredY };
    }

    const float K = config_.modelGainK;
    const float T = std::max(config_.modelTimeConstT, 0.001f);

    // G0(s) = K/(Ts+1)，欧拉离散：state += dt*(K*u - state)/T
    modelStateX_ += dt * (K * pidOutputX - modelStateX_) / T;
    modelStateY_ += dt * (K * pidOutputY - modelStateY_) / T;

    // 环形缓冲：存储当前PID输出
    if (delayBuf_.size() < MAX_DELAY_SAMPLES) {
        delayBuf_.push_back({ pidOutputX, pidOutputY });
    } else {
        delayBuf_[delayBufHead_] = { pidOutputX, pidOutputY };
    }
    delayBufHead_ = (delayBufHead_ + 1) % MAX_DELAY_SAMPLES;
    delayBufCount_ = std::min(delayBufCount_ + 1, MAX_DELAY_SAMPLES);

    // 取 u(t-τ)
    size_t delaySteps = static_cast<size_t>(
        std::ceil(config_.delayTau / std::max(dt, 0.001f))
    );
    delaySteps = std::min(delaySteps, delayBufCount_);

    float uDelayedX = 0.0f;
    float uDelayedY = 0.0f;
    if (delaySteps > 0 && delayBufCount_ > delaySteps) {
        size_t idx = (delayBufHead_ + MAX_DELAY_SAMPLES - delaySteps - 1) % MAX_DELAY_SAMPLES;
        if (idx < delayBuf_.size()) {
            uDelayedX = delayBuf_[idx].first;
            uDelayedY = delayBuf_[idx].second;
        }
    }

    // 延迟模型状态：用 u(t-τ) 驱动同一 G0(s)
    modelStateX_delayed_ += dt * (K * uDelayedX - modelStateX_delayed_) / T;
    modelStateY_delayed_ += dt * (K * uDelayedY - modelStateY_delayed_) / T;

    // Smith修正：实测值 + (无延迟模型输出 - 有延迟模型输出)
    float yCorrectedX = measuredX + (modelStateX_ - modelStateX_delayed_);
    float yCorrectedY = measuredY + (modelStateY_ - modelStateY_delayed_);

    return { yCorrectedX, yCorrectedY };
}

#endif