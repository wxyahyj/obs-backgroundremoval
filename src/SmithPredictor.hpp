#ifndef SMITH_PREDICTOR_HPP
#define SMITH_PREDICTOR_HPP

#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

class SmithPredictor {
public:
    struct Config {
        float modelGainK = 1.0f;
        float modelTimeConstT = 0.05f;
        float delayTau = 0.02f;
        bool enabled = false;
    };

    SmithPredictor();

    void reset();
    void setConfig(const Config& cfg);

    std::pair<float, float> correct(
        float pidOutputX, float pidOutputY,
        float measuredX, float measuredY,
        float dt);

    const Config& getConfig() const { return config_; }

    // 诊断接口：获取内部状态
    float getModelStateX() const { return modelStateX_; }
    float getModelStateY() const { return modelStateY_; }
    float getModelStateDelayedX() const { return modelStateX_delayed_; }
    float getModelStateDelayedY() const { return modelStateY_delayed_; }
    size_t getDelayBufCount() const { return delayBufCount_; }
    size_t getDelaySteps(float dt) const {
        if (dt <= 1e-6f) return 0;
        size_t steps = static_cast<size_t>(
            std::ceil(config_.delayTau / std::max(dt, 0.001f))
        );
        return std::min(steps, delayBufCount_);
    }

private:
    Config config_;

    float modelStateX_;
    float modelStateY_;
    float modelStateX_delayed_;
    float modelStateY_delayed_;

    std::vector<std::pair<float, float>> delayBuf_;
    size_t delayBufCapacity_;
    size_t delayBufHead_;
    size_t delayBufCount_;

    static constexpr size_t MAX_DELAY_SAMPLES = 128;
};

#endif