#ifdef _WIN32

#include "OneEuroFilter.hpp"

static constexpr float PI = 3.14159265358979f;

OneEuroFilter::OneEuroFilter()
    : minCutoff(1.0f)
    , beta(0.0f)
    , dCutoff(1.0f)
    , xPrev(0.0f)
    , dxPrev(0.0f)
    , initialized(false)
{
}

float OneEuroFilter::computeAlpha(float cutoff, float dt) {
    float tau = 1.0f / (2.0f * PI * cutoff);
    return 1.0f / (1.0f + tau / dt);
}

float OneEuroFilter::filter(float value, float dt) {
    if (dt <= 1e-6f) {
        return xPrev;
    }

    if (!initialized) {
        xPrev = value;
        dxPrev = 0.0f;
        initialized = true;
        return value;
    }

    // 计算速度（一阶差分）
    float dx = (value - xPrev) / dt;
    // 平滑速度
    float alphaD = computeAlpha(dCutoff, dt);
    dxPrev = alphaD * dx + (1.0f - alphaD) * dxPrev;

    // 自适应截止频率
    float cutoff = minCutoff + beta * std::abs(dxPrev);
    // 滤波
    float alpha = computeAlpha(cutoff, dt);
    xPrev = alpha * value + (1.0f - alpha) * xPrev;

    return xPrev;
}

void OneEuroFilter::reset() {
    initialized = false;
    xPrev = 0.0f;
    dxPrev = 0.0f;
}

#endif // _WIN32
