#ifndef GAUSSIAN_GRAVITY_CONTROLLER_HPP
#define GAUSSIAN_GRAVITY_CONTROLLER_HPP

#ifdef _WIN32

#include <array>
#include <cmath>
#include <algorithm>

struct GaussianGravityConfig {
    float gravityStrength = 0.5f;      // 引力强度 G
    float maxDistance = 500.0f;        // 最大作用距离
    float softEpsilon = 100.0f;        // 软化系数 ε
    float maxForce = 100.0f;           // 最大输出力
    float smoothingFactor = 0.3f;      // 输出平滑因子
    bool confidenceScale = true;       // 是否按置信度缩放
    float predictionWeight = 0.3f;     // 速度预测权重
};

class GaussianGravityController {
public:
    GaussianGravityController();
    
    void setConfig(const GaussianGravityConfig& config);
    void compute(float target_x, float target_y, float crosshair_x, float crosshair_y,
                 float target_confidence, float velocity_x, float velocity_y,
                 float& out_x, float& out_y);
    void reset();

private:
    GaussianGravityConfig config_;
    
    std::array<float, 2> last_output_;
    
    float calculateGravityForce(float distance);
};

#endif // _WIN32

#endif // GAUSSIAN_GRAVITY_CONTROLLER_HPP
