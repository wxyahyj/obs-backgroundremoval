#ifndef INCREMENTAL_PID_CONTROLLER_HPP
#define INCREMENTAL_PID_CONTROLLER_HPP

#ifdef _WIN32

#include "mpid.hpp"

struct IncrementalPIDConfig {
    float kp = 0.5f;
    float ki = 0.1f;
    float kd = 0.05f;
    float speedX = 1.0f;
    float speedY = 1.0f;
    int aimRadius = 200;
    bool jitterEnabled = false;
    bool pidEnabled = true;
    bool sideCompEnabled = false;
    float sideCompCap = 5.0f;
    float sideCompDenom = 1.0f;
    int centerX = 0;
    int centerY = 0;
};

class IncrementalPIDAdapter {
public:
    IncrementalPIDAdapter();
    
    void update(float errorX, float errorY, float& outX, float& outY);
    void setConfig(const IncrementalPIDConfig& config);
    void reset();
    
    struct DebugTerms {
        float pTerm = 0;        // 比例项 Kp*(e - e1)
        float iTerm = 0;        // 积分项 Ki*e
        float dTerm = 0;        // 微分项 Kd*(e - 2*e1 + e2)
        float pidOutput = 0;    // PID总输出
        int lastDistance = 0;
        int previousOutputX = 0;
    };
    DebugTerms getLastDebugTerms() const { return lastDebugTerms_; }
    
private:
    mist::reconstructed::PidControlChain chain_;
    mist::reconstructed::ChainConfig chainConfig_;
    IncrementalPIDConfig config_;
    DebugTerms lastDebugTerms_;
};

#endif

#endif
