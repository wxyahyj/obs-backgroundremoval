#include "GhostTracker.hpp"
#include <algorithm>
#include <chrono>

// Perlin噪声排列表
const int GhostTracker::permutation_[512] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    // 重复前256个
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
};

GhostTracker::GhostTracker() 
    : dist_(0.0f, 1.0f) {
    // 使用时间戳初始化随机数生成器
    auto seed = static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
    );
    rng_.seed(seed);
}

void GhostTracker::setConfig(const Config& config) {
    config_ = config;
    config_.curvature = std::clamp(config_.curvature, 0.0f, 1.0f);
}

void GhostTracker::reset() {
    isTracking_ = false;
    lockedSideVec_ = {0.0f, 0.0f};
    lockedRailVec_ = {0.0f, 0.0f};
    lockedDynamicMod_ = 1.0f;
    startProjDist_ = 0.0f;
    deadRadius_ = 0.0f;
    frameIdx_ = 0;
    lowestGain_ = 1.0f;
    activeCurvature_ = 0.0f;
    peakBias_ = 1.0f;
    noisePhase_ = 0.0f;
}

// Perlin噪声辅助函数
float GhostTracker::fade(float t) const {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float GhostTracker::lerp(float a, float b, float t) const {
    return a + t * (b - a);
}

float GhostTracker::grad(int hash, float x) const {
    return (hash & 1) == 0 ? x : -x;
}

float GhostTracker::perlin1D(float x) const {
    int X = static_cast<int>(std::floor(x)) & 255;
    x -= std::floor(x);
    
    float u = fade(x);
    
    int a = permutation_[X];
    int b = permutation_[X + 1];
    
    return lerp(grad(a, x), grad(b, x - 1), u);
}

bool GhostTracker::apply(float targetX, float targetY,
                          float targetWidth, float targetHeight,
                          float frameWidth, float frameHeight,
                          float& outOffsetX, float& outOffsetY) {
    // 未启用则不处理
    if (!config_.enabled) {
        outOffsetX = 0.0f;
        outOffsetY = 0.0f;
        return false;
    }
    
    // Phase 1: 初始化锁定
    if (!isTracking_) {
        float rawDist = std::sqrt(targetX * targetX + targetY * targetY);
        if (rawDist < 1e-6f) {
            outOffsetX = 0.0f;
            outOffsetY = 0.0f;
            return false;
        }
        
        // 1. 动态缩放
        float tSize = std::min(targetWidth, targetHeight);
        float fSize = std::min(frameWidth, frameHeight);
        float targetRatio = std::clamp(tSize / fSize, 0.0f, 1.0f);
        float sinArg = 1.57f * targetRatio;
        float sinVal = std::sin(sinArg);
        lockedDynamicMod_ = 1.0f + 0.1f * sinVal;
        deadRadius_ = tSize * lockedDynamicMod_ * 0.5f;
        
        // 2. 垂直吸附
        float rawDirX = targetX / rawDist;
        float rawDirY = targetY / rawDist;
        
        if (std::abs(rawDirY) > config_.verticalSnapRatio * std::abs(rawDirX)) {
            // 垂直吸附
            lockedRailVec_[0] = 0.0f;
            lockedRailVec_[1] = std::copysign(1.0f, rawDirY);
            startProjDist_ = std::abs(targetY);
        } else {
            // 正常轨道
            lockedRailVec_[0] = rawDirX;
            lockedRailVec_[1] = rawDirY;
            startProjDist_ = rawDist;
        }
        
        // 3. 侧向向量计算
        float sideSign = (dist_(rng_) > 0.5f) ? 1.0f : -1.0f;
        lockedSideVec_[0] = -lockedRailVec_[1] * sideSign;
        lockedSideVec_[1] = lockedRailVec_[0] * sideSign;
        
        // 4. 随机化曲率和峰值偏置
        float r0 = config_.randomRangeMin + dist_(rng_) * (config_.randomRangeMax - config_.randomRangeMin);
        float r1 = config_.randomRangeMin + dist_(rng_) * (config_.randomRangeMax - config_.randomRangeMin);
        activeCurvature_ = config_.curvature * r0;
        peakBias_ = r1;
        
        isTracking_ = true;
        frameIdx_ = 0;
        lowestGain_ = 1.0f;
        noisePhase_ = 0.0f;
    }
    
    frameIdx_++;
    
    // Phase 2: 轨迹计算
    // 1. 投影距离
    float projDist = std::max(
        targetX * lockedRailVec_[0] + targetY * lockedRailVec_[1],
        0.0f
    );
    
    // 2. 线性衰减
    float denom = std::max(startProjDist_ - deadRadius_, 1e-6f);
    float relDist = std::clamp((projDist - deadRadius_) / denom, 0.0f, 1.0f);
    
    // 3. 幂函数偏置
    float rawGain = std::pow(relDist, peakBias_);
    
    // 4. 单调锁定
    float finalGain = std::min(rawGain, lowestGain_);
    lowestGain_ = finalGain;
    
    // Phase 3: 应用输出
    // 基础振幅
    float amp = startProjDist_ * activeCurvature_ * finalGain * lockedDynamicMod_;
    
    // Perlin噪声
    noisePhase_ += config_.noiseFreq;
    noiseVec_[0] = perlin1D(noisePhase_);
    noiseVec_[1] = perlin1D(noisePhase_ + 42.0f);  // 偏移42使X/Y独立
    
    // 噪声缩放
    float noiseScale = config_.noiseIntensity * finalGain * lockedDynamicMod_;
    
    // 输出偏移
    outOffsetX = lockedSideVec_[0] * amp + noiseVec_[0] * noiseScale;
    outOffsetY = lockedSideVec_[1] * amp + noiseVec_[1] * noiseScale;
    
    return true;
}
