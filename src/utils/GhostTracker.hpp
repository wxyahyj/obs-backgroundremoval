#pragma once

#include <cmath>
#include <random>
#include <array>

/**
 * GhostTracker - 曲线轨迹生成器
 * 
 * 生成带有Perlin噪声的弧形移动路径，模拟人类鼠标移动轨迹
 * 
 * 特性：
 * - 曲线轨迹（非直线移动）
 * - Perlin噪声（自然的手部抖动效果）
 * - 单调递减（确保轨迹始终向目标收敛）
 * - 垂直吸附（垂直方向时锁定垂直轨道）
 * - 动态缩放（根据目标大小调整参数）
 */
class GhostTracker {
public:
    struct Config {
        bool enabled = false;               // 是否启用
        float curvature = 0.5f;             // 曲线强度 (0.0 - 1.0)
        float noiseIntensity = 12.0f;       // 最大噪声像素
        float verticalSnapRatio = 3.0f;     // 垂直吸附比例阈值
        float noiseFreq = 0.8f;             // Perlin噪声频率
        float randomRangeMin = 0.8f;        // 随机化范围下限
        float randomRangeMax = 1.2f;        // 随机化范围上限
    };

private:
    // 配置
    Config config_;
    
    // 内部状态
    bool isTracking_ = false;
    float startProjDist_ = 0.0f;
    float deadRadius_ = 0.0f;
    int frameIdx_ = 0;
    float lowestGain_ = 1.0f;
    float noisePhase_ = 0.0f;
    
    // 随机化状态
    float activeCurvature_ = 0.0f;
    float peakBias_ = 1.0f;
    
    // 锁定向量
    std::array<float, 2> lockedSideVec_ = {0.0f, 0.0f};
    std::array<float, 2> lockedRailVec_ = {0.0f, 0.0f};
    float lockedDynamicMod_ = 1.0f;
    std::array<float, 2> noiseVec_ = {0.0f, 0.0f};
    
    // 随机数生成器
    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
    
    // Perlin噪声实现
    static const int permutation_[512];
    float fade(float t) const;
    float lerp(float a, float b, float t) const;
    float grad(int hash, float x) const;
    float perlin1D(float x) const;

public:
    GhostTracker();
    ~GhostTracker() = default;
    
    // 配置
    void setConfig(const Config& config);
    const Config& getConfig() const { return config_; }
    
    // 重置状态
    void reset();
    
    /**
     * 应用曲线轨迹
     * 
     * @param targetX 目标X坐标（相对于准心）
     * @param targetY 目标Y坐标（相对于准心）
     * @param targetWidth 目标宽度
     * @param targetHeight 目标高度
     * @param frameWidth 帧宽度
     * @param frameHeight 帧高度
     * @param outOffsetX 输出X偏移
     * @param outOffsetY 输出Y偏移
     * @return 是否成功应用
     */
    bool apply(float targetX, float targetY, 
               float targetWidth, float targetHeight,
               float frameWidth, float frameHeight,
               float& outOffsetX, float& outOffsetY);
};
