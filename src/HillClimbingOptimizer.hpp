#ifndef HILL_CLIMBING_OPTIMIZER_HPP
#define HILL_CLIMBING_OPTIMIZER_HPP

#include <vector>
#include <deque>
#include <string>
#include <functional>
#include <chrono>
#include <mutex>
#include "MouseControllerInterface.hpp"

// 性能指标结构体
struct PerformanceMetrics {
    float avgError;           // 平均跟踪误差
    float maxError;           // 最大跟踪误差
    float oscillation;        // 振荡程度（输出变化方差）
    float smoothness;         // 平滑度（输出变化率）
    float convergenceRate;    // 收敛速度
    float score;              // 综合得分（越小越好）
    
    PerformanceMetrics() : avgError(0), maxError(0), oscillation(0), 
                           smoothness(0), convergenceRate(0), score(0) {}
};

// 参数边界结构体
struct ParameterBounds {
    float min;
    float max;
    float step;           // 初始步长
    float minStep;        // 最小步长
    
    ParameterBounds(float minVal = 0, float maxVal = 1, 
                    float stepVal = 0.01f, float minStepVal = 0.001f)
        : min(minVal), max(maxVal), step(stepVal), minStep(minStepVal) {}
};

// 优化配置结构体
struct OptimizerConfig {
    bool enabled = false;
    int sampleFrames = 300;             // 采样帧数
    int maxIterations = 100;            // 最大迭代次数
    float convergenceThreshold = 0.01f; // 收敛阈值
    float stepDecay = 0.5f;             // 步长衰减系数
    float stepGrowth = 1.2f;            // 步长增长系数
    
    // 目标函数权重
    float weightAvgError = 1.0f;
    float weightMaxError = 0.5f;
    float weightOscillation = 0.3f;
    float weightSmoothness = 0.2f;
    float weightConvergence = 0.3f;
    
    // 参数边界（每种算法）
    std::vector<ParameterBounds> advancedPidBounds;
    std::vector<ParameterBounds> standardPidBounds;
    std::vector<ParameterBounds> dopaPidBounds;
    std::vector<ParameterBounds> chrisPidBounds;
    
    OptimizerConfig() {
        // AdvancedPID参数边界
        advancedPidBounds = {
            {0.05f, 0.5f, 0.01f, 0.001f},   // pidPMin
            {0.3f, 1.0f, 0.02f, 0.002f},    // pidPMax
            {0.001f, 0.05f, 0.001f, 0.0001f}, // pidD
            {0.001f, 0.1f, 0.005f, 0.0005f}, // pidI
            {0.05f, 0.5f, 0.02f, 0.002f},   // derivativeFilterAlpha
            {0.0f, 0.5f, 0.02f, 0.002f},    // kalmanPredictionWeightX
            {0.0f, 0.3f, 0.01f, 0.001f},    // kalmanPredictionWeightY
            {0.0f, 0.5f, 0.02f, 0.002f},    // predictionWeightX
            {0.0f, 0.3f, 0.01f, 0.001f}     // predictionWeightY
        };
        
        // StandardPID参数边界
        standardPidBounds = {
            {0.1f, 0.8f, 0.02f, 0.002f},    // stdKp
            {0.001f, 0.1f, 0.005f, 0.0005f}, // stdKi
            {0.001f, 0.03f, 0.001f, 0.0001f}, // stdKd
            {0.05f, 0.5f, 0.02f, 0.002f},   // stdDerivativeFilterAlpha
            {0.3f, 0.95f, 0.02f, 0.002f},   // stdSmoothingX
            {0.3f, 0.95f, 0.02f, 0.002f}    // stdSmoothingY
        };
        
        // DopaPID参数边界
        dopaPidBounds = {
            {0.3f, 1.5f, 0.05f, 0.005f},    // dopaKpX
            {0.3f, 1.5f, 0.05f, 0.005f},    // dopaKpY
            {0.001f, 0.05f, 0.002f, 0.0002f}, // dopaKiX
            {0.001f, 0.05f, 0.002f, 0.0002f}, // dopaKiY
            {0.01f, 0.1f, 0.005f, 0.0005f}, // dopaKdX
            {0.01f, 0.1f, 0.005f, 0.0005f}, // dopaKdY
            {0.3f, 1.0f, 0.05f, 0.005f},    // dopaPredWeight
            {0.1f, 0.5f, 0.02f, 0.002f}     // dopaDFilterAlpha
        };
        
        // ChrisPID参数边界
        chrisPidBounds = {
            {0.2f, 0.8f, 0.02f, 0.002f},    // chrisKp
            {0.005f, 0.1f, 0.005f, 0.0005f}, // chrisKi
            {0.01f, 0.1f, 0.005f, 0.0005f}, // chrisKd
            {0.2f, 0.8f, 0.05f, 0.005f},    // chrisPredWeightX
            {0.0f, 0.3f, 0.02f, 0.002f},    // chrisPredWeightY
            {0.1f, 0.5f, 0.02f, 0.002f}     // chrisDFilterAlpha
        };
    }
};

// 数据采样点
struct SamplePoint {
    float errorX;
    float errorY;
    float outputX;
    float outputY;
    float targetVelocityX;
    float targetVelocityY;
    std::chrono::steady_clock::time_point timestamp;
};

// 参数更新回调类型
using ParameterUpdateCallback = std::function<void(const std::vector<float>& params)>;

class HillClimbingOptimizer {
public:
    HillClimbingOptimizer();
    ~HillClimbingOptimizer() = default;
    
    // 配置
    void setConfig(const OptimizerConfig& config);
    OptimizerConfig getConfig() const;
    
    // 设置算法类型
    void setAlgorithmType(AlgorithmType type);
    
    // 设置参数更新回调
    void setParameterUpdateCallback(ParameterUpdateCallback callback);
    
    // 数据收集
    void addSample(float errorX, float errorY, 
                   float outputX, float outputY,
                   float targetVelX = 0, float targetVelY = 0);
    
    // 优化控制
    void start();
    void stop();
    void reset();
    bool isRunning() const;
    
    // 获取当前状态
    PerformanceMetrics getCurrentMetrics() const;
    std::vector<float> getCurrentParameters() const;
    std::vector<float> getBestParameters() const;
    int getCurrentIteration() const;
    int getSampleCount() const;
    
    // 设置当前参数（从外部）
    void setCurrentParameters(const std::vector<float>& params);
    
    // 手动触发优化步骤（用于测试）
    bool step();
    
private:
    mutable std::mutex mutex_;
    OptimizerConfig config_;
    AlgorithmType algorithmType_;
    
    // 数据缓冲
    std::deque<SamplePoint> samples_;
    
    // 参数状态
    std::vector<float> currentParams_;
    std::vector<float> bestParams_;
    std::vector<float> currentSteps_;
    PerformanceMetrics currentMetrics_;
    PerformanceMetrics bestMetrics_;
    
    // 优化状态
    bool running_;
    int iteration_;
    int consecutiveNoImprovement_;
    
    // 回调
    ParameterUpdateCallback paramUpdateCallback_;
    
    // 内部方法
    PerformanceMetrics calculateMetrics();
    float calculateScore(const PerformanceMetrics& metrics);
    std::vector<ParameterBounds> getParameterBounds();
    void initializeParameters();
    bool tryParameterUpdate(int paramIndex, float delta);
    void decaySteps();
    void growStep(int paramIndex);
};

#endif // HILL_CLIMBING_OPTIMIZER_HPP
