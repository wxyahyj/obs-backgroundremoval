#ifndef HILL_CLIMBING_OPTIMIZER_HPP
#define HILL_CLIMBING_OPTIMIZER_HPP

#include <vector>
#include <deque>
#include <string>
#include <functional>
#include <chrono>
#include <mutex>
#include <algorithm>
#include "MouseControllerInterface.hpp"

// 优化器运行模式
enum class OptimizationMode {
    TUNING,      // 微调模式：在当前参数基础上微调
    INDEPENDENT  // 独立模式：忽略当前手调PID，从头搜索最优解
};

// 优化策略
enum class OptimizationStrategy {
    STABLE_FIRST,  // 静稳优先：优先降低振荡和饱和
    BALANCED,      // 平衡：综合优化所有指标
    AGGRESSIVE     // 激进：优先追求最小误差
};

// 采样分类类型
enum class SampleType {
    STICKY,   // 稳定跟踪：误差小且稳定
    AMB,      // 歧义：多目标干扰
    REACQ,    // 重获取：目标丢失后重新获取
    SWITCH    // 切换目标
};

// 详细采样统计
struct SamplingStats {
    int stickyCount = 0;
    int ambCount = 0;
    int reacqCount = 0;
    int switchCount = 0;
    
    // 窗口目标指标
    float avgError = 0;
    float p95Error = 0;
    float oscillation = 0;
    float saturation = 0;
    float settleTime = 0;
    float predAccuracy = 0;
};

// 性能指标结构体
struct PerformanceMetrics {
    float avgError;
    float maxError;
    float p95Error;
    float oscillation;
    float smoothness;
    float convergenceRate;
    float saturation;     // 输出饱和率
    float settleTime;     // 稳定时间
    float predAccuracy;   // 预测准确度
    float score;
    
    SamplingStats stats;
    
    PerformanceMetrics() : avgError(0), maxError(0), p95Error(0), oscillation(0),
                           smoothness(0), convergenceRate(0), saturation(0),
                           settleTime(0), predAccuracy(0), score(0) {}
};

// 参数边界结构体
struct ParameterBounds {
    float minVal;
    float maxVal;
    float step;
    float minStep;
    
    ParameterBounds(float minVal = 0, float maxVal = 1, 
                    float step = 0.01f, float minStep = 0.001f)
        : minVal(minVal), maxVal(maxVal), step(step), minStep(minStep) {}
};

// 优化配置结构体
struct OptimizerConfig {
    bool enabled = false;
    
    // 模式和策略
    OptimizationMode mode = OptimizationMode::TUNING;
    OptimizationStrategy strategy = OptimizationStrategy::BALANCED;
    
    // 采样配置
    int sampleFrames = 300;
    int windowSize = 120;
    float minValidSampleRatio = 0.70f;
    
    // 迭代控制
    int maxIterations = 100;
    float targetError = 10.0f;
    bool allowSpeedOptimization = true;
    float stepDecayFactor = 0.16f;
    float convergenceThreshold = 0.01f;
    
    // 目标函数权重（根据策略调整）
    struct Weights {
        float weightAvgError = 1.0f;
        float weightMaxError = 0.5f;
        float weightP95Error = 0.3f;
        float weightOscillation = 0.4f;
        float weightSaturation = 0.5f;
        float weightSmoothness = 0.2f;
        float weightConvergence = 0.3f;
        
        void setForStrategy(OptimizationStrategy strategy);
    } weights;
    
    // 参数边界
    std::vector<ParameterBounds> advancedPidBounds;
    std::vector<ParameterBounds> standardPidBounds;
    std::vector<ParameterBounds> dopaPidBounds;
    std::vector<ParameterBounds> chrisPidBounds;
    
    OptimizerConfig() {
        initDefaultBounds();
    }
    
private:
    void initDefaultBounds();
};

// 数据采样点
struct SamplePoint {
    float errorX;
    float errorY;
    float outputX;
    float outputY;
    float targetVelocityX;
    float targetVelocityY;
    float predictedX;
    float predictedY;
    bool isSaturated;
    SampleType type;
    std::chrono::steady_clock::time_point timestamp;
    
    SamplePoint() : errorX(0), errorY(0), outputX(0), outputY(0),
                     targetVelocityX(0), targetVelocityY(0),
                     predictedX(0), predictedY(0), isSaturated(false),
                     type(SampleType::STICKY) {}
};

using ParameterUpdateCallback = std::function<void(const std::vector<float>& params)>;

// 优化器状态
enum class OptimizerState {
    IDLE,
    COLLECTING_BASELINE,
    EVALUATING_BASELINE,
    TESTING_PARAMETER,
    COLLECTING_TEST,
    EVALUATING_TEST,
    CONVERGED,
    STOPPED
};

class HillClimbingOptimizer {
public:
    HillClimbingOptimizer();
    ~HillClimbingOptimizer() = default;
    
    // 配置
    void setConfig(const OptimizerConfig& config);
    OptimizerConfig getConfig() const;
    
    void setAlgorithmType(AlgorithmType type);
    AlgorithmType getAlgorithmType() const;
    
    void setParameterUpdateCallback(ParameterUpdateCallback callback);
    
    // 数据收集（增强版）
    void addSample(float errorX, float errorY, 
                   float outputX, float outputY,
                   float targetVelX = 0, float targetVelY = 0,
                   float predX = 0, float predY = 0,
                   bool saturated = false,
                   SampleType type = SampleType::STICKY);
    
    // 控制
    void start();
    void stop();
    void reset();
    void applyBestToUI();
    bool isRunning() const;
    
    // 状态查询
    OptimizerState getState() const;
    const char* getStateString() const;
    int getCurrentIteration() const;
    int getSampleCount() const;
    int getRequiredSamples() const;
    float getProgress() const;
    
    // 参数查询
    std::vector<float> getCurrentParameters() const;
    std::vector<float> getBestParameters() const;
    std::vector<std::string> getParameterNames() const;
    
    // 性能指标
    PerformanceMetrics getCurrentMetrics() const;
    PerformanceMetrics getBaselineMetrics() const;
    PerformanceMetrics getBestMetrics() const;
    SamplingStats getSamplingStats() const;
    
    // 设置当前参数
    void setCurrentParameters(const std::vector<float>& params);
    
    // 主更新函数
    void update();
    
private:
    mutable std::mutex mutex_;
    OptimizerConfig config_;
    AlgorithmType algorithmType_;
    
    std::deque<SamplePoint> samples_;
    
    std::vector<float> currentParams_;
    std::vector<float> bestParams_;
    std::vector<float> currentSteps_;
    PerformanceMetrics currentMetrics_;
    PerformanceMetrics baselineMetrics_;
    PerformanceMetrics bestMetrics_;
    
    bool running_;
    int iteration_;
    int consecutiveNoImprovement_;
    OptimizerState state_;
    
    int currentParamIndex_;
    float currentDelta_;
    bool testingIncrease_;
    
    ParameterUpdateCallback paramUpdateCallback_;
    
    // 内部方法
    PerformanceMetrics calculateMetrics();
    float calculateScore(const PerformanceMetrics& metrics);
    std::vector<ParameterBounds> getParameterBounds();
    void initializeParameters();
    void tryNextParameter();
    void applyParameterChange(int index, float delta);
    void revertParameterChange(int index, float delta);
    void moveToNextParameter();
    void checkConvergence();
    void adjustWeightsForStrategy();
    SamplingStats calculateSamplingStats();
    float calculateP95Error();
    float calculateSaturation();
    float calculateSettleTime();
    float calculatePredAccuracy();
};

#endif // HILL_CLIMBING_OPTIMIZER_HPP
