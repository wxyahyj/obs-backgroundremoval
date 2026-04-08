#include "HillClimbingOptimizer.hpp"
#include <obs-module.h>
#include <plugin-support.h>
#include <cmath>
#include <algorithm>
#include <numeric>

HillClimbingOptimizer::HillClimbingOptimizer()
    : algorithmType_(AlgorithmType::AdvancedPID)
    , running_(false)
    , iteration_(0)
    , consecutiveNoImprovement_(0)
{
    initializeParameters();
}

void HillClimbingOptimizer::setConfig(const OptimizerConfig& config)
{
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    initializeParameters();
}

OptimizerConfig HillClimbingOptimizer::getConfig() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void HillClimbingOptimizer::setAlgorithmType(AlgorithmType type)
{
    std::lock_guard<std::mutex> lock(mutex_);
    algorithmType_ = type;
    initializeParameters();
}

void HillClimbingOptimizer::setParameterUpdateCallback(ParameterUpdateCallback callback)
{
    std::lock_guard<std::mutex> lock(mutex_);
    paramUpdateCallback_ = callback;
}

void HillClimbingOptimizer::addSample(float errorX, float errorY, 
                                       float outputX, float outputY,
                                       float targetVelX, float targetVelY)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!running_) return;
    
    SamplePoint sample;
    sample.errorX = errorX;
    sample.errorY = errorY;
    sample.outputX = outputX;
    sample.outputY = outputY;
    sample.targetVelocityX = targetVelX;
    sample.targetVelocityY = targetVelY;
    sample.timestamp = std::chrono::steady_clock::now();
    
    samples_.push_back(sample);
    
    // 限制样本数量
    while (samples_.size() > static_cast<size_t>(config_.sampleFrames * 2)) {
        samples_.pop_front();
    }
}

void HillClimbingOptimizer::start()
{
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = true;
    samples_.clear();
    iteration_ = 0;
    consecutiveNoImprovement_ = 0;
    bestMetrics_ = PerformanceMetrics();
    bestMetrics_.score = std::numeric_limits<float>::max();
    
    obs_log(LOG_INFO, "[HillClimbing] 优化器启动，算法类型: %d", static_cast<int>(algorithmType_));
}

void HillClimbingOptimizer::stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    
    obs_log(LOG_INFO, "[HillClimbing] 优化器停止，迭代次数: %d", iteration_);
    if (bestParams_.size() > 0) {
        obs_log(LOG_INFO, "[HillClimbing] 最优得分: %.4f", bestMetrics_.score);
    }
}

void HillClimbingOptimizer::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
    iteration_ = 0;
    consecutiveNoImprovement_ = 0;
    initializeParameters();
    bestMetrics_ = PerformanceMetrics();
    bestMetrics_.score = std::numeric_limits<float>::max();
}

bool HillClimbingOptimizer::isRunning() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

PerformanceMetrics HillClimbingOptimizer::getCurrentMetrics() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentMetrics_;
}

std::vector<float> HillClimbingOptimizer::getCurrentParameters() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return currentParams_;
}

std::vector<float> HillClimbingOptimizer::getBestParameters() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return bestParams_;
}

int HillClimbingOptimizer::getCurrentIteration() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return iteration_;
}

int HillClimbingOptimizer::getSampleCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(samples_.size());
}

void HillClimbingOptimizer::setCurrentParameters(const std::vector<float>& params)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (params.size() == currentParams_.size()) {
        currentParams_ = params;
    }
}

bool HillClimbingOptimizer::step()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!running_ || samples_.size() < static_cast<size_t>(config_.sampleFrames / 2)) {
        return false;
    }
    
    // 检查是否达到最大迭代次数
    if (iteration_ >= config_.maxIterations) {
        running_ = false;
        obs_log(LOG_INFO, "[HillClimbing] 达到最大迭代次数: %d", config_.maxIterations);
        return false;
    }
    
    // 计算当前性能指标
    currentMetrics_ = calculateMetrics();
    float currentScore = currentMetrics_.score;
    
    // 检查是否有改进
    bool improved = false;
    if (currentScore < bestMetrics_.score) {
        bestMetrics_ = currentMetrics_;
        bestParams_ = currentParams_;
        improved = true;
        consecutiveNoImprovement_ = 0;
        
        obs_log(LOG_INFO, "[HillClimbing] 迭代 %d: 得分 %.4f -> %.4f (改进)", 
                 iteration_, bestMetrics_.score, currentScore);
    } else {
        consecutiveNoImprovement_++;
    }
    
    // 检查收敛
    if (consecutiveNoImprovement_ >= 10) {
        // 连续10次无改进，衰减步长
        decaySteps();
        consecutiveNoImprovement_ = 0;
        
        obs_log(LOG_INFO, "[HillClimbing] 连续无改进，衰减步长");
    }
    
    // 尝试参数更新
    bool anyUpdate = false;
    auto bounds = getParameterBounds();
    
    for (size_t i = 0; i < currentParams_.size(); ++i) {
        // 随机选择方向（先尝试增加，再尝试减少）
        float delta = currentSteps_[i];
        
        // 尝试增加
        if (tryParameterUpdate(static_cast<int>(i), delta)) {
            anyUpdate = true;
            growStep(static_cast<int>(i));
            continue;
        }
        
        // 尝试减少
        if (tryParameterUpdate(static_cast<int>(i), -delta)) {
            anyUpdate = true;
            growStep(static_cast<int>(i));
            continue;
        }
    }
    
    // 清空样本缓冲区，准备下一轮
    samples_.clear();
    iteration_++;
    
    return true;
}

PerformanceMetrics HillClimbingOptimizer::calculateMetrics()
{
    PerformanceMetrics metrics;
    
    if (samples_.empty()) {
        return metrics;
    }
    
    // 计算平均误差
    float sumError = 0;
    float maxError = 0;
    for (const auto& sample : samples_) {
        float error = std::sqrt(sample.errorX * sample.errorX + 
                                sample.errorY * sample.errorY);
        sumError += error;
        maxError = std::max(maxError, error);
    }
    metrics.avgError = sumError / samples_.size();
    metrics.maxError = maxError;
    
    // 计算振荡程度（输出变化的方差）
    if (samples_.size() > 1) {
        float sumDelta = 0;
        float sumDeltaSq = 0;
        for (size_t i = 1; i < samples_.size(); ++i) {
            float dx = samples_[i].outputX - samples_[i-1].outputX;
            float dy = samples_[i].outputY - samples_[i-1].outputY;
            float delta = std::sqrt(dx * dx + dy * dy);
            sumDelta += delta;
            sumDeltaSq += delta * delta;
        }
        float avgDelta = sumDelta / (samples_.size() - 1);
        float avgDeltaSq = sumDeltaSq / (samples_.size() - 1);
        metrics.oscillation = avgDeltaSq - avgDelta * avgDelta; // 方差
    }
    
    // 计算平滑度（输出变化率的平均值）
    if (samples_.size() > 1) {
        float sumChange = 0;
        for (size_t i = 1; i < samples_.size(); ++i) {
            float dx = std::abs(samples_[i].outputX - samples_[i-1].outputX);
            float dy = std::abs(samples_[i].outputY - samples_[i-1].outputY);
            sumChange += dx + dy;
        }
        metrics.smoothness = sumChange / (samples_.size() - 1);
    }
    
    // 计算收敛速度（误差下降率）
    if (samples_.size() > 10) {
        float firstHalfError = 0, secondHalfError = 0;
        size_t half = samples_.size() / 2;
        for (size_t i = 0; i < half; ++i) {
            firstHalfError += std::sqrt(samples_[i].errorX * samples_[i].errorX + 
                                        samples_[i].errorY * samples_[i].errorY);
        }
        for (size_t i = half; i < samples_.size(); ++i) {
            secondHalfError += std::sqrt(samples_[i].errorX * samples_[i].errorX + 
                                         samples_[i].errorY * samples_[i].errorY);
        }
        firstHalfError /= half;
        secondHalfError /= (samples_.size() - half);
        
        if (firstHalfError > 0) {
            metrics.convergenceRate = (firstHalfError - secondHalfError) / firstHalfError;
        }
    }
    
    // 计算综合得分（越小越好）
    metrics.score = calculateScore(metrics);
    
    return metrics;
}

float HillClimbingOptimizer::calculateScore(const PerformanceMetrics& metrics)
{
    // 归一化并加权求和
    // 平均误差：直接使用（像素）
    // 最大误差：归一化到0-1范围
    // 振荡：归一化到0-1范围
    // 平滑度：归一化到0-1范围
    // 收敛速度：已经是比率，直接使用
    
    float normalizedMaxError = metrics.maxError / 500.0f;  // 假设最大误差500像素
    float normalizedOscillation = std::min(metrics.oscillation / 100.0f, 1.0f);
    float normalizedSmoothness = std::min(metrics.smoothness / 50.0f, 1.0f);
    float normalizedConvergence = std::max(0.0f, metrics.convergenceRate);
    
    float score = config_.weightAvgError * metrics.avgError +
                  config_.weightMaxError * normalizedMaxError * 100 +
                  config_.weightOscillation * normalizedOscillation * 100 +
                  config_.weightSmoothness * normalizedSmoothness * 100 -
                  config_.weightConvergence * normalizedConvergence * 50;
    
    return score;
}

std::vector<ParameterBounds> HillClimbingOptimizer::getParameterBounds()
{
    switch (algorithmType_) {
        case AlgorithmType::AdvancedPID:
            return config_.advancedPidBounds;
        case AlgorithmType::StandardPID:
            return config_.standardPidBounds;
        case AlgorithmType::DopaPID:
            return config_.dopaPidBounds;
        case AlgorithmType::ChrisPID:
            return config_.chrisPidBounds;
        default:
            return config_.advancedPidBounds;
    }
}

void HillClimbingOptimizer::initializeParameters()
{
    auto bounds = getParameterBounds();
    currentParams_.resize(bounds.size());
    currentSteps_.resize(bounds.size());
    
    // 初始化参数为边界中点
    for (size_t i = 0; i < bounds.size(); ++i) {
        currentParams_[i] = (bounds[i].minVal + bounds[i].maxVal) / 2.0f;
        currentSteps_[i] = bounds[i].step;
    }
    
    bestParams_ = currentParams_;
}

bool HillClimbingOptimizer::tryParameterUpdate(int paramIndex, float delta)
{
    auto bounds = getParameterBounds();
    if (paramIndex < 0 || paramIndex >= static_cast<int>(bounds.size())) {
        return false;
    }
    
    float newValue = currentParams_[paramIndex] + delta;
    
    // 检查边界
    if (newValue < bounds[paramIndex].minVal || newValue > bounds[paramIndex].maxVal) {
        return false;
    }
    
    // 保存旧值
    float oldValue = currentParams_[paramIndex];
    
    // 尝试新值
    currentParams_[paramIndex] = newValue;
    
    // 计算新得分
    PerformanceMetrics newMetrics = calculateMetrics();
    float newScore = newMetrics.score;
    
    // 如果有改进，保留新值
    if (newScore < currentMetrics_.score) {
        currentMetrics_ = newMetrics;
        
        // 调用回调更新参数
        if (paramUpdateCallback_) {
            paramUpdateCallback_(currentParams_);
        }
        
        return true;
    }
    
    // 无改进，恢复旧值
    currentParams_[paramIndex] = oldValue;
    return false;
}

void HillClimbingOptimizer::decaySteps()
{
    auto bounds = getParameterBounds();
    for (size_t i = 0; i < currentSteps_.size(); ++i) {
        currentSteps_[i] = std::max(bounds[i].minStep, 
                                    currentSteps_[i] * config_.stepDecay);
    }
}

void HillClimbingOptimizer::growStep(int paramIndex)
{
    auto bounds = getParameterBounds();
    if (paramIndex >= 0 && paramIndex < static_cast<int>(bounds.size())) {
        currentSteps_[paramIndex] = std::min(bounds[paramIndex].step,
                                             currentSteps_[paramIndex] * config_.stepGrowth);
    }
}
