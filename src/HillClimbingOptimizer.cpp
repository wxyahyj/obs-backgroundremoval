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
    , state_(OptimizerState::IDLE)
    , currentParamIndex_(0)
    , currentDelta_(0)
    , testingIncrease_(true)
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
    state_ = OptimizerState::COLLECTING;
    currentParamIndex_ = 0;
    testingIncrease_ = true;
    
    bestMetrics_ = PerformanceMetrics();
    bestMetrics_.score = std::numeric_limits<float>::max();
    baselineMetrics_ = PerformanceMetrics();
    baselineMetrics_.score = std::numeric_limits<float>::max();
    
    obs_log(LOG_INFO, "[HillClimbing] 优化器启动，算法类型: %d", static_cast<int>(algorithmType_));
}

void HillClimbingOptimizer::stop()
{
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    state_ = OptimizerState::IDLE;
    
    obs_log(LOG_INFO, "[HillClimbing] 优化器停止，迭代次数: %d", iteration_);
    if (bestParams_.size() > 0 && bestMetrics_.score < std::numeric_limits<float>::max()) {
        obs_log(LOG_INFO, "[HillClimbing] 最优得分: %.4f", bestMetrics_.score);
    }
}

void HillClimbingOptimizer::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
    iteration_ = 0;
    consecutiveNoImprovement_ = 0;
    state_ = OptimizerState::IDLE;
    initializeParameters();
    bestMetrics_ = PerformanceMetrics();
    bestMetrics_.score = std::numeric_limits<float>::max();
    baselineMetrics_ = PerformanceMetrics();
    baselineMetrics_.score = std::numeric_limits<float>::max();
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

OptimizerState HillClimbingOptimizer::getState() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

void HillClimbingOptimizer::setCurrentParameters(const std::vector<float>& params)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (params.size() == currentParams_.size()) {
        currentParams_ = params;
    }
}

void HillClimbingOptimizer::update()
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!running_) return;
    
    // 检查是否达到最大迭代次数
    if (iteration_ >= config_.maxIterations) {
        running_ = false;
        state_ = OptimizerState::IDLE;
        obs_log(LOG_INFO, "[HillClimbing] 达到最大迭代次数: %d", config_.maxIterations);
        return;
    }
    
    switch (state_) {
        case OptimizerState::COLLECTING:
            // 等待收集足够数据
            if (static_cast<int>(samples_.size()) >= config_.sampleFrames) {
                state_ = OptimizerState::EVALUATING;
            }
            break;
            
        case OptimizerState::EVALUATING: {
            // 计算当前性能
            currentMetrics_ = calculateMetrics();
            
            // 如果是基线评估（第一次或回退后）
            if (baselineMetrics_.score >= std::numeric_limits<float>::max() - 1) {
                baselineMetrics_ = currentMetrics_;
                obs_log(LOG_INFO, "[HillClimbing] 基线得分: %.4f (avgError=%.2f, oscillation=%.2f)",
                        baselineMetrics_.score, baselineMetrics_.avgError, baselineMetrics_.oscillation);
                
                // 开始测试第一个参数
                tryNextParameter();
            } else {
                // 比较新得分和基线得分
                if (currentMetrics_.score < baselineMetrics_.score) {
                    // 有改进！更新基线
                    float improvement = baselineMetrics_.score - currentMetrics_.score;
                    baselineMetrics_ = currentMetrics_;
                    
                    if (currentMetrics_.score < bestMetrics_.score) {
                        bestMetrics_ = currentMetrics_;
                        bestParams_ = currentParams_;
                    }
                    
                    obs_log(LOG_INFO, "[HillClimbing] 迭代 %d: 改进 %.4f, 新得分 %.4f (参数%d %s %.4f)",
                            iteration_, improvement, currentMetrics_.score,
                            currentParamIndex_, testingIncrease_ ? "+" : "-",
                            currentDelta_);
                    
                    // 增大步长
                    auto bounds = getParameterBounds();
                    currentSteps_[currentParamIndex_] = std::min(
                        bounds[currentParamIndex_].step,
                        currentSteps_[currentParamIndex_] * config_.stepGrowth);
                    
                    consecutiveNoImprovement_ = 0;
                    
                    // 继续同方向
                    applyParameterChange(currentParamIndex_, 
                                        testingIncrease_ ? currentSteps_[currentParamIndex_] : -currentSteps_[currentParamIndex_]);
                    iteration_++;
                    samples_.clear();
                    state_ = OptimizerState::COLLECTING;
                } else {
                    // 无改进，回退
                    revertParameterChange(currentParamIndex_, 
                                         testingIncrease_ ? currentDelta_ : -currentDelta_);
                    
                    consecutiveNoImprovement_++;
                    
                    // 尝试反方向
                    if (testingIncrease_) {
                        testingIncrease_ = false;
                        currentDelta_ = currentSteps_[currentParamIndex_];
                        applyParameterChange(currentParamIndex_, -currentDelta_);
                        samples_.clear();
                        state_ = OptimizerState::COLLECTING;
                    } else {
                        // 两个方向都试过了，移动到下一个参数
                        moveToNextParameter();
                    }
                }
            }
            break;
        }
        
        case OptimizerState::TESTING:
            // 不应该到达这里
            state_ = OptimizerState::COLLECTING;
            break;
            
        case OptimizerState::IDLE:
        default:
            break;
    }
    
    // 检查连续无改进
    if (consecutiveNoImprovement_ >= static_cast<int>(currentParams_.size()) * 2) {
        // 衰减步长
        auto bounds = getParameterBounds();
        for (size_t i = 0; i < currentSteps_.size(); ++i) {
            currentSteps_[i] = std::max(bounds[i].minStep,
                                        currentSteps_[i] * config_.stepDecay);
        }
        consecutiveNoImprovement_ = 0;
        obs_log(LOG_INFO, "[HillClimbing] 连续无改进，衰减步长");
    }
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
        metrics.oscillation = avgDeltaSq - avgDelta * avgDelta;
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
    
    // 计算综合得分
    metrics.score = calculateScore(metrics);
    
    return metrics;
}

float HillClimbingOptimizer::calculateScore(const PerformanceMetrics& metrics)
{
    float normalizedMaxError = metrics.maxError / 500.0f;
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

void HillClimbingOptimizer::tryNextParameter()
{
    auto bounds = getParameterBounds();
    
    // 找到下一个可以调整的参数
    int attempts = 0;
    while (attempts < static_cast<int>(bounds.size())) {
        currentDelta_ = currentSteps_[currentParamIndex_];
        testingIncrease_ = true;
        
        // 检查是否可以增加
        if (currentParams_[currentParamIndex_] + currentDelta_ <= bounds[currentParamIndex_].maxVal) {
            applyParameterChange(currentParamIndex_, currentDelta_);
            samples_.clear();
            state_ = OptimizerState::COLLECTING;
            obs_log(LOG_INFO, "[HillClimbing] 测试参数 %d: +%.4f (当前值 %.4f)",
                    currentParamIndex_, currentDelta_, currentParams_[currentParamIndex_]);
            return;
        }
        
        // 尝试减少
        if (currentParams_[currentParamIndex_] - currentDelta_ >= bounds[currentParamIndex_].minVal) {
            testingIncrease_ = false;
            applyParameterChange(currentParamIndex_, -currentDelta_);
            samples_.clear();
            state_ = OptimizerState::COLLECTING;
            obs_log(LOG_INFO, "[HillClimbing] 测试参数 %d: -%.4f (当前值 %.4f)",
                    currentParamIndex_, currentDelta_, currentParams_[currentParamIndex_]);
            return;
        }
        
        // 移动到下一个参数
        currentParamIndex_ = (currentParamIndex_ + 1) % static_cast<int>(bounds.size());
        attempts++;
    }
    
    // 所有参数都无法调整
    obs_log(LOG_INFO, "[HillClimbing] 所有参数已达边界，优化完成");
    running_ = false;
    state_ = OptimizerState::IDLE;
}

void HillClimbingOptimizer::applyParameterChange(int index, float delta)
{
    auto bounds = getParameterBounds();
    float newValue = currentParams_[index] + delta;
    newValue = std::max(bounds[index].minVal, std::min(bounds[index].maxVal, newValue));
    currentParams_[index] = newValue;
    
    // 调用回调更新实际参数
    if (paramUpdateCallback_) {
        paramUpdateCallback_(currentParams_);
    }
}

void HillClimbingOptimizer::revertParameterChange(int index, float delta)
{
    currentParams_[index] -= delta;
    
    // 调用回调恢复参数
    if (paramUpdateCallback_) {
        paramUpdateCallback_(currentParams_);
    }
}

void HillClimbingOptimizer::moveToNextParameter()
{
    auto bounds = getParameterBounds();
    currentParamIndex_ = (currentParamIndex_ + 1) % static_cast<int>(bounds.size());
    
    // 重置为尝试增加方向
    testingIncrease_ = true;
    currentDelta_ = currentSteps_[currentParamIndex_];
    
    // 尝试下一个参数
    tryNextParameter();
}
