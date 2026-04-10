#include "HillClimbingOptimizer.hpp"
#include <obs-module.h>
#include <plugin-support.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>

// OptimizerConfig::Weights 实现
void OptimizerConfig::Weights::setForStrategy(OptimizationStrategy strategy) {
    switch (strategy) {
        case OptimizationStrategy::STABLE_FIRST:
            weightAvgError = 0.8f;
            weightMaxError = 0.3f;
            weightP95Error = 0.2f;
            weightOscillation = 1.0f;    // 高权重
            weightSaturation = 0.8f;     // 高权重
            weightSmoothness = 0.5f;
            weightConvergence = 0.2f;
            break;
        case OptimizationStrategy::BALANCED:
            weightAvgError = 1.0f;
            weightMaxError = 0.5f;
            weightP95Error = 0.3f;
            weightOscillation = 0.4f;
            weightSaturation = 0.5f;
            weightSmoothness = 0.2f;
            weightConvergence = 0.3f;
            break;
        case OptimizationStrategy::AGGRESSIVE:
            weightAvgError = 1.5f;       // 高权重
            weightMaxError = 0.8f;
            weightP95Error = 0.5f;
            weightOscillation = 0.2f;    // 低权重
            weightSaturation = 0.3f;
            weightSmoothness = 0.1f;
            weightConvergence = 0.5f;
            break;
    }
}

// OptimizerConfig 默认边界初始化
void OptimizerConfig::initDefaultBounds() {
    // AdvancedPID完整参数：PID核心 + 预测 + 滤波
    // 0-3: PID核心 (PMin, PMax, D, I)
    // 4: D项滤波
    // 5-6: 卡尔曼预测权重
    // 7-8: 导数预测权重
    // 9-10: 输出平滑
    // 5-6: 导数预测权重
    // 7-8: 平滑系数
    advancedPidBounds = {
        {0.05f, 0.5f, 0.01f, 0.001f},     // pidPMin
        {0.3f, 1.0f, 0.02f, 0.002f},      // pidPMax
        {0.001f, 0.05f, 0.001f, 0.0001f},  // pidD
        {0.001f, 0.1f, 0.005f, 0.0005f},   // pidI
        {0.05f, 0.5f, 0.02f, 0.002f},      // derivativeFilterAlpha
        {0.0f, 0.5f, 0.02f, 0.002f},       // predictionWeightX
        {0.0f, 0.3f, 0.01f, 0.001f},       // predictionWeightY
        {0.3f, 0.95f, 0.02f, 0.002f},      // aimSmoothingX
        {0.3f, 0.95f, 0.02f, 0.002f}       // aimSmoothingY
    };
    
    // StandardPID: PID + 滤波 + 平滑
    standardPidBounds = {
        {0.1f, 0.8f, 0.02f, 0.002f},      // stdKp
        {0.001f, 0.1f, 0.005f, 0.0005f},   // stdKi
        {0.001f, 0.03f, 0.001f, 0.0001f},  // stdKd
        {0.05f, 0.5f, 0.02f, 0.002f},      // stdDerivativeFilterAlpha
        {0.3f, 0.95f, 0.02f, 0.002f},      // stdSmoothingX
        {0.3f, 0.95f, 0.02f, 0.002f},      // stdSmoothingY
        {0.03f, 0.2f, 0.01f, 0.001f}       // maxPredictionTime
    };
    
    // ChrisPID: PID + 预测 + 滤波
    chrisPidBounds = {
        {0.2f, 0.8f, 0.02f, 0.002f},      // chrisKp
        {0.005f, 0.1f, 0.005f, 0.0005f},  // chrisKi
        {0.01f, 0.1f, 0.005f, 0.0005f},   // chrisKd
        {0.2f, 0.8f, 0.05f, 0.005f},      // chrisPredWeightX
        {0.0f, 0.3f, 0.02f, 0.002f},      // chrisPredWeightY
        {0.1f, 0.5f, 0.02f, 0.002f},      // chrisDFilterAlpha
        {0.3f, 0.95f, 0.02f, 0.002f},      // aimSmoothingX
        {0.3f, 0.95f, 0.02f, 0.002f},      // aimSmoothingY
        {0.05f, 0.4f, 0.02f, 0.002f},      // velocitySmoothFactor
        {0.05f, 0.4f, 0.02f, 0.002f}       // accelerationSmoothFactor
    };
    
    weights.setForStrategy(strategy);
}

// HillClimbingOptimizer 实现
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

void HillClimbingOptimizer::setConfig(const OptimizerConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    config_.weights.setForStrategy(config_.strategy);
    if (config_.mode == OptimizationMode::INDEPENDENT) {
        initializeParameters();
    }
}

OptimizerConfig HillClimbingOptimizer::getConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void HillClimbingOptimizer::setAlgorithmType(AlgorithmType type) {
    std::lock_guard<std::mutex> lock(mutex_);
    algorithmType_ = type;
    if (config_.mode == OptimizationMode::INDEPENDENT) {
        initializeParameters();
    }
}

AlgorithmType HillClimbingOptimizer::getAlgorithmType() const {
    return algorithmType_;
}

void HillClimbingOptimizer::setParameterUpdateCallback(ParameterUpdateCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    paramUpdateCallback_ = callback;
}

void HillClimbingOptimizer::addSample(float errorX, float errorY,
                                       float outputX, float outputY,
                                       float targetVelX, float targetVelY,
                                       float predX, float predY,
                                       bool saturated, SampleType type) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return;
    
    SamplePoint sample;
    sample.errorX = errorX;
    sample.errorY = errorY;
    sample.outputX = outputX;
    sample.outputY = outputY;
    sample.targetVelocityX = targetVelX;
    sample.targetVelocityY = targetVelY;
    sample.predictedX = predX;
    sample.predictedY = predY;
    sample.isSaturated = saturated;
    sample.type = type;
    sample.timestamp = std::chrono::steady_clock::now();
    
    samples_.push_back(sample);
    
    while (samples_.size() > static_cast<size_t>(config_.sampleFrames * 2)) {
        samples_.pop_front();
    }
}

void HillClimbingOptimizer::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = true;
    samples_.clear();
    iteration_ = 0;
    consecutiveNoImprovement_ = 0;
    state_ = OptimizerState::COLLECTING_BASELINE;
    currentParamIndex_ = 0;
    testingIncrease_ = true;
    
    bestMetrics_ = PerformanceMetrics();
    bestMetrics_.score = std::numeric_limits<float>::max();
    baselineMetrics_ = PerformanceMetrics();
    baselineMetrics_.score = std::numeric_limits<float>::max();
    
    adjustWeightsForStrategy();
    
    obs_log(LOG_INFO, "[AutoTune] 优化器启动: 模式=%d 策略=%d 目标误差=%.1fpx 样本数=%d",
            static_cast<int>(config_.mode), static_cast<int>(config_.strategy),
            config_.targetError, config_.sampleFrames);
}

void HillClimbingOptimizer::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    state_ = OptimizerState::STOPPED;
    
    obs_log(LOG_INFO, "[AutoTune] 优化器停止: 迭代次数=%d 最佳得分=%.4f",
            iteration_, bestMetrics_.score);
}

void HillClimbingOptimizer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
    iteration_ = 0;
    consecutiveNoImprovement_ = 0;
    state_ = OptimizerState::IDLE;
    if (config_.mode == OptimizationMode::INDEPENDENT) {
        initializeParameters();
    }
    bestMetrics_ = PerformanceMetrics();
    bestMetrics_.score = std::numeric_limits<float>::max();
    baselineMetrics_ = PerformanceMetrics();
    baselineMetrics_.score = std::numeric_limits<float>::max();
}

bool HillClimbingOptimizer::isRunning() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_;
}

OptimizerState HillClimbingOptimizer::getState() const {
    return state_;
}

const char* HillClimbingOptimizer::getStateString() const {
    switch (state_) {
        case OptimizerState::IDLE: return "空闲";
        case OptimizerState::COLLECTING_BASELINE: return "收集基线";
        case OptimizerState::EVALUATING_BASELINE: return "评估基线";
        case OptimizerState::TESTING_PARAMETER: return "测试参数";
        case OptimizerState::COLLECTING_TEST: return "收集测试数据";
        case OptimizerState::EVALUATING_TEST: return "评估测试结果";
        case OptimizerState::CONVERGED: return "已收敛";
        case OptimizerState::STOPPED: return "已停止";
        default: return "未知";
    }
}

int HillClimbingOptimizer::getCurrentIteration() const {
    return iteration_;
}

int HillClimbingOptimizer::getSampleCount() const {
    return static_cast<int>(samples_.size());
}

int HillClimbingOptimizer::getRequiredSamples() const {
    return config_.sampleFrames;
}

float HillClimbingOptimizer::getProgress() const {
    int required = getRequiredSamples();
    if (required <= 0) return 0;
    float sampleProgress = std::min(1.0f, static_cast<float>(getSampleCount()) / required);
    float iterProgress = std::min(1.0f, static_cast<float>(iteration_) / config_.maxIterations);
    return (iterProgress * 0.7f + sampleProgress * 0.3f) * 100.0f;
}

std::vector<float> HillClimbingOptimizer::getCurrentParameters() const {
    return currentParams_;
}

std::vector<float> HillClimbingOptimizer::getBestParameters() const {
    return bestParams_;
}

std::vector<std::string> HillClimbingOptimizer::getParameterNames() const {
    switch (algorithmType_) {
        case AlgorithmType::AdvancedPID:
            return {"PMin", "PMax", "D", "I", "DFilterAlpha",
                    "PredWeightX", "PredWeightY", "SmoothX", "SmoothY"};
        case AlgorithmType::StandardPID:
            return {"Kp", "Ki", "Kd", "DFilterAlpha", "SmoothX", "SmoothY", "MaxPredTime"};
        case AlgorithmType::ChrisPID:
            return {"Kp", "Ki", "Kd", "PredWeightX", "PredWeightY", "DFilterAlpha",
                    "SmoothX", "SmoothY"};
        default:
            return {};
    }
}

PerformanceMetrics HillClimbingOptimizer::getCurrentMetrics() const {
    return currentMetrics_;
}

PerformanceMetrics HillClimbingOptimizer::getBaselineMetrics() const {
    return baselineMetrics_;
}

PerformanceMetrics HillClimbingOptimizer::getBestMetrics() const {
    return bestMetrics_;
}

SamplingStats HillClimbingOptimizer::getSamplingStats() const {
    return currentMetrics_.stats;
}

void HillClimbingOptimizer::setCurrentParameters(const std::vector<float>& params) {
    if (params.size() == currentParams_.size()) {
        currentParams_ = params;
    }
}

void HillClimbingOptimizer::applyBestToUI() {
    if (bestParams_.size() > 0 && paramUpdateCallback_) {
        paramUpdateCallback_(bestParams_);
        obs_log(LOG_INFO, "[AutoTune] 已应用最优参数到界面");
    }
}

void HillClimbingOptimizer::update() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return;
    
    // 检查是否达到最大迭代次数或目标误差
    if (iteration_ >= config_.maxIterations) {
        obs_log(LOG_INFO, "[AutoTune] 达到最大迭代次数: %d", config_.maxIterations);
        state_ = OptimizerState::CONVERGED;
        running_ = false;
        return;
    }
    
    // 检查是否达到目标误差
    if (baselineMetrics_.avgError > 0 && baselineMetrics_.avgError <= config_.targetError) {
        obs_log(LOG_INFO, "[AutoTune] 达到目标误差: %.2f <= %.2f px",
                baselineMetrics_.avgError, config_.targetError);
        state_ = OptimizerState::CONVERGED;
        running_ = false;
        return;
    }
    
    switch (state_) {
        case OptimizerState::COLLECTING_BASELINE:
            if (static_cast<int>(samples_.size()) >= config_.sampleFrames) {
                state_ = OptimizerState::EVALUATING_BASELINE;
            }
            break;
            
        case OptimizerState::EVALUATING_BASELINE: {
            currentMetrics_ = calculateMetrics();
            
            // 检查有效样本比例
            int validSamples = currentMetrics_.stats.stickyCount + currentMetrics_.stats.reacqCount;
            float validRatio = static_cast<float>(validSamples) / 
                              std::max(1, static_cast<int>(samples_.size()));
            
            if (validRatio < config_.minValidSampleRatio) {
                obs_log(LOG_INFO, "[AutoTune] 有效样本比例不足: %.2f < %.2f，重新采样",
                        validRatio, config_.minValidSampleRatio);
                samples_.clear();
                state_ = OptimizerState::COLLECTING_BASELINE;
                break;
            }
            
            baselineMetrics_ = currentMetrics_;
            obs_log(LOG_INFO, "[AutoTune] 基线评估完成: 得分=%.2f err=%.1f osc=%.2f sat=%.2f p95=%.1f",
                    baselineMetrics_.score, baselineMetrics_.avgError,
                    baselineMetrics_.oscillation, baselineMetrics_.saturation,
                    baselineMetrics_.p95Error);
            
            tryNextParameter();
            break;
        }
        
        case OptimizerState::COLLECTING_TEST:
            if (static_cast<int>(samples_.size()) >= config_.sampleFrames) {
                state_ = OptimizerState::EVALUATING_TEST;
            }
            break;
            
        case OptimizerState::EVALUATING_TEST: {
            currentMetrics_ = calculateMetrics();
            
            if (currentMetrics_.score < baselineMetrics_.score) {
                float improvement = baselineMetrics_.score - currentMetrics_.score;
                baselineMetrics_ = currentMetrics_;
                
                if (currentMetrics_.score < bestMetrics_.score) {
                    bestMetrics_ = currentMetrics_;
                    bestParams_ = currentParams_;
                }
                
                consecutiveNoImprovement_ = 0;
                
                auto bounds = getParameterBounds();
                currentSteps_[currentParamIndex_] = std::min(
                    bounds[currentParamIndex_].step,
                    currentSteps_[currentParamIndex_] * (1.0f + config_.stepDecayFactor));
                
                obs_log(LOG_INFO, "[AutoTune] 迭代%d: 改进%.2f 得分=%.2f 参数[%d]%s%.4f err=%.1f osc=%.2f",
                        iteration_, improvement, currentMetrics_.score,
                        currentParamIndex_, testingIncrease_ ? "+" : "-",
                        currentDelta_, currentMetrics_.avgError, currentMetrics_.oscillation);
                
                applyParameterChange(currentParamIndex_,
                                    testingIncrease_ ? currentSteps_[currentParamIndex_] : -currentSteps_[currentParamIndex_]);
                iteration_++;
                samples_.clear();
                state_ = OptimizerState::COLLECTING_TEST;
                
            } else {
                revertParameterChange(currentParamIndex_,
                                     testingIncrease_ ? currentDelta_ : -currentDelta_);
                consecutiveNoImprovement_++;
                
                if (testingIncrease_) {
                    testingIncrease_ = false;
                    currentDelta_ = currentSteps_[currentParamIndex_];
                    applyParameterChange(currentParamIndex_, -currentDelta_);
                    samples_.clear();
                    state_ = OptimizerState::COLLECTING_TEST;
                } else {
                    moveToNextParameter();
                }
            }
            break;
        }
        
        default:
            break;
    }
    
    checkConvergence();
}

PerformanceMetrics HillClimbingOptimizer::calculateMetrics() {
    PerformanceMetrics metrics;
    if (samples_.empty()) return metrics;
    
    // 计算采样统计
    metrics.stats = calculateSamplingStats();
    
    // 收集误差值用于计算统计量
    std::vector<float> errors;
    errors.reserve(samples_.size());
    
    float sumError = 0;
    float maxError = 0;
    int saturatedCount = 0;
    
    for (const auto& sample : samples_) {
        float error = std::sqrt(sample.errorX * sample.errorX + sample.errorY * sample.errorY);
        errors.push_back(error);
        sumError += error;
        maxError = std::max(maxError, error);
        if (sample.isSaturated) saturatedCount++;
    }
    
    metrics.avgError = sumError / samples_.size();
    metrics.maxError = maxError;
    metrics.saturation = static_cast<float>(saturatedCount) / samples_.size();
    
    // P95误差
    if (!errors.empty()) {
        std::vector<float> sortedErrors = errors;
        std::sort(sortedErrors.begin(), sortedErrors.end());
        size_t p95Idx = static_cast<size_t>(sortedErrors.size() * 0.95);
        p95Idx = std::min(p95Idx, sortedErrors.size() - 1);
        metrics.p95Error = sortedErrors[p95Idx];
    }
    
    // 振荡度
    if (samples_.size() > 1) {
        float sumDelta = 0, sumDeltaSq = 0;
        for (size_t i = 1; i < samples_.size(); ++i) {
            float dx = samples_[i].outputX - samples_[i-1].outputX;
            float dy = samples_[i].outputY - samples_[i-1].outputY;
            float delta = std::sqrt(dx*dx + dy*dy);
            sumDelta += delta;
            sumDeltaSq += delta * delta;
        }
        float avgDelta = sumDelta / (samples_.size() - 1);
        float avgDeltaSq = sumDeltaSq / (samples_.size() - 1);
        metrics.oscillation = std::max(0.0f, avgDeltaSq - avgDelta * avgDelta);
    }
    
    // 平滑度
    if (samples_.size() > 1) {
        float sumChange = 0;
        for (size_t i = 1; i < samples_.size(); ++i) {
            sumChange += std::abs(samples_[i].outputX - samples_[i-1].outputX) +
                        std::abs(samples_[i].outputY - samples_[i-1].outputY);
        }
        metrics.smoothness = sumChange / (samples_.size() - 1);
    }
    
    // 收敛速度
    if (samples_.size() > 10) {
        float firstHalfErr = 0, secondHalfErr = 0;
        size_t half = samples_.size() / 2;
        for (size_t i = 0; i < half; ++i)
            firstHalfErr += errors[i];
        for (size_t i = half; i < samples_.size(); ++i)
            secondHalfErr += errors[i];
        firstHalfErr /= half;
        secondHalfErr /= (samples_.size() - half);
        if (firstHalfErr > 0)
            metrics.convergenceRate = (firstHalfErr - secondHalfErr) / firstHalfErr;
    }
    
    // 预测准确度
    metrics.predAccuracy = calculatePredAccuracy();
    
    // 稳定时间（简化计算）
    metrics.settleTime = calculateSettleTime();
    
    // 综合得分
    metrics.score = calculateScore(metrics);
    
    return metrics;
}

float HillClimbingOptimizer::calculateScore(const PerformanceMetrics& m) {
    const auto& w = config_.weights;
    
    float normMaxErr = m.maxError / 500.0f;
    float normP95Err = m.p95Error / 300.0f;
    float normOsc = std::min(m.oscillation / 100.0f, 1.0f);
    float normSat = m.saturation;
    float normSmooth = std::min(m.smoothness / 50.0f, 1.0f);
    float normConv = std::max(0.0f, m.convergenceRate);
    
    float score = w.weightAvgError * m.avgError +
                  w.weightMaxError * normMaxErr * 100 +
                  w.weightP95Error * normP95Err * 80 +
                  w.weightOscillation * normOsc * 100 +
                  w.weightSaturation * normSat * 150 +
                  w.weightSmoothness * normSmooth * 100 -
                  w.weightConvergence * normConv * 50;
    
    return score;
}

std::vector<ParameterBounds> HillClimbingOptimizer::getParameterBounds() {
    switch (algorithmType_) {
        case AlgorithmType::AdvancedPID: return config_.advancedPidBounds;
        case AlgorithmType::StandardPID: return config_.standardPidBounds;
        case AlgorithmType::ChrisPID: return config_.chrisPidBounds;
        default: return config_.advancedPidBounds;
    }
}

void HillClimbingOptimizer::initializeParameters() {
    auto bounds = getParameterBounds();
    currentParams_.resize(bounds.size());
    currentSteps_.resize(bounds.size());
    
    for (size_t i = 0; i < bounds.size(); ++i) {
        currentParams_[i] = (bounds[i].minVal + bounds[i].maxVal) / 2.0f;
        currentSteps_[i] = bounds[i].step;
    }
    bestParams_ = currentParams_;
}

void HillClimbingOptimizer::tryNextParameter() {
    auto bounds = getParameterBounds();
    int attempts = 0;
    
    while (attempts < static_cast<int>(bounds.size())) {
        currentDelta_ = currentSteps_[currentParamIndex_];
        testingIncrease_ = true;
        
        if (currentParams_[currentParamIndex_] + currentDelta_ <= bounds[currentParamIndex_].maxVal) {
            applyParameterChange(currentParamIndex_, currentDelta_);
            samples_.clear();
            state_ = OptimizerState::COLLECTING_TEST;
            obs_log(LOG_INFO, "[AutoTune] 测试参数[%d]: %s%.4f (当前=%.4f)",
                    currentParamIndex_, "+", currentDelta_, currentParams_[currentParamIndex_]);
            return;
        }
        
        if (currentParams_[currentParamIndex_] - currentDelta_ >= bounds[currentParamIndex_].minVal) {
            testingIncrease_ = false;
            applyParameterChange(currentParamIndex_, -currentDelta_);
            samples_.clear();
            state_ = OptimizerState::COLLECTING_TEST;
            obs_log(LOG_INFO, "[AutoTune] 测试参数[%d]: %s%.4f (当前=%.4f)",
                    currentParamIndex_, "-", currentDelta_, currentParams_[currentParamIndex_]);
            return;
        }
        
        currentParamIndex_ = (currentParamIndex_ + 1) % static_cast<int>(bounds.size());
        attempts++;
    }
    
    obs_log(LOG_INFO, "[AutoTune] 所有参数已达边界，优化完成");
    state_ = OptimizerState::CONVERGED;
    running_ = false;
}

void HillClimbingOptimizer::applyParameterChange(int index, float delta) {
    auto bounds = getParameterBounds();
    float newVal = std::clamp(currentParams_[index] + delta,
                             bounds[index].minVal, bounds[index].maxVal);
    currentParams_[index] = newVal;
    
    if (paramUpdateCallback_) {
        paramUpdateCallback_(currentParams_);
    }
}

void HillClimbingOptimizer::revertParameterChange(int index, float delta) {
    currentParams_[index] -= delta;
    
    if (paramUpdateCallback_) {
        paramUpdateCallback_(currentParams_);
    }
}

void HillClimbingOptimizer::moveToNextParameter() {
    auto bounds = getParameterBounds();
    currentParamIndex_ = (currentParamIndex_ + 1) % static_cast<int>(bounds.size());
    testingIncrease_ = true;
    currentDelta_ = currentSteps_[currentParamIndex_];
    tryNextParameter();
}

void HillClimbingOptimizer::checkConvergence() {
    auto bounds = getParameterBounds();
    if (consecutiveNoImprovement_ >= static_cast<int>(bounds.size()) * 2) {
        for (size_t i = 0; i < currentSteps_.size(); ++i) {
            currentSteps_[i] = std::max(bounds[i].minStep,
                                        currentSteps_[i] * (1.0f - config_.stepDecayFactor));
        }
        consecutiveNoImprovement_ = 0;
        obs_log(LOG_INFO, "[AutoTune] 步长衰减");
        
        bool allMinStep = true;
        for (size_t i = 0; i < currentSteps_.size(); ++i) {
            if (currentSteps_[i] > bounds[i].minStep * 1.5f) {
                allMinStep = false;
                break;
            }
        }
        
        if (allMinStep) {
            obs_log(LOG_INFO, "[AutoTune] 所有步长已达最小值，收敛");
            state_ = OptimizerState::CONVERGED;
            running_ = false;
        }
    }
}

void HillClimbingOptimizer::adjustWeightsForStrategy() {
    config_.weights.setForStrategy(config_.strategy);
}

SamplingStats HillClimbingOptimizer::calculateSamplingStats() {
    SamplingStats stats;
    
    for (const auto& sample : samples_) {
        switch (sample.type) {
            case SampleType::STICKY: stats.stickyCount++; break;
            case SampleType::AMB: stats.ambCount++; break;
            case SampleType::REACQ: stats.reacqCount++; break;
            case SampleType::SWITCH: stats.switchCount++; break;
        }
    }
    
    return stats;
}

float HillClimbingOptimizer::calculateP95Error() {
    if (samples_.empty()) return 0;
    
    std::vector<float> errors;
    errors.reserve(samples_.size());
    for (const auto& s : samples_)
        errors.push_back(std::sqrt(s.errorX*s.errorX + s.errorY*s.errorY));
    
    std::sort(errors.begin(), errors.end());
    size_t idx = static_cast<size_t>(errors.size() * 0.95);
    idx = std::min(idx, errors.size() - 1);
    return errors[idx];
}

float HillClimbingOptimizer::calculateSaturation() {
    if (samples_.empty()) return 0;
    int satCount = 0;
    for (const auto& s : samples_)
        if (s.isSaturated) satCount++;
    return static_cast<float>(satCount) / samples_.size();
}

float HillClimbingOptimizer::calculateSettleTime() {
    if (samples_.size() < 10) return 0;
    
    float threshold = 5.0f;
    int settleStart = -1;
    
    for (int i = static_cast<int>(samples_.size()) - 1; i >= 0; --i) {
        float err = std::sqrt(samples_[i].errorX * samples_[i].errorX +
                               samples_[i].errorY * samples_[i].errorY);
        if (err > threshold) {
            settleStart = i;
            break;
        }
    }
    
    if (settleStart < 0) return 1.0f;
    
    int settleFrames = static_cast<int>(samples_.size()) - 1 - settleStart;
    return std::min(1.0f, static_cast<float>(settleFrames) / samples_.size());
}

float HillClimbingOptimizer::calculatePredAccuracy() {
    if (samples_.empty()) return 0.5f;
    
    float totalPredErr = 0;
    int count = 0;
    
    for (const auto& s : samples_) {
        if (s.predictedX != 0 || s.predictedY != 0) {
            float predErr = std::sqrt((s.errorX - s.predictedX) * (s.errorX - s.predictedX) +
                                     (s.errorY - s.predictedY) * (s.errorY - s.predictedY));
            totalPredErr += predErr;
            count++;
        }
    }
    
    if (count == 0) return 0.5f;
    float avgPredErr = totalPredErr / count;
    float avgError = calculateP95Error();
    
    if (avgError <= 0) return 0.5f;
    return std::max(0.0f, std::min(1.0f, 1.0f - avgPredErr / avgError));
}
