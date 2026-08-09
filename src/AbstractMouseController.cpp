#ifdef _WIN32

#include "AbstractMouseController.hpp"
#include <obs-module.h>
#include <plugin-support.h>
#include <cmath>
#include <algorithm>

AbstractMouseController::AbstractMouseController()
    : cachedScreenWidth(0)
    , cachedScreenHeight(0)
    , isMoving(false)
    , pidPreviousErrorX(0.0f)
    , pidPreviousErrorY(0.0f)
    , filteredDeltaErrorX(0.0f)
    , filteredDeltaErrorY(0.0f)
    , previousErrorX(0.0f)
    , previousErrorY(0.0f)
    , previousTargetX(0.0f)
    , previousTargetY(0.0f)
    , targetVelocityX(0.0f)
    , targetVelocityY(0.0f)
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
    , integralX(0.0f)
    , integralY(0.0f)
    , integralGainX(0.0f)
    , integralGainY(0.0f)
    , adaptivePGainX(1.0f)
    , adaptivePGainY(1.0f)
    , adaptiveIGainX(1.0f)
    , adaptiveIGainY(1.0f)
    , kf2X(0.0f, 1.0f, 0.0f, 1.0f)
    , kf2Y(0.0f, 1.0f, 0.0f, 1.0f)
    , kalmanOutputX(0.1f, 1.0f, 0.0f, 1.0f)
    , kalmanOutputY(0.1f, 1.0f, 0.0f, 1.0f)
    , kf3X_x(0.0f)
    , kf3X_P(1.0f)
    , kf3Y_x(0.0f)
    , kf3Y_P(1.0f)
    , lastOutputX(0.0f)
    , lastOutputY(0.0f)
    , lockedTrackId(-1)
    , lastRecoilTime(std::chrono::steady_clock::now())
    , isFiring(false)
    , lastTickTime(std::chrono::steady_clock::now())
    , deltaTime(0.016f)
    , hotkeyPressStartTime(std::chrono::steady_clock::now())
    , yUnlockActive(false)
    , lastAutoTriggerTime(std::chrono::steady_clock::now())
    , autoTriggerFireStartTime(std::chrono::steady_clock::now())
    , autoTriggerDelayStartTime(std::chrono::steady_clock::now())
    , autoTriggerHolding(false)
    , autoTriggerWaitingForDelay(false)
    , currentFireDuration(50)
    , randomGenerator(std::random_device{}())
    , currentTargetTrackId(-1)
    , targetLockStartTime(std::chrono::steady_clock::now())
    , currentTargetDistance(0.0f)
    , pendingTargetTrackId(-1)
    , pendingTargetStartTime(std::chrono::steady_clock::now())
    , pendingTargetScore(0.0f)
    , currentTargetScore(0.0f)
    , bezierPhase(0.0f)
    , pidDataCallback_(nullptr)
    , enableNeuralPath_(false)
    , neuralPathInitialized_(false)
    , enableNeuralPathDebug_(false)
    , neuralPathIndex_(0)
    , lastNeuralTargetX_(0.0f)
    , lastNeuralTargetY_(0.0f)
    , externalPidInitialized_(false)
    , avgInferenceTimeMs_(0.0f)
{
    startPos = { 0, 0 };
    targetPos = { 0, 0 };
    cachedScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    cachedScreenHeight = GetSystemMetrics(SM_CYSCREEN);
}

void AbstractMouseController::updateConfig(const MouseControllerConfig& newConfig)
{
    std::lock_guard<std::mutex> lock(mutex);
    bool configChanged = (config.enableMouseControl != newConfig.enableMouseControl ||
                          config.autoTriggerEnabled != newConfig.autoTriggerEnabled ||
                          config.autoTriggerFireDuration != newConfig.autoTriggerFireDuration ||
                          config.autoTriggerInterval != newConfig.autoTriggerInterval);
    
    // 检测外部PID参数变化
    bool externalPidParamsChanged = (
        config.externalKpX != newConfig.externalKpX ||
        config.externalKiX != newConfig.externalKiX ||
        config.externalKdX != newConfig.externalKdX ||
        config.externalKpY != newConfig.externalKpY ||
        config.externalKiY != newConfig.externalKiY ||
        config.externalKdY != newConfig.externalKdY ||
        config.externalPredictX != newConfig.externalPredictX ||
        config.externalPredictY != newConfig.externalPredictY ||
        config.externalRateX != newConfig.externalRateX ||
        config.externalRateY != newConfig.externalRateY
    );
    
    config = newConfig;
    
    // 外部PID参数变化时重置初始化状态
    if (externalPidParamsChanged) {
        externalPidInitialized_ = false;
        externalPidX.reset();
        externalPidY.reset();
    }
    
    config.bezierCurvature = std::clamp(config.bezierCurvature, 0.0f, 1.0f);
    config.bezierRandomness = std::clamp(config.bezierRandomness, 0.0f, 0.5f);
    
    // 更新DerivativePredictor参数
    predictor.setMaxPredictionTime(config.maxPredictionTime);
    
    // 更新神经网络轨迹生成器配置
    enableNeuralPath_ = config.enableNeuralPath;

    // 更新GhostTracker配置（只在config变更时设置一次）
    {
        GhostTracker::Config ghostConfig;
        ghostConfig.enabled = config.enableGhostTracker;
        ghostConfig.curvature = config.ghostCurvature;
        ghostConfig.noiseIntensity = config.ghostNoiseIntensity;
        ghostConfig.verticalSnapRatio = config.ghostVerticalSnapRatio;
        ghostConfig.noiseFreq = config.ghostNoiseFreq;
        ghostTracker.setConfig(ghostConfig);
    }

    // IMM交互多模型滤波器配置同步
    {
        IMMFilter::Config immCfg;
        immCfg.enabled = config.immFilterEnabled;
        immCfg.processNoisePos = config.immProcessNoisePos;
        immCfg.processNoiseVel = config.immProcessNoiseVel;
        immCfg.processNoiseAcc = config.immProcessNoiseAcc;
        immCfg.processNoiseTurn = config.immProcessNoiseTurn;
        immCfg.measurementNoiseX = config.immMeasurementNoiseX;
        immCfg.measurementNoiseY = config.immMeasurementNoiseY;
        immCfg.activeModels = config.immActiveModels;
        immFilter.setConfig(immCfg);
    }

    // 变分贝叶斯鲁棒滤波器配置同步
    {
        VariationalBayesFilter::Config vbCfg;
        vbCfg.enabled = config.useVbFilter;
        vbCfg.processNoisePos = config.vbProcessNoisePos;
        vbCfg.processNoiseVel = config.vbProcessNoiseVel;
        vbCfg.measurementNoiseX = config.vbMeasurementNoiseX;
        vbCfg.measurementNoiseY = config.vbMeasurementNoiseY;
        vbCfg.nu0 = config.vbNu0;
        vbCfg.rho = config.vbRho;
        vbCfg.iterations = config.vbIterations;
        vbCfg.outlierGate = config.vbOutlierGate;
        vbFilter.setConfig(vbCfg);
    }
    enableNeuralPathDebug_ = config.enableNeuralPathDebug;
    initializeNeuralPathIfNeeded();
    
    // Smith预估器配置同步
    {
        SmithPredictor::Config smithCfg;
        smithCfg.enabled = config.smithPredictorEnabled;
        smithCfg.modelGainK = config.smithModelGain;
        smithCfg.modelTimeConstT = config.smithModelTau > 0.0f ? config.smithModelTau : 0.05f;
        // 自动tau：实测推理延迟×2（覆盖渲染+显示+鼠标延迟）+ 最小30ms地板值
        if (config.smithAutoTau) {
            float baseTau = avgInferenceTimeMs_ > 0.0f
                ? avgInferenceTimeMs_ / 1000.0f * 2.0f   // 2倍安全系数
                : 0.060f;                                 // 未推理时默认60ms
            // 执行延迟（鼠标→游戏→画面反馈）加到输出端模型延迟上
            baseTau += config.mouseLatencyMs * 0.001f;
            smithCfg.delayTau = std::max(baseTau, 0.030f); // 最小30ms地板
        } else {
            smithCfg.delayTau = config.smithModelTau;
        }
        // 仅在enabled状态或关键参数变化时记录日志，避免刷屏
        static SmithPredictor::Config lastLoggedCfg = {};
        bool cfgChanged = (smithCfg.enabled != lastLoggedCfg.enabled)
                       || (smithCfg.modelGainK != lastLoggedCfg.modelGainK)
                       || (smithCfg.modelTimeConstT != lastLoggedCfg.modelTimeConstT)
                       || (std::abs(smithCfg.delayTau - lastLoggedCfg.delayTau) > 0.001f);
        if (smithCfg.enabled && cfgChanged) {
            obs_log(LOG_INFO, "[%s] Smith诊断: enabled=%d K=%.2f T=%.4fs tau=%.4fs autoTau=%d avgInferMs=%.2f",
                    getLogPrefix(), smithCfg.enabled, smithCfg.modelGainK, smithCfg.modelTimeConstT,
                    smithCfg.delayTau, config.smithAutoTau, avgInferenceTimeMs_);
            lastLoggedCfg = smithCfg;
        }
        smithPredictor.setConfig(smithCfg);
    }

    // 自适应PID控制器配置同步
    {
        AdaptivePIDController::Config adaptiveCfg;
        adaptiveCfg.kp = config.adaptivePidKp;
        adaptiveCfg.ki = config.adaptivePidKi;
        adaptiveCfg.kd = config.adaptivePidKd;
        adaptiveCfg.deadZone = config.adaptivePidDeadZone;
        adaptiveCfg.integralLimit = config.adaptivePidIntegralLimit;
        adaptiveCfg.integralDeadzone = config.adaptivePidIntegralDeadzone;
        adaptiveCfg.integralGainThreshold = config.adaptivePidIntegralGainThreshold;
        adaptiveCfg.integralGainRate = config.adaptivePidIntegralGainRate;
        adaptiveCfg.integralGainVelocityLimit = config.adaptivePidIntegralGainVelocityLimit;
        adaptiveCfg.outputLimit = config.adaptivePidOutputLimit;
        adaptivePidX_.configure(adaptiveCfg);
        adaptivePidY_.configure(adaptiveCfg);
    }
    
    if (configChanged) {
        obs_log(LOG_INFO, "[%s] Config updated: enableMouseControl=%d, autoTriggerEnabled=%d, fireDuration=%dms, interval=%dms",
                getLogPrefix(), config.enableMouseControl, config.autoTriggerEnabled, 
                config.autoTriggerFireDuration, config.autoTriggerInterval);
    }
}

void AbstractMouseController::initializeNeuralPathIfNeeded()
{
    if (enableNeuralPath_ && !neuralPathInitialized_) {
        obs_log(LOG_INFO, "[%s] Neural path PREDICTOR INITIALIZING: width=%d, height=%d, radius=%d, step=%.1f, points=%d",
                getLogPrefix(),
                config.inferenceFrameWidth > 0 ? config.inferenceFrameWidth : 1920,
                config.inferenceFrameHeight > 0 ? config.inferenceFrameHeight : 1080,
                config.neuralTargetRadius,
                config.neuralMouseStepSize,
                config.neuralPathPoints);
                
        neuralPathPredictor_.init(
            config.inferenceFrameWidth > 0 ? config.inferenceFrameWidth : 1920,
            config.inferenceFrameHeight > 0 ? config.inferenceFrameHeight : 1080,
            config.neuralTargetRadius,
            config.neuralMouseStepSize,
            config.neuralPathPoints
        );
        neuralPathInitialized_ = true;
        obs_log(LOG_INFO, "[%s] Neural path predictor INITIALIZED SUCCESSFULLY", getLogPrefix());
    } else if (enableNeuralPath_ && neuralPathInitialized_) {
        if (enableNeuralPathDebug_) obs_log(LOG_DEBUG, "[%s] Neural path predictor already initialized (skipping)", getLogPrefix());
    } else {
        if (enableNeuralPathDebug_) obs_log(LOG_DEBUG, "[%s] Neural path DISABLED (enableNeuralPath=%d)", getLogPrefix(), enableNeuralPath_ ? 1 : 0);
    }
}

MouseControllerConfig AbstractMouseController::getConfig() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return config;
}

void AbstractMouseController::setDetections(const std::vector<Detection>& detections)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
    lastDetectionsUpdate_ = std::chrono::steady_clock::now();
}

void AbstractMouseController::setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
    config.inferenceFrameWidth = frameWidth;
    config.inferenceFrameHeight = frameHeight;
    config.cropOffsetX = cropX;
    config.cropOffsetY = cropY;
    lastDetectionsUpdate_ = std::chrono::steady_clock::now();
}

void AbstractMouseController::setDetectionsWithFrameSize(std::vector<Detection>&& detections, int frameWidth, int frameHeight, int cropX, int cropY)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = std::move(detections);
    config.inferenceFrameWidth = frameWidth;
    config.inferenceFrameHeight = frameHeight;
    config.cropOffsetX = cropX;
    config.cropOffsetY = cropY;
    lastDetectionsUpdate_ = std::chrono::steady_clock::now();
}

void AbstractMouseController::setInferenceTimeMs(float ms)
{
    std::lock_guard<std::mutex> lock(mutex);
    // 仅在变化超过1ms时记录日志，避免刷屏
    if (std::abs(ms - avgInferenceTimeMs_) > 1.0f) {
        obs_log(LOG_DEBUG, "[%s] Smith诊断: avgInferenceTimeMs 更新 %.2f -> %.2f ms",
                getLogPrefix(), avgInferenceTimeMs_, ms);
    }
    avgInferenceTimeMs_ = ms;
}

// 默认: 查本机虚拟键状态。MAKCU 等硬件控制器覆写 (双机场景经设备上报).
bool AbstractMouseController::isPhysicalButtonPressed(int vk)
{
    if (vk <= 0) {
        return false;
    }
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

void AbstractMouseController::tick()
{
    std::lock_guard<std::mutex> lock(mutex);

    if (!config.enableMouseControl) {
        smoothedTargetX_ = -1.0f;
        smoothedTargetY_ = -1.0f;
        if (autoTriggerHolding) {
            performClickUp();
            autoTriggerHolding = false;
        }
        autoTriggerWaitingForDelay = false;
        isMoving = false;
        return;
    }

    bool hotkeyPressed = isPhysicalButtonPressed(config.hotkeyVirtualKey);
    bool shouldAim = config.continuousAimEnabled || hotkeyPressed;

    // ========================================
    // DEBUG_LOG: 热键/AIM入口状态 (每30帧一次避免刷屏)
    // ========================================
    static int s_hotkeyLogCounter = 0;
    if (s_hotkeyLogCounter++ % 30 == 0) {
        obs_log(LOG_INFO, "[%s] AIM_ENTRY: hotkey=%d(0x%02X) pressed=%d continuous=%d shouldAim=%d isMoving=%d locked=%d dets=%zu infW=%d infH=%d fov=%d",
                getLogPrefix(),
                config.hotkeyVirtualKey, config.hotkeyVirtualKey & 0xFF,
                hotkeyPressed ? 1 : 0,
                config.continuousAimEnabled ? 1 : 0,
                shouldAim ? 1 : 0,
                isMoving ? 1 : 0,
                lockedTrackId,
                currentDetections.size(),
                config.inferenceFrameWidth, config.inferenceFrameHeight,
                config.fovRadiusPixels);
    }

    if (!shouldAim) {
        smoothedTargetX_ = -1.0f;
        smoothedTargetY_ = -1.0f;
        if (isMoving) {
            obs_log(LOG_INFO, "[%s] AIM_EXIT: shouldAim=false isMoving=true → resetPid+Motion", getLogPrefix());
            isMoving = false;
            resetPidState();
            resetMotionState();
        } else {
            // 即使isMoving为false（目标丢失导致），热键松开时也要重置所有状态
            integralX = 0.0f;
            integralY = 0.0f;
            // 保持与 resetPidState 一致：integralGain 给 0.5 避免下一次按热键冷启动归零
            integralGainX = 0.5f;
            integralGainY = 0.5f;
            lockedTrackId = -1;  // 重置目标锁定
            lockMissCount_ = 0;
        }
        yUnlockActive = false;
        releaseAutoTrigger();
        return;
    }

    if (!config.continuousAimEnabled) {
        if (!wasHotkeyPressed_ && hotkeyPressed) {
            hotkeyPressStartTime = std::chrono::steady_clock::now();
            yUnlockActive = false;
        }
        wasHotkeyPressed_ = hotkeyPressed;

        if (config.yUnlockEnabled) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - hotkeyPressStartTime).count();
            if (elapsed >= config.yUnlockDelayMs) {
                yUnlockActive = true;
            }
        } else {
            yUnlockActive = false;
        }
    } else {
        yUnlockActive = false;
    }

    auto now = std::chrono::steady_clock::now();
    // High-resolution dt in seconds (do NOT truncate to whole milliseconds)
    deltaTime = std::chrono::duration<float>(now - lastTickTime).count();
    deltaTime = std::max(0.001f, std::min(deltaTime, 0.05f));
    lastTickTime = now;

    Detection* target = selectTarget();
    if (!target) {
        // DEBUG_LOG: target=nullptr 情况
        static int s_nullTargetLog = 0;
        if (s_nullTargetLog++ % 30 == 0) {
            obs_log(LOG_INFO, "[%s] TARGET_NULL: locked=%d missCnt=%d/%d dets=%zu isMoving=%d → %s",
                    getLogPrefix(),
                    lockedTrackId, lockMissCount_, kMaxLockMissFrames,
                    currentDetections.size(), isMoving ? 1 : 0,
                    (isMoving && lockMissCount_ >= kMaxLockMissFrames) ? "RESET(真丢)" : "冻结(宽限内)");
        }
        if (isMoving) {
            if (lockMissCount_ >= kMaxLockMissFrames) {
                // 真丢（超宽限）：清空预测/滤波/Smith，避免旧目标状态污染新目标
                // 积分仍由 resetPidState 在热键松开时彻底清零
                isMoving = false;
                predictor.reset();
                immFilter.reset();
                vbFilter.reset();
                oneEuroX_.reset();
                oneEuroY_.reset();
                oneEuroLockedTrackId_ = -1;
                smithPredictor.reset();
                adaptivePidX_.reset();
                adaptivePidY_.reset();
                resetMotionState();
            }
            // 宽限内（missCnt<8）：不 reset，滤波器状态自然冻结（本帧不 predict/update），
            // 目标回来(同ID)从旧速度/位置续跟。但冻结期速度估计仍指向旧方向，
            // 恢复后前几帧压低预测权重，防止旧速度持续注入推偏
            freezeRecoverFrames_ = kMaxLockMissFrames;
        }
        // 目标丢失时重置自动扳机
        if (autoTriggerHolding) {
            auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
        }
        return;
    }
    
    // 日志：目标选择
    targetFrameCount_++;
    if (targetFrameCount_ % 60 == 1 && enableNeuralPathDebug_) {
        obs_log(LOG_INFO, "[%s] TARGET SELECTED: classId=%d, conf=%.2f, center=(%.3f,%.3f), wh=(%.3f,%.3f)",
                getLogPrefix(), target->classId, target->confidence,
                target->centerX, target->centerY, target->width, target->height);
    }
    
    // 诊断：检测框稳定性追踪
    float centerDeltaX = std::abs(target->centerX - lastCenterX_);
    float centerDeltaY = std::abs(target->centerY - lastCenterY_);
    float maxDeltaThisFrame = std::max(centerDeltaX, centerDeltaY);
    if (maxDeltaThisFrame > maxCenterDelta_) maxCenterDelta_ = maxDeltaThisFrame;
    lastCenterX_ = target->centerX;
    lastCenterY_ = target->centerY;

    if (targetFrameCount_ % 120 == 1 && enableNeuralPathDebug_) {
        obs_log(LOG_INFO, "[%s] DETECTION STABILITY: centerDelta=(%.4f,%.4f) maxDelta=%.4f (120帧内)",
                getLogPrefix(), centerDeltaX, centerDeltaY, maxCenterDelta_);
        maxCenterDelta_ = 0;
    }

    float fovCenterX = config.inferenceFrameWidth / 2.0f;
    float fovCenterY = config.inferenceFrameHeight / 2.0f;
    
    // 准星位置优先：如果有准星检测结果，用准星位置作为瞄准起点
    if (aimOriginX_ >= 0.0f && aimOriginY_ >= 0.0f) {
        fovCenterX = aimOriginX_;
        fovCenterY = aimOriginY_;
    }
    
    if (targetFrameCount_ % 60 == 1 && enableNeuralPathDebug_) {
        obs_log(LOG_INFO, "[%s] NEURAL PATH STATUS: enabled=%d, initialized=%d, hasDetections=%zu",
                getLogPrefix(), enableNeuralPath_ ? 1 : 0, neuralPathInitialized_ ? 1 : 0,
                currentDetections.size());
    }

    float targetPixelX = target->centerX * config.inferenceFrameWidth;
    float yOffsetPixels = config.targetYOffset * 0.01f * target->height * config.inferenceFrameHeight;
    float targetPixelY = target->centerY * config.inferenceFrameHeight - yOffsetPixels;
    float targetPixelW = target->width * config.inferenceFrameWidth;
    float targetPixelH = target->height * config.inferenceFrameHeight;
    
    // 神经网络轨迹生成
    if (enableNeuralPath_) {
        neuralLogCount_++;
        if (neuralLogCount_ % 60 == 1 && enableNeuralPathDebug_) {
            obs_log(LOG_INFO, "[%s] NEURAL PATH ACTIVE: using neural trajectory (enabled=%d, initialized=%d)",
                    getLogPrefix(), enableNeuralPath_ ? 1 : 0, neuralPathInitialized_ ? 1 : 0);
        }
        
        // 计算目标相对位置
        double relativeTargetX = targetPixelX - fovCenterX;
        double relativeTargetY = targetPixelY - fovCenterY;
        
        frameCount_++;
        if (frameCount_ % 30 == 1 && enableNeuralPathDebug_) {
            obs_log(LOG_INFO, "[%s] NeuralPath FRAME=%d: target=(%.1f,%.1f) fovCenter=(%.1f,%.1f) relative=(%.1f,%.1f) initialized=%d",
                    getLogPrefix(), frameCount_, targetPixelX, targetPixelY, fovCenterX, fovCenterY,
                    relativeTargetX, relativeTargetY, neuralPathInitialized_ ? 1 : 0);
        }
        
        // 检查目标是否变化（阈值判断）
        float targetChangeThreshold = 15.0f; // 目标变化阈值（像素）
        bool targetChanged = std::abs(targetPixelX - lastNeuralTargetX_) > targetChangeThreshold ||
                            std::abs(targetPixelY - lastNeuralTargetY_) > targetChangeThreshold;
        
        // 检查是否已到达目标（相对位置接近0）
        float reachThreshold = static_cast<float>(config.neuralTargetRadius); // 使用配置的目标半径
        bool targetReached = std::abs(relativeTargetX) < reachThreshold &&
                            std::abs(relativeTargetY) < reachThreshold;
        
        // 只有目标变化或轨迹执行完毕且未到达目标时，才重新生成轨迹
        if (targetChanged || (neuralPathIndex_ >= neuralPathPoints_.size() && !targetReached)) {
            if (enableNeuralPathDebug_) obs_log(LOG_INFO, "[%s] NeuralPath REGENERATING: targetChanged=%d, targetReached=%d, index=%zu, size=%zu, lastTarget=(%.1f,%.1f)",
                    getLogPrefix(), targetChanged ? 1 : 0, targetReached ? 1 : 0, neuralPathIndex_, neuralPathPoints_.size(),
                    lastNeuralTargetX_, lastNeuralTargetY_);
                    
            neuralPathPoints_ = neuralPathPredictor_.moveTo(relativeTargetX, relativeTargetY);
            neuralPathIndex_ = 0;
            lastNeuralTargetX_ = targetPixelX;
            lastNeuralTargetY_ = targetPixelY;
            
            if (enableNeuralPathDebug_) obs_log(LOG_INFO, "[%s] NeuralPath GENERATED %zu points", getLogPrefix(), neuralPathPoints_.size());
            if (enableNeuralPathDebug_ && !neuralPathPoints_.empty()) {
                obs_log(LOG_INFO, "[%s] NeuralPath FIRST POINT: (%.1f,%.1f)", getLogPrefix(),
                        neuralPathPoints_[0].first, neuralPathPoints_[0].second);
            }
        }
        
        // 执行轨迹移动 - 每帧可消费多个路径点加速到达
        if (!neuralPathPoints_.empty() && neuralPathIndex_ < neuralPathPoints_.size()) {
            int dx = 0, dy = 0;
            int consumeCount = std::min(config.neuralConsumePerFrame,
                static_cast<int>(neuralPathPoints_.size() - neuralPathIndex_));
            for (int i = 0; i < consumeCount; i++) {
                dx += static_cast<int>(std::round(neuralPathPoints_[neuralPathIndex_].first));
                dy += static_cast<int>(std::round(neuralPathPoints_[neuralPathIndex_].second));
                neuralPathIndex_++;
            }

            moveFrameCount_++;
            if (moveFrameCount_ % 10 == 1 && enableNeuralPathDebug_) {
                obs_log(LOG_INFO, "[%s] NeuralPath MOVE: consumed=%d, dx=%d, dy=%d, remaining=%zu",
                        getLogPrefix(), consumeCount, dx, dy,
                        neuralPathPoints_.size() - neuralPathIndex_);
            }
            
            if (pidDataCallback_) {
                PidDebugData data;
                data.errorX = static_cast<float>(relativeTargetX);
                data.errorY = static_cast<float>(relativeTargetY);
                data.outputX = static_cast<float>(dx);
                data.outputY = static_cast<float>(dy);
                data.targetX = targetPixelX;
                data.targetY = targetPixelY;
                data.algorithmType = 5; // NeuralPath
                pidDataCallback_(data);
            }
            
            moveMouse(dx, dy);
            return;
        }
    }

    if (deltaTime > 0.001f) {
        targetVelocityX = (targetPixelX - previousTargetX) / deltaTime;
        targetVelocityY = (targetPixelY - previousTargetY) / deltaTime;
    }
    previousTargetX = targetPixelX;
    previousTargetY = targetPixelY;

    // 目标中心自适应 EMA 平滑: 低分辨率/双机画面 (320x320) 检测框每帧跳几十像素,
    // 平滑后 error 稳定 → 输出平滑不突跳, 且不衰减输出 (真实目标移动仍能跟上)。
    // alpha 按跳动量自适应: 大跳(目标真动/换目标)高 alpha 快跟, 小幅抖(检测噪点)低 alpha 压平。
    // 注意: 平滑有固定滞后(稳态滞后 = v*(1-α)/α), 目标急转弯时准星拖着旧方向走。
    // 仅低分辨率/双机场景开启 (aimSmoothingEnabled), 高分辨率本地推理默认关。
    if (config.aimSmoothingEnabled) {
    {
        // 目标切换 → 强制重置平滑状态（旧目标的平滑位置属于旧目标，不能带入）
        if (lockedTrackId != smoothedTrackId_) {
            smoothedTrackId_ = lockedTrackId;
            smoothedTargetX_ = -1.0f;
            smoothedTargetY_ = -1.0f;
        }
        if (smoothedTargetX_ < 0.0f) {
            smoothedTargetX_ = targetPixelX;
            smoothedTargetY_ = targetPixelY;
        } else {
            float jump = std::max(std::abs(targetPixelX - smoothedTargetX_),
                                  std::abs(targetPixelY - smoothedTargetY_));
            float alpha = 0.4f;
            if (jump > 60.0f) {
                // 目标跳很远: 换目标或真的大移动, 直接跟 (避免永久滞后)
                smoothedTargetX_ = targetPixelX;
                smoothedTargetY_ = targetPixelY;
                alpha = 1.0f;
            } else if (jump > 30.0f) {
                alpha = 0.8f;
            } else if (jump < 6.0f) {
                alpha = 0.25f; // 小幅抖: 重平滑压噪
            }
            smoothedTargetX_ += alpha * (targetPixelX - smoothedTargetX_);
            smoothedTargetY_ += alpha * (targetPixelY - smoothedTargetY_);
            targetPixelX = smoothedTargetX_;
            targetPixelY = smoothedTargetY_;
        }
    }
    }

    float errorX = targetPixelX - fovCenterX + config.screenOffsetX;
    float errorY = targetPixelY - fovCenterY + config.screenOffsetY;

    // OneEuro：自适应截止，静止压抖、快移少滞后（Casiez 2012）
    if (config.useOneEuroFilter) {
        if (lockedTrackId != oneEuroLockedTrackId_) {
            oneEuroX_.reset();
            oneEuroY_.reset();
            oneEuroLockedTrackId_ = lockedTrackId;
        }
        oneEuroX_.setMinCutoff(config.oneEuroMinCutoff);
        oneEuroX_.setBeta(config.oneEuroBeta);
        oneEuroX_.setDCutoff(config.oneEuroDCutoff);
        oneEuroY_.setMinCutoff(config.oneEuroMinCutoff);
        oneEuroY_.setBeta(config.oneEuroBeta);
        oneEuroY_.setDCutoff(config.oneEuroDCutoff);
        float dtEuro = (deltaTime > 1e-4f) ? deltaTime : (1.0f / 60.0f);
        errorX = oneEuroX_.filter(errorX, dtEuro);
        errorY = oneEuroY_.filter(errorY, dtEuro);
    }

    float distanceSquared = errorX * errorX + errorY * errorY;
    float deadZoneSquared = config.deadZonePixels * config.deadZonePixels;
    
    deadZoneFrameCount_++;
    if (deadZoneFrameCount_ % 60 == 1 && enableNeuralPathDebug_) {
        float distance = std::sqrt(distanceSquared);
        obs_log(LOG_INFO, "[%s] DEADZONE CHECK: error=(%.1f,%.1f) distance=%.1f deadZone=%.1f inDeadZone=%d",
                getLogPrefix(), errorX, errorY, distance, config.deadZonePixels,
                distanceSquared < deadZoneSquared ? 1 : 0);
    }
    
    if (distanceSquared < deadZoneSquared) {
        // DEBUG_LOG: 进入死区（目标在死区内）- 这可能是你"动一次就不移动"的根因
        static int s_deadzoneLog = 0;
        if (s_deadzoneLog++ % 30 == 0) {
            obs_log(LOG_INFO, "[%s] DEADZONE_ENTER: dist=%.3f dz=%.3f err=(%.3f,%.3f) isMoving=%d → resetPid/Motion + RETURN",
                    getLogPrefix(),
                    std::sqrt(distanceSquared), config.deadZonePixels,
                    errorX, errorY, isMoving ? 1 : 0);
        }
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        if (autoTriggerHolding) {
            auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
        }
        // 死区内仍执行压枪补偿(后座独立于瞄准误差;目标居中时同样需要压枪)
        if (config.autoRecoilControlEnabled && checkFiring()) {
            float recoilPerSecond = config.recoilStrength /
                                    (static_cast<float>(config.recoilSpeed) / 1000.0f);
            float dy = recoilPerSecond * deltaTime;
            moveMouse(0, static_cast<int>(dy));
        }
        // 死区内也执行扳机(目标对准准心时开枪,不应被死区跳过)
        if (config.autoTriggerEnabled) {
            const float dist = std::sqrt(distanceSquared);
            if (autoTriggerHolding) {
                auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
                if (fireElapsed >= currentFireDuration) {
                    releaseAutoTrigger();
                    lastAutoTriggerTime = now;
                }
                if (dist > config.autoTriggerRadius * 2.0f) {
                    releaseAutoTrigger();
                    lastAutoTriggerTime = now;
                }
            } else if (dist < config.autoTriggerRadius) {
                auto cooldownElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastAutoTriggerTime).count();
                if (cooldownElapsed >= config.autoTriggerInterval) {
                    if (!autoTriggerWaitingForDelay) {
                        autoTriggerWaitingForDelay = true;
                        autoTriggerDelayStartTime = now;
                    }
                    auto delayElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerDelayStartTime).count();
                    int totalDelay = config.autoTriggerFireDelay + getRandomDelay();
                    if (delayElapsed >= totalDelay) {
                        performAutoClick();
                    }
                }
            } else {
                autoTriggerWaitingForDelay = false;
            }
        }
        return;
    }

    float distance = std::sqrt(distanceSquared);

    if (config.autoTriggerEnabled) {
        if (autoTriggerHolding) {
            auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
            // 目标丢失时释放按键（目标离开触发半径的2倍范围）
            if (distance > config.autoTriggerRadius * 2.0f) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
        } else {
            if (distance < config.autoTriggerRadius) {
                // 先检查冷却时间，避免无效等待
                auto cooldownElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastAutoTriggerTime).count();
                if (cooldownElapsed >= config.autoTriggerInterval) {
                    if (!autoTriggerWaitingForDelay) {
                        autoTriggerWaitingForDelay = true;
                        autoTriggerDelayStartTime = now;
                    }
                    
                    auto delayElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerDelayStartTime).count();
                    int totalDelay = config.autoTriggerFireDelay + getRandomDelay();
                    
                    if (delayElapsed >= totalDelay) {
                        performAutoClick();
                    }
                }
            } else {
                autoTriggerWaitingForDelay = false;
            }
        }
    }

    isMoving = true;

    float moveX = 0.0f, moveY = 0.0f;

    // DEBUG_LOG: 算法分发起点
    static int s_algoDispatchLog = 0;
    if (s_algoDispatchLog++ % 20 == 0) {
        const char* algoName = "UNKNOWN";
        switch (config.algorithmType) {
            case AlgorithmType::AdvancedPID:  algoName = "AdvancedPID"; break;
            case AlgorithmType::ExternalPID:  algoName = "ExternalPID(mpid)"; break;
            case AlgorithmType::AimController:algoName = "AimController(ChrisPID)"; break;
            case AlgorithmType::SlewRate:     algoName = "SlewRate"; break;
            case AlgorithmType::AdaptivePID:  algoName = "AdaptivePID"; break;
        }
        obs_log(LOG_INFO, "[%s] ALGO_DISPATCH: algo=%d(%s) err=(%.2f,%.2f) dist=%.2f fov=%d maxMove=%.2f deadZone=%.2f dt=%.4fs lastApplied=%d",
                getLogPrefix(),
                (int)config.algorithmType, algoName,
                errorX, errorY, distance, config.fovRadiusPixels,
                config.maxPixelMove, config.deadZonePixels, deltaTime,
                (int)lastAppliedAlgorithm_);
    }

    switch (config.algorithmType) {
        case AlgorithmType::AdvancedPID: {
            // 高级PID（动态P增益+自适应D+卡尔曼滤波输出级联
            if (lastAppliedAlgorithm_ != AlgorithmType::AdvancedPID) {
                obs_log(LOG_INFO, "[%s] ALGO_SWITCH: → AdvancedPID resetPidState", getLogPrefix());
                resetPidState();
                lastAppliedAlgorithm_ = AlgorithmType::AdvancedPID;
            }

            float pX = calculateDynamicP(distance) * getCurrentPGain();
            float pY = pX;
            float adaptiveFactorX = 0.0f, adaptiveFactorY = 0.0f;

            float deltaErrorX = errorX - pidPreviousErrorX;
            float deltaErrorY = errorY - pidPreviousErrorY;
            float dX = calculateAdaptiveD(distance, deltaErrorX, errorX, adaptiveFactorX);
            float dY = calculateAdaptiveD(distance, deltaErrorY, errorY, adaptiveFactorY);
            filteredDeltaErrorX = 0.7f * filteredDeltaErrorX + 0.3f * deltaErrorX;
            filteredDeltaErrorY = 0.7f * filteredDeltaErrorY + 0.3f * deltaErrorY;

            // 积分项带自适应死区
            const float iDeadZone = 1.0f;
            if (std::abs(errorX) > iDeadZone) {
                integralGainX = std::clamp(integralGainX + 0.01f, 0.0f, 1.0f);
            } else {
                integralGainX = std::clamp(integralGainX - 0.02f, 0.0f, 1.0f);
            }
            if (std::abs(errorY) > iDeadZone) {
                integralGainY = std::clamp(integralGainY + 0.01f, 0.0f, 1.0f);
            } else {
                integralGainY = std::clamp(integralGainY - 0.02f, 0.0f, 1.0f);
            }
            integralX = std::clamp(integralX + errorX * deltaTime, -config.integralLimit, config.integralLimit);
            integralY = std::clamp(integralY + errorY * deltaTime, -config.integralLimit, config.integralLimit);
            float iX = config.pidI * integralGainX * integralX;
            float iY = config.pidI * integralGainY * integralY;

            float rawOutX = pX * errorX + iX + dX;
            float rawOutY = pY * errorY + iY + dY;

            // 两级卡尔曼输出滤波
            float filteredX = kalmanOutputX.update(rawOutX);
            float filteredY = kalmanOutputY.update(rawOutY);

            moveX = filteredX;
            moveY = filteredY;

            // DEBUG_LOG: AdvancedPID 内部输出值（每20帧）
            static int s_advLog = 0;
            if (s_advLog++ % 20 == 0) {
                obs_log(LOG_INFO, "[%s] AdvancedPID: err=(%.3f,%.3f) P=(%.3f,%.3f) I=(%.3f,%.3f)(gain=%.3f/%.3f) D=(%.3f,%.3f) raw=(%.3f,%.3f) kf=(%.3f,%.3f)",
                        getLogPrefix(),
                        errorX, errorY,
                        pX * errorX, pY * errorY,
                        iX, iY, integralGainX, integralGainY,
                        dX, dY, rawOutX, rawOutY, filteredX, filteredY);
            }

            pidPreviousErrorX = errorX;
            pidPreviousErrorY = errorY;
            lastOutputX = moveX;
            lastOutputY = moveY;

            if (pidDataCallback_) {
                PidDebugData data;
                data.errorX = errorX;
                data.errorY = errorY;
                data.outputX = moveX;
                data.outputY = moveY;
                data.targetX = targetPixelX;
                data.targetY = targetPixelY;
                data.targetVelocityX = targetVelocityX;
                data.targetVelocityY = targetVelocityY;
                data.currentKp = pX;
                data.currentKi = config.pidI * integralGainX;
                data.currentKd = dX / (std::abs(errorX) > 0.001f ? errorX : 1.0f);
                data.algorithmType = 0;
                data.isFiring = isFiring;
                pidDataCallback_(data);
            }
            break;
        }

        case AlgorithmType::ExternalPID: {
            // 专业PID（mpid逆向重构版：4路滤波+atan2软限幅+变积分模式）
            if (lastAppliedAlgorithm_ != AlgorithmType::ExternalPID) {
                externalPidInitialized_ = false;
                lastAppliedAlgorithm_ = AlgorithmType::ExternalPID;
            }
            if (!externalPidInitialized_) {
                externalPidX.configure(config.externalKpX, config.externalKiX, config.externalKdX);
                externalPidY.configure(config.externalKpY, config.externalKiY, config.externalKdY);
                externalPidX.update_params(config.externalKpX, config.externalKiX, config.externalKdX,
                                           config.externalPredictX, config.externalRateX);
                externalPidY.update_params(config.externalKpY, config.externalKiY, config.externalKdY,
                                           config.externalPredictY, config.externalRateY);
                externalPidX.set_base(static_cast<int>(config.externalKiMode),
                                       config.externalKpLimit, 1000.0,
                                       config.externalKdLimit, config.externalOutputLimit,
                                       1.0, config.externalKiDeadband);
                externalPidY.set_base(static_cast<int>(config.externalKiMode),
                                       config.externalKpLimit, 1000.0,
                                       config.externalKdLimit, config.externalOutputLimit,
                                       1.0, config.externalKiDeadband);
                // KiRate 控制卡尔曼系数转换到 KF3.q：ki_rate->kf3_q
                float kf3Q = std::max(0.001f, config.externalKiRate);
                // 限制set_base已设kf3_q，因此直接手动覆盖
                externalPidInitialized_ = true;
                externalPidX.reset();
                externalPidY.reset();
            }

            double ox = externalPidX.update(errorX);
            double oy = externalPidY.update(errorY);
            // 对误差符号反推 ki_rate 控制 kf3_q 影响外部库无法直接调用私有成员，但已在set_base中配置
            moveX = static_cast<float>(ox);
            moveY = static_cast<float>(oy);

            static int s_extLog = 0;
            if (s_extLog++ % 20 == 0) {
                obs_log(LOG_INFO, "[%s] ExternalPID(mpid): err=(%.3f,%.3f) out=(%.4f,%.4f) init=%d mode=%d KpLim=%.2f KdLim=%.2f OutLim=%.2f",
                        getLogPrefix(),
                        errorX, errorY, moveX, moveY,
                        externalPidInitialized_ ? 1 : 0,
                        (int)config.externalKiMode,
                        config.externalKpLimit, config.externalKdLimit, config.externalOutputLimit);
            }

            lastOutputX = moveX;
            lastOutputY = moveY;

            if (pidDataCallback_) {
                PidDebugData data;
                data.errorX = errorX;
                data.errorY = errorY;
                data.outputX = moveX;
                data.outputY = moveY;
                data.targetX = targetPixelX;
                data.targetY = targetPixelY;
                data.targetVelocityX = targetVelocityX;
                data.targetVelocityY = targetVelocityY;
                data.currentKp = config.externalKpX;
                data.currentKi = config.externalKiX;
                data.currentKd = config.externalKdX;
                data.algorithmType = 1;
                data.isFiring = isFiring;
                pidDataCallback_(data);
            }
            break;
        }

        case AlgorithmType::AimController: {
            // aim 控制器（增量式PID+运动预测+柏林噪声+渐入+输出限幅）
            if (lastAppliedAlgorithm_ != AlgorithmType::AimController) {
                aimController_.reset();
                lastAppliedAlgorithm_ = AlgorithmType::AimController;
            }
            double noiseAmp = config.aimNoiseEnabled ? config.aimNoiseAmplitude : 0.0;
            aim::AimOutput out = aimController_.update(
                errorX, errorY,
                config.aimPredictionWeightX,
                config.aimPredictionWeightY,
                config.aimInitScale,
                config.aimRampTime,
                config.aimOutputMax,
                noiseAmp);
            moveX = static_cast<float>(out.move_x);
            moveY = static_cast<float>(out.move_y);
            static int s_aimLog = 0;
            if (s_aimLog++ % 20 == 0) {
                obs_log(LOG_INFO, "[%s] AimController(ChrisPID): err=(%.3f,%.3f) out=(%.4f,%.4f) ramp=%.3fs maxOut=%.2f noise=%.2f predW=(%.2f,%.2f)",
                        getLogPrefix(),
                        errorX, errorY, moveX, moveY,
                        config.aimRampTime, config.aimOutputMax, noiseAmp,
                        config.aimPredictionWeightX, config.aimPredictionWeightY);
            }
            lastOutputX = moveX;
            lastOutputY = moveY;
            if (pidDataCallback_) {
                PidDebugData data;
                data.errorX = errorX;
                data.errorY = errorY;
                data.outputX = moveX;
                data.outputY = moveY;
                data.targetX = targetPixelX;
                data.targetY = targetPixelY;
                data.targetVelocityX = targetVelocityX;
                data.targetVelocityY = targetVelocityY;
                data.currentKp = config.aimKp;
                data.currentKi = config.aimKi;
                data.currentKd = config.aimKd;
                data.algorithmType = 2;
                data.isFiring = isFiring;
                pidDataCallback_(data);
            }
            break;
        }

        case AlgorithmType::SlewRate: {
            // SlewRate（限速平滑趋近+阻尼制动+归一化误差
            if (!slewRateInitialized_ || lastAppliedAlgorithm_ != AlgorithmType::SlewRate) {
                slewRuntime_ = {};
                slewRateInitialized_ = true;
                lastAppliedAlgorithm_ = AlgorithmType::SlewRate;
            }

            slewrate::SlewControllerParameters params;
            params.outputGain = config.slewRateOutputGain;
            params.responseSmoothing = config.slewRateResponseSmoothing;
            params.approachDamping = config.slewRateApproachDamping;
            params.updateIntervalMs = config.slewRateUpdateIntervalMs;
            params.normalizationScale = config.slewRateNormalizationScale;

            float closeThreshold = config.deadZonePixels * 3.0f;
            bool closeToTarget = distance < std::max(closeThreshold, 5.0f);
            float elapsedMs = (deltaTime > 0.0f) ? deltaTime * 1000.0f : 16.67f;
            slewrate::SlewAimOutput out = slewrate::updateControllerCore(
                slewRuntime_, errorX, errorY, elapsedMs, params, true, closeToTarget);

            moveX = out.dx;
            moveY = out.dy;
            static int s_slewLog = 0;
            if (s_slewLog++ % 20 == 0) {
                obs_log(LOG_INFO, "[%s] SlewRate: err=(%.3f,%.3f) out=(%.4f,%.4f) gain=%.4f smooth=%.5f damp=%.2f close=%d elapsed=%.1fms",
                        getLogPrefix(),
                        errorX, errorY, moveX, moveY,
                        config.slewRateOutputGain, config.slewRateResponseSmoothing,
                        config.slewRateApproachDamping, closeToTarget ? 1 : 0, elapsedMs);
            }
            lastOutputX = moveX;
            lastOutputY = moveY;
            if (pidDataCallback_) {
                PidDebugData data;
                data.errorX = errorX;
                data.errorY = errorY;
                data.outputX = moveX;
                data.outputY = moveY;
                data.targetX = targetPixelX;
                data.targetY = targetPixelY;
                data.targetVelocityX = targetVelocityX;
                data.targetVelocityY = targetVelocityY;
                data.currentKp = config.slewRateOutputGain;
                data.currentKi = 0.0f;
                data.currentKd = 0.0f;
                data.algorithmType = 3;
                data.isFiring = isFiring;
                pidDataCallback_(data);
            }
            break;
        }

        case AlgorithmType::AdaptivePID:
        default: {
            // 自适应PID控制器（位置式PID+自适应积分增益+积分死区+双重抗饱和）
            if (lastAppliedAlgorithm_ != AlgorithmType::AdaptivePID) {
                adaptivePidX_.reset();
                adaptivePidY_.reset();
                lastAppliedAlgorithm_ = AlgorithmType::AdaptivePID;
            }

            float adaptiveErrorX = errorX;
            float adaptiveErrorY = errorY;

            // 诊断：预测/Smith 各级贡献量（定位"预测开着但没输出"问题）
            float smithDx = 0.0f, smithDy = 0.0f;   // Smith 补偿量
            float predAddX = 0.0f, predAddY = 0.0f; // 实际加入的预测增量
            float predVelX = 0.0f, predVelY = 0.0f; // 滤波器速度估计
            float predGated = 0.0f;                 // 1=门控关预测 0=预测生效

            bool smithOn = false;
            if (config.smithPredictorEnabled) {
                auto [smithCX, smithCY] = smithPredictor.correct(
                    lastOutputX, lastOutputY, adaptiveErrorX, adaptiveErrorY, deltaTime);
                smithDx = smithCX - adaptiveErrorX;
                smithDy = smithCY - adaptiveErrorY;
                adaptiveErrorX = smithCX;
                adaptiveErrorY = smithCY;
                smithOn = true;
            }

            float predWX = config.predictionWeightX;
            float predWY = config.predictionWeightY;
            if (smithOn && (config.immFilterEnabled || config.useVbFilter)) {
                predWX *= 0.5f;
                predWY *= 0.5f;
            }
            // 目标丢失恢复期预测衰减：冻结期速度估计不可信（旧方向），渐进恢复
            if (freezeRecoverFrames_ > 0) {
                predWX *= 0.25f;
                predWY *= 0.25f;
                freezeRecoverFrames_--;
            }

            // 预测外推时间尺度：一帧控制周期。
            // 注：曾用"检测年龄+执行延迟"外推(仿真验证)，速度估计噪声×h放大，
            // 直线段 RMS 恶化(IMM 6.7→12.1)，且测量延迟被PID闭环天然吸收 → 回滚。
            // 执行延迟(输出端)由 Smith 预估器补偿(delayTau含mouseLatencyMs)。
            const float predHorizonSec = deltaTime;
            const float maxPredPx = std::max(0.0f, config.maxPredictionPixels);

            if (config.immFilterEnabled) {
                immFilter.predict(deltaTime, previousMoveX, previousMoveY);
                immFilter.update(errorX, errorY);
                float immDeltaX = 0.0f, immDeltaY = 0.0f;
                immFilter.getPrediction(predHorizonSec, immDeltaX, immDeltaY);
                float estX = 0.0f, estY = 0.0f;
                immFilter.getState(estX, estY, predVelX, predVelY);
                if (maxPredPx > 0.0f) {
                    immDeltaX = std::clamp(immDeltaX, -maxPredPx, maxPredPx);
                    immDeltaY = std::clamp(immDeltaY, -maxPredPx, maxPredPx);
                }
                // 机动门控：真机动（位移+σ双阈值，排除检测框噪声跳）时速度估计不可信，关提前量
                // 目标切换帧：预测状态属于旧目标，强制关
                bool trackSwitched = (lockedTrackId != lastPredictionTrackId_);
                lastPredictionTrackId_ = lockedTrackId;
                if (!immFilter.maneuverDetected() && !trackSwitched) {
                    predAddX = predWX * immDeltaX;
                    predAddY = predWY * immDeltaY;
                    adaptiveErrorX += predAddX;
                    adaptiveErrorY += predAddY;
                } else {
                    predGated = 1.0f;
                }
            }
            else if (config.useVbFilter) {
                // 变分贝叶斯鲁棒滤波：R在线估计+野值抑制，语义同 IMM（只补提前量）
                vbFilter.predict(deltaTime, previousMoveX, previousMoveY);
                vbFilter.update(errorX, errorY);
                float vbDeltaX = 0.0f, vbDeltaY = 0.0f;
                vbFilter.getPrediction(predHorizonSec, vbDeltaX, vbDeltaY);
                float estX = 0.0f, estY = 0.0f;
                vbFilter.getState(estX, estY, predVelX, predVelY);
                if (maxPredPx > 0.0f) {
                    vbDeltaX = std::clamp(vbDeltaX, -maxPredPx, maxPredPx);
                    vbDeltaY = std::clamp(vbDeltaY, -maxPredPx, maxPredPx);
                }
                // 机动门控同上
                bool trackSwitched = (lockedTrackId != lastPredictionTrackId_);
                lastPredictionTrackId_ = lockedTrackId;
                if (!vbFilter.maneuverDetected() && !trackSwitched) {
                    predAddX = predWX * vbDeltaX;
                    predAddY = predWY * vbDeltaY;
                    adaptiveErrorX += predAddX;
                    adaptiveErrorY += predAddY;
                } else {
                    predGated = 1.0f;
                }
            }
            else if (config.useDerivativePredictor) {
                predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
                float derivPredictedX = 0.0f, derivPredictedY = 0.0f;
                predictor.predict(predHorizonSec, derivPredictedX, derivPredictedY);
                if (maxPredPx > 0.0f) {
                    derivPredictedX = std::clamp(derivPredictedX, -maxPredPx, maxPredPx);
                    derivPredictedY = std::clamp(derivPredictedY, -maxPredPx, maxPredPx);
                }
                predAddX = predWX * derivPredictedX;
                predAddY = predWY * derivPredictedY;
                adaptiveErrorX += predAddX;
                adaptiveErrorY += predAddY;
            }

            moveX = adaptivePidX_.update(adaptiveErrorX, deltaTime);
            moveY = adaptivePidY_.update(adaptiveErrorY, deltaTime);

            // 速度前馈：目标速度估计直接进输出 u += Kv·v·dt（独立通道）。
            // 当前"预测进误差"等效前馈 = Kp×predW×dt ≈ 0.0017 几乎为零，
            // P 独自扛高速跟踪 → 静差 = v/Kp 数学必然追不上。
            // 机动/换目标时速度不可信，前馈随预测一起关
            if (predGated == 0.0f && config.velocityFeedforward > 0.0f) {
                moveX += config.velocityFeedforward * predVelX * deltaTime;
                moveY += config.velocityFeedforward * predVelY * deltaTime;
            }

            static int s_adaptLog = 0;
            if (s_adaptLog++ % 20 == 0) {
                obs_log(LOG_INFO, "[%s] AdaptivePID: rawErr=(%.3f,%.3f) adapErr=(%.3f,%.3f) out=(%.4f,%.4f) pid=(P:%.2f,%.2f I:%.2f,%.2f D:%.2f,%.2f) ff=(%.2f,%.2f) smith=%d smithΔ=(%.2f,%.2f) predΔ=(%.2f,%.2f) vel=(%.1f,%.1f) gate=%d imm=%d vb=%d derivPred=%d dt=%.4f K=(%.2f,%.2f,%.2f)",
                        getLogPrefix(),
                        errorX, errorY, adaptiveErrorX, adaptiveErrorY,
                        moveX, moveY,
                        adaptivePidX_.lastPOut_, adaptivePidY_.lastPOut_,
                        adaptivePidX_.lastIOut_, adaptivePidY_.lastIOut_,
                        adaptivePidX_.lastDOut_, adaptivePidY_.lastDOut_,
                        config.velocityFeedforward * predVelX * deltaTime,
                        config.velocityFeedforward * predVelY * deltaTime,
                        smithOn ? 1 : 0,
                        smithDx, smithDy,
                        predAddX, predAddY,
                        predVelX, predVelY,
                        (int)predGated,
                        config.immFilterEnabled ? 1 : 0,
                        config.useVbFilter ? 1 : 0,
                        config.useDerivativePredictor ? 1 : 0,
                        deltaTime,
                        config.adaptivePidKp, config.adaptivePidKi, config.adaptivePidKd);
            }

            if (pidDataCallback_) {
                PidDebugData data;
                data.errorX = errorX;
                data.errorY = errorY;
                data.outputX = moveX;
                data.outputY = moveY;
                data.targetX = targetPixelX;
                data.targetY = targetPixelY;
                data.targetVelocityX = targetVelocityX;
                data.targetVelocityY = targetVelocityY;
                data.currentKp = config.adaptivePidKp;
                data.currentKi = config.adaptivePidKi;
                data.currentKd = config.adaptivePidKd;
                data.algorithmType = 8;
                data.isFiring = isFiring;
                pidDataCallback_(data);
            }

            previousErrorX = errorX;
            previousErrorY = errorY;
            lastOutputX = moveX;
            lastOutputY = moveY;
            break;
        }
    }

    bool firing = checkFiring();
    
    if (firing && config.autoRecoilControlEnabled) {
        moveY *= config.recoilPidGainScale;
    }
    
    float moveDistSquared = moveX * moveX + moveY * moveY;
    float maxMoveSquared = config.maxPixelMove * config.maxPixelMove;
    if (moveDistSquared > maxMoveSquared && moveDistSquared > 0.0f) {
        float scale = config.maxPixelMove / std::sqrt(moveDistSquared);
        moveX *= scale;
        moveY *= scale;
    }

    // 检测结果新鲜度衰减: 自上次 setDetections 超过一个检测周期后, 移动输出指数衰减。
    // 双机/UDP 场景检测帧率低 (30fps 或更低), 两帧检测之间 error 不变, PID 持续推同一方向
    // 造成来回过冲振荡; 衰减让陈旧位置信息不再持续驱动。单机高检测帧率下每 tick 有新结果, 不触发。
    {
        auto nowD = std::chrono::steady_clock::now();
        double sinceDetMs = std::chrono::duration<double, std::milli>(nowD - lastDetectionsUpdate_).count();
        if (sinceDetMs > 12.0) {
            double decay = std::pow(0.5, sinceDetMs / 12.0);
            moveX = static_cast<float>(moveX * decay);
            moveY = static_cast<float>(moveY * decay);
        }
        // 诊断采样: 每 120 tick 打一次 (双机移动抖动排查)
        if (logCounter_++ % 120 == 0) {
            obs_log(LOG_INFO, "[%s] tick诊断 err=(%.1f,%.1f) move=(%.1f,%.1f) sinceDet=%.1fms det=%zu locked=%d",
                    getLogPrefix(), errorX, errorY, moveX, moveY, sinceDetMs,
                    currentDetections.size(), lockedTrackId);
        }
    }
    
    if (yUnlockActive) {
        moveY = 0.0f;
    }
    
    if (config.enableBezierMovement) {
        float moveDistance = std::sqrt(moveX * moveX + moveY * moveY);
        if (moveDistance > 1.0f) {
            float perpX = -moveY / moveDistance;
            float perpY = moveX / moveDistance;
            
            bezierPhase += deltaTime * 3.0f;
            
            float curvatureOffset = std::sin(bezierPhase) * config.bezierCurvature * moveDistance * 0.3f;
            
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            float randomFactor = dist(randomGenerator) * config.bezierRandomness;
            curvatureOffset *= (1.0f + randomFactor);
            
            moveX += perpX * curvatureOffset;
            moveY += perpY * curvatureOffset;
        }
    }
    
    // GhostTracker曲线轨迹
    // 注意：GhostTracker的偏移是修改目标位置，不是叠加到移动量
    // 这里我们用它来生成垂直于移动方向的曲线偏移
    if (config.enableGhostTracker) {
        float moveDist = std::sqrt(moveX * moveX + moveY * moveY);
        if (moveDist > 1.0f) {
            // 计算移动方向的垂直向量
            float dirX = moveX / moveDist;
            float dirY = moveY / moveDist;
            float perpX = -dirY;  // 垂直于移动方向
            float perpY = dirX;

            // 复用上方已选中的target，不再重复调用selectTarget()
            // （原代码重复调用会导致目标锁定状态不一致）
            float ghostOffsetX = 0.0f, ghostOffsetY = 0.0f;
            if (target) {
                int fw = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth :
                         ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
                int fh = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight :
                         ((config.sourceHeight > 0) ? config.sourceHeight : 1080);
                float targetW = target->width * fw;
                float targetH = target->height * fh;

                // GhostTracker配置已在updateConfig()中设置，不再每帧重建
                // 传入误差（相对于准心的偏移）
                if (ghostTracker.apply(errorX, errorY,
                                       targetW, targetH,
                                       static_cast<float>(fw), static_cast<float>(fh),
                                       ghostOffsetX, ghostOffsetY)) {
                    // 将GhostTracker的偏移投影到移动方向的垂直方向
                    // ghostOffset是目标位置的偏移，我们需要转换为移动量的偏移
                    float curveOffset = (ghostOffsetX * perpX + ghostOffsetY * perpY);
                    // 限制曲线偏移量，不超过移动量的50%
                    curveOffset = std::clamp(curveOffset, -moveDist * 0.5f, moveDist * 0.5f);
                    
                    moveX += perpX * curveOffset;
                    moveY += perpY * curveOffset;
                }
            }
        }
    }

    // 时间相关移动：只缩放 PID/控制输出（位移/参考帧）
    // 注意：AdaptivePID 内部已用 dt 做 I/D，此处 timeFactor 仍按「60Hz 参考位移」兼容，
    // 不要在 PID 内再乘 dt 输出。
    float finalMoveX = moveX;
    float finalMoveY = moveY;

    if (config.enableTimeBasedMovement && deltaTime > 0.0f) {
        // 时间缩放 = dt×目标帧率。限幅 [0.1, 2.0]：首帧/卡顿帧 dt 被 clamp 到 50ms 时
        // 无上限会 ×3 突跳输出（日志实证 final=3×movePre），限幅后单帧最多补 2 参考帧，
        // 余量由后续帧自然补（PID 误差仍在）
        float timeFactor = std::clamp(deltaTime * config.targetFrameRate, 0.1f, 2.0f);
        finalMoveX *= timeFactor;
        finalMoveY *= timeFactor;
    }

    // 压枪补偿：连续速率 (strength/speed 视为像素/秒)，只乘一次 dt
    // 必须在 timeFactor 之后加入，避免低 FPS 时 recoil 被双重放大
    if (config.autoRecoilControlEnabled && firing) {
        // recoilSpeed is ms per full strength unit → rate = strength / (speed/1000) px/s
        float recoilPerSecond = config.recoilStrength / (static_cast<float>(config.recoilSpeed) / 1000.0f);
        finalMoveY += recoilPerSecond * deltaTime;
    }

    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;

    // 子像素累积：把 float 余数累计起来，凑够 1 mickey 再发送。
    // 游戏 Raw Input 下 SendInput 是整数 mickey；原 static_cast<int> 会把 0.7px × 10 帧
    // 全部截断成 0 → 视觉"不动"。用 std::floor(std::abs + sign) 保证方向正确。
    // 最后防线：非有限值(NaN/Inf)直接置 0——任何上游除零/发散不再飞鼠标(INT_MIN)
    if (!std::isfinite(finalMoveX)) finalMoveX = 0.0f;
    if (!std::isfinite(finalMoveY)) finalMoveY = 0.0f;
    float accumX = subpixelAccumX_ + finalMoveX;
    float accumY = subpixelAccumY_ + finalMoveY;
    int sendDx = 0, sendDy = 0;
    if (accumX >= 0.0f) {
        sendDx = static_cast<int>(std::floor(accumX));
    } else {
        sendDx = -static_cast<int>(std::floor(-accumX));
    }
    if (accumY >= 0.0f) {
        sendDy = static_cast<int>(std::floor(accumY));
    } else {
        sendDy = -static_cast<int>(std::floor(-accumY));
    }
    subpixelAccumX_ = accumX - static_cast<float>(sendDx);
    subpixelAccumY_ = accumY - static_cast<float>(sendDy);
    // 钳制余数避免无限漂移（正常范围 (-1, 1)，这里留一点冗余）
    subpixelAccumX_ = std::clamp(subpixelAccumX_, -2.0f, 2.0f);
    subpixelAccumY_ = std::clamp(subpixelAccumY_, -2.0f, 2.0f);

    // DEBUG_LOG: 最终输出+发送 - 每10帧打印一次（排查"动一次就不动"必看）
    static int s_sendLog = 0;
    if (s_sendLog++ % 10 == 0 || sendDx != 0 || sendDy != 0) {
        obs_log(LOG_INFO, "[%s] SEND: algo=%d movePre=(%.4f,%.4f) final=(%.4f,%.4f) "
                         "accum=(%.4f,%.4f) accRem=(%.4f,%.4f) → SendInput(%+d,%+d) "
                         "yUnlock=%d timeBased=%d targetFPS=%.0f firing=%d recoil=%d",
                getLogPrefix(),
                (int)config.algorithmType,
                moveX, moveY, finalMoveX, finalMoveY,
                accumX, accumY, subpixelAccumX_, subpixelAccumY_,
                sendDx, sendDy,
                yUnlockActive ? 1 : 0,
                config.enableTimeBasedMovement ? 1 : 0,
                config.targetFrameRate,
                firing ? 1 : 0,
                (config.autoRecoilControlEnabled && firing) ? 1 : 0);
    } else {
        // 每30帧即使 send=0 也打印一次心跳（确认 tick 没停）
        static int s_heartbeatLog = 0;
        if (s_heartbeatLog++ % 30 == 0) {
            obs_log(LOG_INFO, "[%s] SEND_HEARTBEAT: algo=%d final=(%.4f,%.4f) accRem=(%.4f,%.4f) → SendInput(0,0) [no integer reached yet]",
                    getLogPrefix(), (int)config.algorithmType,
                    finalMoveX, finalMoveY, subpixelAccumX_, subpixelAccumY_);
        }
    }

    moveMouse(sendDx, sendDy);
}

Detection* AbstractMouseController::selectTarget()
{
    if (currentDetections.empty()) {
        // 空检测：宽限期内保留锁，避免一帧漏检就乱转火
        if (lockedTrackId >= 0 && lockMissCount_ < kMaxLockMissFrames) {
            lockMissCount_++;
            return nullptr;
        }
        lockedTrackId = -1;
        lockMissCount_ = 0;
        pendingTargetTrackId = -1;
        currentTargetScore = 0.0f;
        pendingTargetScore = 0.0f;
        return nullptr;
    }

    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth :
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight :
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    int fovCenterX = frameWidth / 2;
    int fovCenterY = frameHeight / 2;
    float fovRadius = static_cast<float>(config.fovRadiusPixels);

    // 准星位置优先：如果有准星检测结果，用准星位置作为目标选择中心
    if (aimOriginX_ >= 0.0f && aimOriginY_ >= 0.0f) {
        fovCenterX = static_cast<int>(aimOriginX_);
        fovCenterY = static_cast<int>(aimOriginY_);
    }

    // 计算所有在FOV内的目标分数
    Detection* bestTarget = nullptr;
    float bestScore = -1.0f;
    Detection* currentTarget = nullptr;
    float currentScore = 0.0f;

    for (auto& det : currentDetections) {
        float pixelX = det.centerX * frameWidth;
        float pixelY = det.centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        float distSq = dx * dx + dy * dy;

        if (distSq > fovRadius * fovRadius) continue;

        float distance = std::sqrt(distSq);
        float distanceScore = 1.0f / (1.0f + distance * 0.01f);
        float confidenceScore = det.confidence;
        float score = 0.6f * distanceScore + 0.4f * confidenceScore;
        // 锁粘性：当前锁 ID 加分，减少边界抖动切目标（Bar-Shalom 跟踪连续性思想）
        if (det.trackId == lockedTrackId && lockedTrackId >= 0) {
            score += 0.15f;
        }

        if (score > bestScore) {
            bestScore = score;
            bestTarget = &det;
        }

        if (det.trackId == lockedTrackId) {
            currentTarget = &det;
            currentScore = score;
        }
    }

    // 没有任何目标
    if (!bestTarget) {
        if (lockedTrackId >= 0 && lockMissCount_ < kMaxLockMissFrames) {
            lockMissCount_++;
            return nullptr;
        }
        lockedTrackId = -1;
        lockMissCount_ = 0;
        pendingTargetTrackId = -1;
        currentTargetScore = 0.0f;
        pendingTargetScore = 0.0f;
        currentTargetDistance = 0.0f;
        return nullptr;
    }

    // 锁 ID 本帧不在列表：宽限期内不立刻转 best，防 ID 闪断乱锁邻居
    if (lockedTrackId >= 0 && !currentTarget) {
        lockMissCount_++;
        if (lockMissCount_ < kMaxLockMissFrames) {
            return nullptr;
        }
        // 宽限耗尽，允许落锁到 best
        lockedTrackId = bestTarget->trackId;
        lockMissCount_ = 0;
        pendingTargetTrackId = -1;
        currentTargetScore = bestScore;
        pendingTargetScore = 0.0f;
        float pixelX = bestTarget->centerX * frameWidth;
        float pixelY = bestTarget->centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        currentTargetDistance = std::sqrt(dx * dx + dy * dy);
        return bestTarget;
    }

    // 如果当前没有锁定目标，直接选择最佳目标
    if (lockedTrackId < 0) {
        lockedTrackId = bestTarget->trackId;
        lockMissCount_ = 0;
        pendingTargetTrackId = -1;
        currentTargetScore = bestScore;
        pendingTargetScore = 0.0f;
        // 新目标：重启 P-gain ramp，并清 Smith/IMM 等旧状态
        targetLockStartTime = std::chrono::steady_clock::now();
        smithPredictor.reset();
        immFilter.reset();
        vbFilter.reset();
        oneEuroX_.reset();
        oneEuroY_.reset();
        oneEuroLockedTrackId_ = lockedTrackId;
        adaptivePidX_.reset();
        adaptivePidY_.reset();
        float pixelX = bestTarget->centerX * frameWidth;
        float pixelY = bestTarget->centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        currentTargetDistance = std::sqrt(dx * dx + dy * dy);
        return bestTarget;
    }

    lockMissCount_ = 0;

    // 当前有锁定目标，检查是否需要切换
    // 新目标必须比当前目标好一定容差才考虑切换
    float scoreImprovement = bestScore - currentScore;
    float toleranceThreshold = config.targetSwitchTolerance;

    if (scoreImprovement <= toleranceThreshold) {
        // 新目标不够好，继续锁定当前目标，清除待切换状态
        pendingTargetTrackId = -1;
        pendingTargetScore = 0.0f;
        float pixelX = currentTarget->centerX * frameWidth;
        float pixelY = currentTarget->centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        currentTargetDistance = std::sqrt(dx * dx + dy * dy);
        return currentTarget;
    }

    // 新目标比当前目标好超过容差，检查是否需要延迟
    int switchDelayMs = config.targetSwitchDelayMs;

    if (switchDelayMs <= 0) {
        // 无延迟，直接切换
        lockedTrackId = bestTarget->trackId;
        pendingTargetTrackId = -1;
        currentTargetScore = bestScore;
        pendingTargetScore = 0.0f;
        float pixelX = bestTarget->centerX * frameWidth;
        float pixelY = bestTarget->centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        currentTargetDistance = std::sqrt(dx * dx + dy * dy);
        return bestTarget;
    }

    // 有延迟，检查是否已经在等待切换
    if (pendingTargetTrackId == bestTarget->trackId) {
        // 已经在等待这个目标，检查延迟时间
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - pendingTargetStartTime).count();

        if (elapsed >= switchDelayMs) {
            // 延迟时间已到，切换目标
            lockedTrackId = bestTarget->trackId;
            pendingTargetTrackId = -1;
            currentTargetScore = bestScore;
            pendingTargetScore = 0.0f;
            float pixelX = bestTarget->centerX * frameWidth;
            float pixelY = bestTarget->centerY * frameHeight;
            float dx = pixelX - fovCenterX;
            float dy = pixelY - fovCenterY;
            currentTargetDistance = std::sqrt(dx * dx + dy * dy);
            return bestTarget;
        } else {
            // 延迟时间未到，继续锁定当前目标
            float pixelX = currentTarget->centerX * frameWidth;
            float pixelY = currentTarget->centerY * frameHeight;
            float dx = pixelX - fovCenterX;
            float dy = pixelY - fovCenterY;
            currentTargetDistance = std::sqrt(dx * dx + dy * dy);
            return currentTarget;
        }
    } else {
        // 新的候选目标，开始计时
        pendingTargetTrackId = bestTarget->trackId;
        pendingTargetStartTime = std::chrono::steady_clock::now();
        pendingTargetScore = bestScore;
        // 继续锁定当前目标
        float pixelX = currentTarget->centerX * frameWidth;
        float pixelY = currentTarget->centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        currentTargetDistance = std::sqrt(dx * dx + dy * dy);
        return currentTarget;
    }
}

float AbstractMouseController::calculateDynamicP(float distance)
{
    float normalizedDistance = distance / static_cast<float>(config.fovRadiusPixels);
    normalizedDistance = std::max(0.0f, std::min(1.0f, normalizedDistance));
    float distancePower = std::pow(normalizedDistance, config.pidPSlope);
    float p = config.pidPMin + (config.pidPMax - config.pidPMin) * distancePower;
    return std::max(config.pidPMin, std::min(config.pidPMax, p));
}

float AbstractMouseController::calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor)
{
    // 归一化距离（0~1）：FOV内近距离小D，远距离大D
    float normalizedDistance = distance / static_cast<float>(config.fovRadiusPixels);
    normalizedDistance = std::clamp(normalizedDistance, 0.0f, 1.0f);

    // 误差变化率阈值判断：抖动时抑制D，平稳时恢复D
    float absDelta = std::abs(deltaError);
    float absError = std::abs(error);

    // 抖动因子：deltaError 相对误差较大时认为在振荡，压低D增益
    float jitterFactor = 1.0f;
    if (absError > 0.1f && absDelta > absError * 0.5f) {
        jitterFactor = std::max(0.2f, 1.0f - (absDelta / absError - 0.5f));
    }

    // 距离因子：远距离稍大D，便于快速拉近；近距离小D避免超调
    float distanceFactor = 0.5f + normalizedDistance * 0.5f;

    // 合成自适应系数
    adaptiveFactor = jitterFactor * distanceFactor;
    adaptiveFactor = std::clamp(adaptiveFactor, 0.1f, 1.5f);

    // D项输出 = Kd * deltaError * adaptiveFactor * dTermScale
    // 注：derivativeFilterAlpha 已在调用侧通过 filteredDeltaError[X/Y] 做平滑
    return config.pidD * deltaError * adaptiveFactor * config.dTermScale;
}

float AbstractMouseController::getCurrentPGain()
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetLockStartTime).count() / 1000.0f;
    
    // 防除零：rampDuration=0 或刚锁定(elapsed≈0)时 0/0=NaN → pX NaN → 输出 NaN
    // → SendInput(INT_MIN) 鼠标飞屏角。下限 1ms 保持合理行为
    float duration = std::max(config.pGainRampDuration, 0.001f);
    float rampFactor = std::min(elapsed / duration, 1.0f);
    float currentScale = config.pGainRampInitialScale + (1.0f - config.pGainRampInitialScale) * rampFactor;
    
    return currentScale;
}

void AbstractMouseController::resetPidState()
{
    pidPreviousErrorX = 0.0f;
    pidPreviousErrorY = 0.0f;
    filteredDeltaErrorX = 0.0f;
    filteredDeltaErrorY = 0.0f;
    integralX = 0.0f;
    integralY = 0.0f;
    // integralGain 初始 0 会导致 I 项冷启动几十帧才爬升；给 0.5 初值保留"慢积分"特性但不至于无输出
    integralGainX = 0.5f;
    integralGainY = 0.5f;
    adaptivePGainX = 1.0f;
    adaptivePGainY = 1.0f;
    adaptiveIGainX = 1.0f;
    adaptiveIGainY = 1.0f;
    kf2X.reset();
    kf2Y.reset();
    kalmanOutputX.reset();
    kalmanOutputY.reset();
    kf3X_x = 0.0f;
    kf3X_P = 1.0f;
    kf3Y_x = 0.0f;
    kf3Y_P = 1.0f;
    lastOutputX = 0.0f;
    lastOutputY = 0.0f;
    predictor.reset();
    immFilter.reset();
    vbFilter.reset();
    oneEuroX_.reset();
    oneEuroY_.reset();
    oneEuroLockedTrackId_ = -1;
    smithPredictor.reset();
    adaptivePidX_.reset();
    adaptivePidY_.reset();
    // 子像素累积器清零（跨目标/跨热键周期不保留余数，避免上一次的小余数污染新瞄准）
    subpixelAccumX_ = 0.0f;
    subpixelAccumY_ = 0.0f;
}

void AbstractMouseController::resetMotionState()
{
    currentVelocityX = 0.0f;
    currentVelocityY = 0.0f;
    currentAccelerationX = 0.0f;
    currentAccelerationY = 0.0f;
    previousMoveX = 0.0f;
    previousMoveY = 0.0f;
}

int AbstractMouseController::getRandomDelay()
{
    if (!config.autoTriggerDelayRandomEnabled) {
        return 0;
    }
    if (config.autoTriggerDelayRandomMin >= config.autoTriggerDelayRandomMax) {
        return config.autoTriggerDelayRandomMin;
    }
    std::uniform_int_distribution<int> dist(config.autoTriggerDelayRandomMin, config.autoTriggerDelayRandomMax);
    return dist(randomGenerator);
}

int AbstractMouseController::getRandomDuration()
{
    if (!config.autoTriggerDurationRandomEnabled) {
        return 0;
    }
    if (config.autoTriggerDurationRandomMin >= config.autoTriggerDurationRandomMax) {
        return config.autoTriggerDurationRandomMin;
    }
    std::uniform_int_distribution<int> dist(config.autoTriggerDurationRandomMin, config.autoTriggerDurationRandomMax);
    return dist(randomGenerator);
}

void AbstractMouseController::performAutoClick()
{
    performClickDown();
    autoTriggerHolding = true;
    isFiring = true;
    autoTriggerFireStartTime = std::chrono::steady_clock::now();
    currentFireDuration = config.autoTriggerFireDuration + getRandomDuration();
}

void AbstractMouseController::releaseAutoTrigger()
{
    if (autoTriggerHolding) {
        performClickUp();
        autoTriggerHolding = false;
    }
    isFiring = false;
    autoTriggerWaitingForDelay = false;
}

void AbstractMouseController::setCurrentWeapon(const std::string& weaponName)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentWeapon = weaponName;
}

std::string AbstractMouseController::getCurrentWeapon() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return currentWeapon;
}

void AbstractMouseController::setPidDataCallback(PidDataCallback callback)
{
    pidDataCallback_ = callback;
}

void AbstractMouseController::setAimOrigin(float x, float y)
{
    aimOriginX_ = x;
    aimOriginY_ = y;
}

const char* AbstractMouseController::getLogPrefix() const
{
    return "";
}

#endif
