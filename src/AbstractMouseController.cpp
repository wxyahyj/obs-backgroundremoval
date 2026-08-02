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
}

void AbstractMouseController::setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
    config.inferenceFrameWidth = frameWidth;
    config.inferenceFrameHeight = frameHeight;
    config.cropOffsetX = cropX;
    config.cropOffsetY = cropY;
}

void AbstractMouseController::setDetectionsWithFrameSize(std::vector<Detection>&& detections, int frameWidth, int frameHeight, int cropX, int cropY)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = std::move(detections);
    config.inferenceFrameWidth = frameWidth;
    config.inferenceFrameHeight = frameHeight;
    config.cropOffsetX = cropX;
    config.cropOffsetY = cropY;
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

void AbstractMouseController::tick()
{
    std::lock_guard<std::mutex> lock(mutex);

    if (!config.enableMouseControl) {
        if (autoTriggerHolding) {
            performClickUp();
            autoTriggerHolding = false;
        }
        autoTriggerWaitingForDelay = false;
        isMoving = false;
        return;
    }

    bool hotkeyPressed = (GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000) != 0;
    bool shouldAim = config.continuousAimEnabled || hotkeyPressed;

    if (!shouldAim) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        } else {
            // 即使isMoving为false（目标丢失导致），热键松开时也要重置所有状态
            integralX = 0.0f;
            integralY = 0.0f;
            integralGainX = 0.0f;
            integralGainY = 0.0f;
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
        if (isMoving) {
            isMoving = false;
            // 目标丢失：清空预测/滤波/Smith，避免旧目标状态污染新目标
            // 积分仍由 resetPidState 在热键松开时彻底清零
            predictor.reset();
            immFilter.reset();
            oneEuroX_.reset();
            oneEuroY_.reset();
            oneEuroLockedTrackId_ = -1;
            smithPredictor.reset();
            adaptivePidX_.reset();
            adaptivePidY_.reset();
            resetMotionState();
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

    float moveX, moveY;

    if (config.algorithmType == AlgorithmType::AdaptivePID) {
        // 自适应PID控制器（位置式PID+自适应积分增益+积分死区+双重抗饱和）
        // 算法切换时重置状态
        if (lastAppliedAlgorithm_ != AlgorithmType::AdaptivePID) {
            adaptivePidX_.reset();
            adaptivePidY_.reset();
            lastAppliedAlgorithm_ = AlgorithmType::AdaptivePID;
        }

        float adaptiveErrorX = errorX;
        float adaptiveErrorY = errorY;

        // Smith 先补偿；预测只加性叠加
        bool smithOn = false;
        if (config.smithPredictorEnabled) {
            auto [smithCX, smithCY] = smithPredictor.correct(
                lastOutputX, lastOutputY, adaptiveErrorX, adaptiveErrorY, deltaTime);
            adaptiveErrorX = smithCX;
            adaptiveErrorY = smithCY;
            smithOn = true;
        }

        float predWX = config.predictionWeightX;
        float predWY = config.predictionWeightY;
        if (smithOn && config.immFilterEnabled) {
            predWX *= 0.5f;
            predWY *= 0.5f;
        }

        if (config.immFilterEnabled) {
            immFilter.predict(deltaTime, previousMoveX, previousMoveY);
            immFilter.update(errorX, errorY);
            float immDeltaX = 0.0f, immDeltaY = 0.0f;
            immFilter.getPrediction(deltaTime, immDeltaX, immDeltaY);
            adaptiveErrorX += predWX * immDeltaX;
            adaptiveErrorY += predWY * immDeltaY;
        }
        else if (config.useDerivativePredictor) {
            predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
            float derivPredictedX = 0.0f, derivPredictedY = 0.0f;
            predictor.predict(deltaTime, derivPredictedX, derivPredictedY);
            adaptiveErrorX += predWX * derivPredictedX;
            adaptiveErrorY += predWY * derivPredictedY;
        }

        // dt-aware AdaptivePID (I*dt, D/dt, time-normalized adapt rate)
        moveX = adaptivePidX_.update(adaptiveErrorX, deltaTime);
        moveY = adaptivePidY_.update(adaptiveErrorY, deltaTime);

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
            data.algorithmType = 8;  // 8=AdaptivePID
            data.isFiring = isFiring;
            pidDataCallback_(data);
        }

        previousErrorX = errorX;
        previousErrorY = errorY;
        lastOutputX = moveX;
        lastOutputY = moveY;
    } else {
        // 非上述算法：更新 lastAppliedAlgorithm_ 以便下次切换检测
        lastAppliedAlgorithm_ = config.algorithmType;
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
        float timeFactor = deltaTime * config.targetFrameRate;
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
    
    moveMouse(static_cast<int>(finalMoveX), static_cast<int>(finalMoveY));
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

float AbstractMouseController::getCurrentPGain()
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetLockStartTime).count() / 1000.0f;
    
    float rampFactor = std::min(elapsed / config.pGainRampDuration, 1.0f);
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
    integralGainX = 0.0f;
    integralGainY = 0.0f;
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
    oneEuroX_.reset();
    oneEuroY_.reset();
    oneEuroLockedTrackId_ = -1;
    smithPredictor.reset();
    adaptivePidX_.reset();
    adaptivePidY_.reset();
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
