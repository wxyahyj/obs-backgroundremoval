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
    , stdIntegralX(0.0f)
    , stdIntegralY(0.0f)
    , stdIntegralGainX(0.0f)
    , stdIntegralGainY(0.0f)
    , stdLastErrorX(0.0f)
    , stdLastErrorY(0.0f)
    , stdFilteredDeltaErrorX(0.0f)
    , stdFilteredDeltaErrorY(0.0f)
    , stdPreviousMoveX(0.0f)
    , stdPreviousMoveY(0.0f)
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
    , kalmanFilterInitialized(false)
    , bezierPhase(0.0f)
    , pidDataCallback_(nullptr)
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
    
    // 如果优化器正在运行，保留优化后的参数
    if (optimizer_.isRunning()) {
        // 根据算法类型保存优化器调整的参数
        AlgorithmType algoType = config.algorithmType;
        
        // AdvancedPID 参数
        float savedPidPMin = config.pidPMin;
        float savedPidPMax = config.pidPMax;
        float savedPidD = config.pidD;
        float savedPidI = config.pidI;
        float savedDerivativeFilterAlpha = config.derivativeFilterAlpha;
        float savedKalmanPredX = config.kalmanPredictionWeightX;
        float savedKalmanPredY = config.kalmanPredictionWeightY;
        float savedPredX = config.predictionWeightX;
        float savedPredY = config.predictionWeightY;
        
        // StandardPID 参数
        float savedStdKp = config.stdKp;
        float savedStdKi = config.stdKi;
        float savedStdKd = config.stdKd;
        float savedStdDerivativeFilterAlpha = config.stdDerivativeFilterAlpha;
        float savedStdSmoothingX = config.stdSmoothingX;
        float savedStdSmoothingY = config.stdSmoothingY;
        
        // DopaPID 参数
        float savedDopaKpX = config.dopaKpX;
        float savedDopaKpY = config.dopaKpY;
        float savedDopaKiX = config.dopaKiX;
        float savedDopaKiY = config.dopaKiY;
        float savedDopaKdX = config.dopaKdX;
        float savedDopaKdY = config.dopaKdY;
        float savedDopaPredWeight = config.dopaPredWeight;
        float savedDopaDFilterAlpha = config.dopaDFilterAlpha;
        
        // ChrisPID 参数
        float savedChrisKp = config.chrisKp;
        float savedChrisKi = config.chrisKi;
        float savedChrisKd = config.chrisKd;
        float savedChrisPredWeightX = config.chrisPredWeightX;
        float savedChrisPredWeightY = config.chrisPredWeightY;
        float savedChrisDFilterAlpha = config.chrisDFilterAlpha;
        
        // 应用新配置
        config = newConfig;
        
        // 根据算法类型恢复优化器调整的参数
        switch (algoType) {
            case AlgorithmType::AdvancedPID:
                config.pidPMin = savedPidPMin;
                config.pidPMax = savedPidPMax;
                config.pidD = savedPidD;
                config.pidI = savedPidI;
                config.derivativeFilterAlpha = savedDerivativeFilterAlpha;
                config.kalmanPredictionWeightX = savedKalmanPredX;
                config.kalmanPredictionWeightY = savedKalmanPredY;
                config.predictionWeightX = savedPredX;
                config.predictionWeightY = savedPredY;
                break;
            case AlgorithmType::StandardPID:
                config.stdKp = savedStdKp;
                config.stdKi = savedStdKi;
                config.stdKd = savedStdKd;
                config.stdDerivativeFilterAlpha = savedStdDerivativeFilterAlpha;
                config.stdSmoothingX = savedStdSmoothingX;
                config.stdSmoothingY = savedStdSmoothingY;
                break;
            case AlgorithmType::DopaPID:
                config.dopaKpX = savedDopaKpX;
                config.dopaKpY = savedDopaKpY;
                config.dopaKiX = savedDopaKiX;
                config.dopaKiY = savedDopaKiY;
                config.dopaKdX = savedDopaKdX;
                config.dopaKdY = savedDopaKdY;
                config.dopaPredWeight = savedDopaPredWeight;
                config.dopaDFilterAlpha = savedDopaDFilterAlpha;
                break;
            case AlgorithmType::ChrisPID:
                config.chrisKp = savedChrisKp;
                config.chrisKi = savedChrisKi;
                config.chrisKd = savedChrisKd;
                config.chrisPredWeightX = savedChrisPredWeightX;
                config.chrisPredWeightY = savedChrisPredWeightY;
                config.chrisDFilterAlpha = savedChrisDFilterAlpha;
                break;
        }
    } else {
        config = newConfig;
    }
    
    config.bezierCurvature = std::clamp(config.bezierCurvature, 0.0f, 1.0f);
    config.bezierRandomness = std::clamp(config.bezierRandomness, 0.0f, 0.5f);
    config.kalmanPredictionWeightX = std::clamp(config.kalmanPredictionWeightX, 0.0f, 1.0f);
    config.kalmanPredictionWeightY = std::clamp(config.kalmanPredictionWeightY, 0.0f, 1.0f);
    
    // 更新DerivativePredictor参数
    predictor.setSmoothFactors(config.velocitySmoothFactor, config.accelerationSmoothFactor);
    predictor.setMaxPredictionTime(config.maxPredictionTime);
    
    // 更新优化器配置
    optimizer_.setAlgorithmType(config.algorithmType);
    OptimizerConfig optConfig = optimizer_.getConfig();
    optConfig.enabled = config.optimizationEnabled;
    
    // 设置模式和策略
    optConfig.mode = (config.optimizationMode == 0) ? OptimizationMode::TUNING : OptimizationMode::INDEPENDENT;
    switch (config.optimizationStrategy) {
        case 0: optConfig.strategy = OptimizationStrategy::STABLE_FIRST; break;
        case 1: optConfig.strategy = OptimizationStrategy::BALANCED; break;
        case 2: optConfig.strategy = OptimizationStrategy::AGGRESSIVE; break;
        default: optConfig.strategy = OptimizationStrategy::BALANCED; break;
    }
    
    // 设置采样和迭代参数
    optConfig.sampleFrames = config.optimizationSampleFrames;
    optConfig.maxIterations = config.optimizationMaxIterations;
    optConfig.targetError = config.optimizationTargetError;
    optConfig.allowSpeedOptimization = config.optimizationAllowSpeedOpt;
    optConfig.stepDecayFactor = config.optimizationStepDecayFactor;
    optConfig.minValidSampleRatio = config.optimizationMinValidRatio;
    
    optimizer_.setConfig(optConfig);
    
    // 设置参数更新回调
    if (config.optimizationEnabled && !optimizer_.isRunning()) {
        // 将当前参数值设置到优化器
        std::vector<float> currentParams = extractCurrentParameters();
        optimizer_.setCurrentParameters(currentParams);
        
        optimizer_.setParameterUpdateCallback([this](const std::vector<float>& params) {
            this->applyOptimizedParameters(params);
        });
        optimizer_.start();
        obs_log(LOG_INFO, "[%s] 优化器已启动: algorithmType=%d, sampleFrames=%d, maxIterations=%d",
                getLogPrefix(), static_cast<int>(config.algorithmType), 
                config.optimizationSampleFrames, config.optimizationMaxIterations);
    } else if (!config.optimizationEnabled && optimizer_.isRunning()) {
        optimizer_.stop();
        obs_log(LOG_INFO, "[%s] 优化器已停止", getLogPrefix());
    }
    
    if (configChanged) {
        obs_log(LOG_INFO, "[%s] Config updated: enableMouseControl=%d, autoTriggerEnabled=%d, fireDuration=%dms, interval=%dms",
                getLogPrefix(), config.enableMouseControl, config.autoTriggerEnabled, 
                config.autoTriggerFireDuration, config.autoTriggerInterval);
    }
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
        }
        yUnlockActive = false;
        releaseAutoTrigger();
        return;
    }

    if (!config.continuousAimEnabled) {
        static bool wasHotkeyPressed = false;
        if (!wasHotkeyPressed && hotkeyPressed) {
            hotkeyPressStartTime = std::chrono::steady_clock::now();
            yUnlockActive = false;
        }
        wasHotkeyPressed = hotkeyPressed;

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
    deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTickTime).count() / 1000.0f;
    deltaTime = std::max(0.001f, std::min(deltaTime, 0.05f));
    lastTickTime = now;

    Detection* target = selectTarget();
    if (!target) {
        if (isMoving) {
            isMoving = false;
            // 目标丢失时只重置预测器，不重置积分项
            // 这样积分可以继续累积，PI控制才能真正发挥作用
            predictor.reset();
            kalmanFilter.reset();
            kalmanFilterInitialized = false;
            dopaController_.resetPredictor();
            chrisController_.resetPredictor();
            resetMotionState();
        }
        if (autoTriggerHolding) {
            auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
        }
        return;
    }

    float fovCenterX = config.inferenceFrameWidth / 2.0f;
    float fovCenterY = config.inferenceFrameHeight / 2.0f;

    float targetPixelX = target->centerX * config.inferenceFrameWidth;
    float yOffsetPixels = config.targetYOffset * 0.01f * target->height * config.inferenceFrameHeight;
    float targetPixelY = target->centerY * config.inferenceFrameHeight - yOffsetPixels;

    if (deltaTime > 0.001f) {
        targetVelocityX = (targetPixelX - previousTargetX) / deltaTime;
        targetVelocityY = (targetPixelY - previousTargetY) / deltaTime;
    }
    previousTargetX = targetPixelX;
    previousTargetY = targetPixelY;

    float errorX = targetPixelX - fovCenterX + config.screenOffsetX;
    float errorY = targetPixelY - fovCenterY + config.screenOffsetY;

    float distanceSquared = errorX * errorX + errorY * errorY;
    float deadZoneSquared = config.deadZonePixels * config.deadZonePixels;
    
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
        } else {
            if (distance < config.autoTriggerRadius) {
                if (!autoTriggerWaitingForDelay) {
                    autoTriggerWaitingForDelay = true;
                    autoTriggerDelayStartTime = now;
                }
                
                auto delayElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerDelayStartTime).count();
                int totalDelay = config.autoTriggerFireDelay + getRandomDelay();
                
                if (delayElapsed >= totalDelay) {
                    auto cooldownElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastAutoTriggerTime).count();
                    if (cooldownElapsed >= config.autoTriggerInterval) {
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

    if (config.algorithmType == AlgorithmType::StandardPID) {
        float rawMoveX = calculateStandardPID(errorX, stdIntegralX, stdIntegralGainX, 
                                      stdLastErrorX, stdFilteredDeltaErrorX, deltaTime);
        float rawMoveY = calculateStandardPID(errorY, stdIntegralY, stdIntegralGainY, 
                                      stdLastErrorY, stdFilteredDeltaErrorY, deltaTime);
        
        moveX = stdPreviousMoveX * (1.0f - config.stdSmoothingX) + rawMoveX * config.stdSmoothingX;
        moveY = stdPreviousMoveY * (1.0f - config.stdSmoothingY) + rawMoveY * config.stdSmoothingY;
        stdPreviousMoveX = moveX;
        stdPreviousMoveY = moveY;
        
        static int stdLogCounter = 0;
        if (++stdLogCounter >= 30) {
            stdLogCounter = 0;
            blog(LOG_INFO, "[%s标准PID] errorX=%.1f errorY=%.1f | rawMoveX=%.1f rawMoveY=%.1f | moveX=%.1f moveY=%.1f | stdKp=%.2f stdKd=%.3f",
                 getLogPrefix(), errorX, errorY, rawMoveX, rawMoveY, moveX, moveY, config.stdKp, config.stdKd);
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
            data.currentKp = config.stdKp;
            data.currentKi = config.stdKi;
            data.currentKd = config.stdKd;
            pidDataCallback_(data);
        }
        
        pidPreviousErrorX = 0.0f;
        pidPreviousErrorY = 0.0f;
        previousErrorX = 0.0f;
        previousErrorY = 0.0f;
        filteredDeltaErrorX = 0.0f;
        filteredDeltaErrorY = 0.0f;
        integralX = 0.0f;
        integralY = 0.0f;
    } else if (config.algorithmType == AlgorithmType::DopaPID) {
        DopaPIDConfig dopaConfig;
        dopaConfig.kpX = config.dopaKpX;
        dopaConfig.kpY = config.dopaKpY;
        dopaConfig.kiX = config.dopaKiX;
        dopaConfig.kiY = config.dopaKiY;
        dopaConfig.kdX = config.dopaKdX;
        dopaConfig.kdY = config.dopaKdY;
        dopaConfig.windupGuardX = config.dopaWindupGuardX;
        dopaConfig.windupGuardY = config.dopaWindupGuardY;
        dopaConfig.outputLimitMinX = config.dopaOutputLimitMinX;
        dopaConfig.outputLimitMaxX = config.dopaOutputLimitMaxX;
        dopaConfig.outputLimitMinY = config.dopaOutputLimitMinY;
        dopaConfig.outputLimitMaxY = config.dopaOutputLimitMaxY;
        dopaConfig.predWeight = config.dopaPredWeight;
        dopaConfig.gameFps = config.dopaGameFps;
        dopaConfig.dFilterAlpha = config.dopaDFilterAlpha;
        
        dopaController_.setConfig(dopaConfig);
        dopaController_.compute(targetPixelX, targetPixelY, fovCenterX, fovCenterY, moveX, moveY);
        
        static int dopaLogCounter = 0;
        if (++dopaLogCounter >= 30) {
            dopaLogCounter = 0;
            blog(LOG_INFO, "[DopaPID] errorX=%.1f errorY=%.1f | moveX=%.1f moveY=%.1f | kpX=%.2f kpY=%.2f",
                 errorX, errorY, moveX, moveY, config.dopaKpX, config.dopaKpY);
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
            data.currentKp = config.dopaKpX;
            data.currentKi = config.dopaKiX;
            data.currentKd = config.dopaKdX;
            pidDataCallback_(data);
        }
        
        pidPreviousErrorX = 0.0f;
        pidPreviousErrorY = 0.0f;
        previousErrorX = 0.0f;
        previousErrorY = 0.0f;
        filteredDeltaErrorX = 0.0f;
        filteredDeltaErrorY = 0.0f;
        integralX = 0.0f;
        integralY = 0.0f;
        stdIntegralX = 0.0f;
        stdIntegralY = 0.0f;
        stdIntegralGainX = 0.0f;
        stdIntegralGainY = 0.0f;
    } else if (config.algorithmType == AlgorithmType::ChrisPID) {
        ChrisPIDConfig chrisConfig;
        chrisConfig.kp = config.chrisKp;
        chrisConfig.ki = config.chrisKi;
        chrisConfig.kd = config.chrisKd;
        chrisConfig.predWeightX = config.chrisPredWeightX;
        chrisConfig.predWeightY = config.chrisPredWeightY;
        chrisConfig.initScale = config.chrisInitScale;
        chrisConfig.rampTime = config.chrisRampTime;
        chrisConfig.outputMax = config.chrisOutputMax;
        chrisConfig.iMax = config.chrisIMax;
        chrisConfig.dFilterAlpha = config.chrisDFilterAlpha;
        
        chrisController_.setConfig(chrisConfig);
        double currentTime = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        chrisController_.update(errorX, errorY, currentTime, moveX, moveY);
        
        static int chrisLogCounter = 0;
        if (++chrisLogCounter >= 30) {
            chrisLogCounter = 0;
            blog(LOG_INFO, "[ChrisPID] errorX=%.1f errorY=%.1f | moveX=%.1f moveY=%.1f | kp=%.2f",
                 errorX, errorY, moveX, moveY, config.chrisKp);
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
            data.currentKp = config.chrisKp;
            data.currentKi = config.chrisKi;
            data.currentKd = config.chrisKd;
            pidDataCallback_(data);
        }
        
        pidPreviousErrorX = 0.0f;
        pidPreviousErrorY = 0.0f;
        previousErrorX = 0.0f;
        previousErrorY = 0.0f;
        filteredDeltaErrorX = 0.0f;
        filteredDeltaErrorY = 0.0f;
        integralX = 0.0f;
        integralY = 0.0f;
        stdIntegralX = 0.0f;
        stdIntegralY = 0.0f;
        stdIntegralGainX = 0.0f;
        stdIntegralGainY = 0.0f;
    } else if (config.algorithmType == AlgorithmType::AdvancedPID) {
        float predictedX = 0.0f, predictedY = 0.0f;

        if (config.useKalmanFilter) {
            if (!kalmanFilterInitialized) {
                kalmanFilter.init(targetPixelX, targetPixelY);
                kalmanFilter.setProcessNoise(config.kalmanProcessNoise);
                kalmanFilter.setMeasurementNoise(config.kalmanMeasurementNoise);
                kalmanFilter.setConfidenceScale(config.kalmanConfidenceScale);
                kalmanFilterInitialized = true;
            }

            kalmanFilter.setProcessNoise(config.kalmanProcessNoise);
            kalmanFilter.setMeasurementNoise(config.kalmanMeasurementNoise);
            kalmanFilter.setConfidenceScale(config.kalmanConfidenceScale);

            kalmanFilter.predict(deltaTime);
            kalmanFilter.update(targetPixelX, targetPixelY, target->confidence);
            kalmanFilter.getPredictionOffset(deltaTime, targetPixelX, targetPixelY, predictedX, predictedY);
        } 
        
        float derivPredictedX = 0.0f, derivPredictedY = 0.0f;
        if (config.useDerivativePredictor) {
            predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
            predictor.predict(deltaTime, derivPredictedX, derivPredictedY);
        }

        // 卡尔曼预测：预测目标位置偏移，加到误差上（瞄准预测位置）
        float fusedErrorX = errorX;
        float fusedErrorY = errorY;
        
        if (config.useKalmanFilter) {
            fusedErrorX += config.kalmanPredictionWeightX * predictedX;
            fusedErrorY += config.kalmanPredictionWeightY * predictedY;
        }

        float dynamicP = calculateDynamicP(distance) * getCurrentPGain();

        float dTermX = 0.0f;
        float dTermY = 0.0f;
        if (deltaTime > 1e-6f) {
            float deltaErrorX = fusedErrorX - pidPreviousErrorX;
            float deltaErrorY = fusedErrorY - pidPreviousErrorY;
            dTermX = deltaErrorX / deltaTime * config.pidD;
            dTermY = deltaErrorY / deltaTime * config.pidD;
        }
        dTermX = std::clamp(dTermX, -50.0f, 50.0f);
        dTermY = std::clamp(dTermY, -50.0f, 50.0f);

        float alpha = config.derivativeFilterAlpha;
        filteredDeltaErrorX = alpha * dTermX + (1.0f - alpha) * filteredDeltaErrorX;
        filteredDeltaErrorY = alpha * dTermY + (1.0f - alpha) * filteredDeltaErrorY;

        // 自适应积分增益控制
        bool shouldIntegrateX = adjustIntegralGain(fusedErrorX, pidPreviousErrorX, integralGainX);
        bool shouldIntegrateY = adjustIntegralGain(fusedErrorY, pidPreviousErrorY, integralGainY);

        if (std::abs(fusedErrorX) >= config.integralDeadZone && shouldIntegrateX) {
            integralX += fusedErrorX;
            integralX = std::clamp(integralX, -config.integralLimit, config.integralLimit);
        } else {
            integralX *= 0.9f;  // 衰减而非完全清零
        }
        if (std::abs(fusedErrorY) >= config.integralDeadZone && shouldIntegrateY) {
            integralY += fusedErrorY;
            integralY = std::clamp(integralY, -config.integralLimit, config.integralLimit);
        } else {
            integralY *= 0.9f;  // 衰减而非完全清零
        }

        float integralTermX = config.pidI * integralX;
        float integralTermY = config.pidI * integralY;

        float pidOutputX = dynamicP * fusedErrorX + filteredDeltaErrorX + integralTermX;
        float pidOutputY = dynamicP * fusedErrorY + filteredDeltaErrorY + integralTermY;

        // 导数预测作为前馈项：预测目标移动方向，提前移动
        // 注意：前馈项直接加到输出上，不经过PID计算
        if (config.useDerivativePredictor) {
            pidOutputX += config.predictionWeightX * derivPredictedX;
            pidOutputY += config.predictionWeightY * derivPredictedY;
        }

        moveX = pidOutputX;
        moveY = pidOutputY;

        static int logCounter = 0;
        if (++logCounter >= 30) {
            logCounter = 0;
            blog(LOG_INFO, "[%s高级PID] dt=%.4f | errorX=%.1f errorY=%.1f | fusedX=%.1f fusedY=%.1f | dynamicP=%.3f",
                 getLogPrefix(), deltaTime, errorX, errorY, fusedErrorX, fusedErrorY, dynamicP);
            blog(LOG_INFO, "[%s高级PID] dTermX=%.2f dTermY=%.2f | filteredDX=%.2f filteredDY=%.2f | pidD=%.4f",
                 getLogPrefix(), dTermX, dTermY, filteredDeltaErrorX, filteredDeltaErrorY, config.pidD);
            blog(LOG_INFO, "[%s高级PID] integralX=%.2f integralY=%.2f | iTermX=%.2f iTermY=%.2f | iGainX=%.2f iGainY=%.2f | pidI=%.4f | outX=%.1f outY=%.1f",
                 getLogPrefix(), integralX, integralY, integralTermX, integralTermY, integralGainX, integralGainY, config.pidI, pidOutputX, pidOutputY);
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
            data.currentKp = dynamicP;
            data.currentKi = config.pidI;
            data.currentKd = config.pidD;
            pidDataCallback_(data);
        }

        pidPreviousErrorX = fusedErrorX;
        pidPreviousErrorY = fusedErrorY;
        previousErrorX = errorX;
        previousErrorY = errorY;
        
        stdIntegralX = 0.0f;
        stdIntegralY = 0.0f;
        stdIntegralGainX = 0.0f;
        stdIntegralGainY = 0.0f;
        stdLastErrorX = errorX;
        stdLastErrorY = errorY;
        stdFilteredDeltaErrorX = 0.0f;
        stdFilteredDeltaErrorY = 0.0f;
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

    if (config.autoRecoilControlEnabled && firing) {
        float recoilPerMs = config.recoilStrength / static_cast<float>(config.recoilSpeed);
        float recoilThisFrame = recoilPerMs * deltaTime * 1000.0f;
        moveY += recoilThisFrame;
    }

    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;
    
    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;
    
    // 收集优化器数据（增强版）
    if (config.optimizationEnabled && optimizer_.isRunning()) {
        float predX = predictedTargetPos.x - currentTargetPos.x;
        float predY = predictedTargetPos.y - currentTargetPos.y;
        bool saturated = (std::abs(finalMoveX) >= config.outputMaxX || 
                         std::abs(finalMoveY) >= config.outputMaxY);
        
        SampleType sampleType = SampleType::STICKY;
        if (lockedTrackId < 0 && pendingTargetTrackId >= 0) {
            sampleType = SampleType::REACQ;
        }
        
        optimizer_.addSample(errorX, errorY, finalMoveX, finalMoveY,
                            targetVelocityX, targetVelocityY,
                            predX, predY, saturated, sampleType);
        
        // 每帧调用优化器更新
        optimizer_.update();
    }
    
    moveMouse(static_cast<int>(finalMoveX), static_cast<int>(finalMoveY));
}

Detection* AbstractMouseController::selectTarget()
{
    if (currentDetections.empty()) {
        lockedTrackId = -1;
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
        lockedTrackId = -1;
        pendingTargetTrackId = -1;
        currentTargetScore = 0.0f;
        pendingTargetScore = 0.0f;
        currentTargetDistance = 0.0f;
        return nullptr;
    }

    // 如果当前没有锁定目标，直接选择最佳目标
    if (lockedTrackId < 0 || !currentTarget) {
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

bool AbstractMouseController::adjustIntegralGain(float error, float lastError, float& integralGain)
{
    float errorDerivative = std::abs(error - lastError);

    if (std::abs(error) < config.integralSeparationThreshold) {
        float adaptRate = config.integralRate * (1.0f - errorDerivative / (config.integralSeparationThreshold * 2.0f));
        adaptRate = (adaptRate < 0.0f) ? 0.0f : ((adaptRate > config.integralRate) ? config.integralRate : adaptRate);
        integralGain = (integralGain + adaptRate > 1.0f) ? 1.0f : integralGain + adaptRate;
    }
    else {
        float decay = 0.1f + 0.9f * std::tanh(std::abs(error) / (config.integralSeparationThreshold * 2.0f));
        integralGain *= (1.0f - 0.05f * decay);
    }
    integralGain = (integralGain < 0.0f) ? 0.0f : ((integralGain > 1.0f) ? 1.0f : integralGain);
    return integralGain > 0.01f;
}

float AbstractMouseController::calculateIntegral(float error, float& integral, float& integralGain, float lastError, float deltaTime)
{
    UNUSED_PARAMETER(deltaTime);
    
    if (std::abs(error) < config.integralDeadZone) {
        return 0.0f;
    }

    if (adjustIntegralGain(error, lastError, integralGain))
    {
        integral += error;
        integral = std::max(-config.integralLimit, std::min(integral, config.integralLimit));
    }
    else
    {
        integral = 0;
    }

    float ki = config.pidI * integral;
    
    return ki;
}

float AbstractMouseController::getCurrentPGain()
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetLockStartTime).count() / 1000.0f;
    
    float rampFactor = std::min(elapsed / config.pGainRampDuration, 1.0f);
    float currentScale = config.pGainRampInitialScale + (1.0f - config.pGainRampInitialScale) * rampFactor;
    
    return currentScale;
}

float AbstractMouseController::calculateStandardPID(float error, float& integral, float& integralGain, 
                                             float& lastError, float& filteredDeltaError, float deltaTime)
{
    UNUSED_PARAMETER(deltaTime);
    
    if (std::abs(error) < config.stdDeadZone) {
        error = 0.0f;
    }

    float kp = config.stdKp * error;

    if (adjustStandardIntegral(error, lastError, integralGain))
    {
        integral += error;
        integral = std::clamp(integral, -config.stdIntegralLimit, config.stdIntegralLimit);
    }
    else
    {
        integral = 0;
    }

    float ki = (std::abs(integral) > config.stdIntegralDeadzone) ? config.stdKi * integral : 0;

    float deltaError = error - lastError;
    filteredDeltaError = config.stdDerivativeFilterAlpha * deltaError + 
                         (1.0f - config.stdDerivativeFilterAlpha) * filteredDeltaError;
    float kd = config.stdKd * filteredDeltaError;

    float output = kp + ki + kd;

    output = std::clamp(output, -config.stdOutputLimit, config.stdOutputLimit);

    lastError = error;

    return output;
}

bool AbstractMouseController::adjustStandardIntegral(float error, float lastError, float& integralGain)
{
    float errorDerivative = std::abs(error - lastError);

    if (std::abs(error) < config.stdIntegralThreshold) {
        float adaptRate = config.stdIntegralRate * (1.0f - errorDerivative / (config.stdIntegralThreshold * 2.0f));
        adaptRate = (adaptRate < 0.0f) ? 0.0f : ((adaptRate > config.stdIntegralRate) ? config.stdIntegralRate : adaptRate);
        integralGain = (integralGain + adaptRate > 1.0f) ? 1.0f : integralGain + adaptRate;
    }
    else {
        float decay = 0.1f + 0.9f * std::tanh(std::abs(error) / (config.stdIntegralThreshold * 2.0f));
        integralGain *= (1.0f - 0.05f * decay);
    }
    integralGain = (integralGain < 0.0f) ? 0.0f : ((integralGain > 1.0f) ? 1.0f : integralGain);
    return integralGain > 0.01f;
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
    predictor.reset();
    kalmanFilter.reset();
    kalmanFilterInitialized = false;
    stdIntegralX = 0.0f;
    stdIntegralY = 0.0f;
    stdIntegralGainX = 0.0f;
    stdIntegralGainY = 0.0f;
    stdLastErrorX = 0.0f;
    stdLastErrorY = 0.0f;
    stdFilteredDeltaErrorX = 0.0f;
    stdFilteredDeltaErrorY = 0.0f;
    stdPreviousMoveX = 0.0f;
    stdPreviousMoveY = 0.0f;
    lockedTrackId = -1;
    pendingTargetTrackId = -1;
    pendingTargetScore = 0.0f;
    currentTargetScore = 0.0f;
    dopaController_.reset();
    chrisController_.reset();
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

bool AbstractMouseController::getKalmanPrediction(float& predX, float& predY) const
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!kalmanFilterInitialized || !config.useKalmanFilter) {
        return false;
    }
    kalmanFilter.getPrediction(deltaTime, predX, predY);
    return true;
}

void AbstractMouseController::setPidDataCallback(PidDataCallback callback)
{
    pidDataCallback_ = callback;
}

const char* AbstractMouseController::getLogPrefix() const
{
    return "";
}

std::vector<float> AbstractMouseController::extractCurrentParameters()
{
    std::vector<float> params;
    
    switch (config.algorithmType) {
        case AlgorithmType::AdvancedPID:
            params = {
                config.pidPMin,
                config.pidPMax,
                config.pidD,
                config.pidI,
                config.derivativeFilterAlpha,
                config.kalmanPredictionWeightX,
                config.kalmanPredictionWeightY,
                config.predictionWeightX,
                config.predictionWeightY
            };
            obs_log(LOG_INFO, "[HillClimbing] 提取当前AdvancedPID参数: PMin=%.3f PMax=%.3f D=%.4f I=%.4f",
                    config.pidPMin, config.pidPMax, config.pidD, config.pidI);
            break;
            
        case AlgorithmType::StandardPID:
            params = {
                config.stdKp,
                config.stdKi,
                config.stdKd,
                config.stdDerivativeFilterAlpha,
                config.stdSmoothingX,
                config.stdSmoothingY
            };
            obs_log(LOG_INFO, "[HillClimbing] 提取当前StandardPID参数: Kp=%.3f Ki=%.4f Kd=%.4f",
                    config.stdKp, config.stdKi, config.stdKd);
            break;
            
        case AlgorithmType::DopaPID:
            params = {
                config.dopaKpX,
                config.dopaKpY,
                config.dopaKiX,
                config.dopaKiY,
                config.dopaKdX,
                config.dopaKdY,
                config.dopaPredWeight,
                config.dopaDFilterAlpha
            };
            obs_log(LOG_INFO, "[HillClimbing] 提取当前DopaPID参数: KpX=%.3f KpY=%.3f",
                    config.dopaKpX, config.dopaKpY);
            break;
            
        case AlgorithmType::ChrisPID:
            params = {
                config.chrisKp,
                config.chrisKi,
                config.chrisKd,
                config.chrisPredWeightX,
                config.chrisPredWeightY,
                config.chrisDFilterAlpha
            };
            obs_log(LOG_INFO, "[HillClimbing] 提取当前ChrisPID参数: Kp=%.3f Ki=%.4f Kd=%.4f",
                    config.chrisKp, config.chrisKi, config.chrisKd);
            break;
    }
    
    return params;
}

void AbstractMouseController::applyOptimizedParameters(const std::vector<float>& params)
{
    if (params.empty()) return;
    
    switch (config.algorithmType) {
        case AlgorithmType::AdvancedPID:
            if (params.size() >= 9) {
                config.pidPMin = params[0];
                config.pidPMax = params[1];
                config.pidD = params[2];
                config.pidI = params[3];
                config.derivativeFilterAlpha = params[4];
                config.kalmanPredictionWeightX = params[5];
                config.kalmanPredictionWeightY = params[6];
                config.predictionWeightX = params[7];
                config.predictionWeightY = params[8];
                
                obs_log(LOG_INFO, "[HillClimbing] AdvancedPID参数更新: PMin=%.3f PMax=%.3f D=%.4f I=%.4f",
                        config.pidPMin, config.pidPMax, config.pidD, config.pidI);
            }
            break;
            
        case AlgorithmType::StandardPID:
            if (params.size() >= 6) {
                config.stdKp = params[0];
                config.stdKi = params[1];
                config.stdKd = params[2];
                config.stdDerivativeFilterAlpha = params[3];
                config.stdSmoothingX = params[4];
                config.stdSmoothingY = params[5];
                
                obs_log(LOG_INFO, "[HillClimbing] StandardPID参数更新: Kp=%.3f Ki=%.4f Kd=%.4f",
                        config.stdKp, config.stdKi, config.stdKd);
            }
            break;
            
        case AlgorithmType::DopaPID:
            if (params.size() >= 8) {
                config.dopaKpX = params[0];
                config.dopaKpY = params[1];
                config.dopaKiX = params[2];
                config.dopaKiY = params[3];
                config.dopaKdX = params[4];
                config.dopaKdY = params[5];
                config.dopaPredWeight = params[6];
                config.dopaDFilterAlpha = params[7];
                
                obs_log(LOG_INFO, "[HillClimbing] DopaPID参数更新: KpX=%.3f KpY=%.3f KiX=%.4f KiY=%.4f",
                        config.dopaKpX, config.dopaKpY, config.dopaKiX, config.dopaKiY);
            }
            break;
            
        case AlgorithmType::ChrisPID:
            if (params.size() >= 6) {
                config.chrisKp = params[0];
                config.chrisKi = params[1];
                config.chrisKd = params[2];
                config.chrisPredWeightX = params[3];
                config.chrisPredWeightY = params[4];
                config.chrisDFilterAlpha = params[5];
                
                obs_log(LOG_INFO, "[HillClimbing] ChrisPID参数更新: Kp=%.3f Ki=%.4f Kd=%.4f",
                        config.chrisKp, config.chrisKi, config.chrisKd);
            }
            break;
    }
}

#endif
