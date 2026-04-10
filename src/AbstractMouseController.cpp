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
        
        // AdvancedPID 参数 (14个)
        float savedPidPMin = config.pidPMin;
        float savedPidPMax = config.pidPMax;
        float savedPidD = config.pidD;
        float savedPidI = config.pidI;
        float savedDerivativeFilterAlpha = config.derivativeFilterAlpha;
        float savedPredX = config.predictionWeightX;
        float savedPredY = config.predictionWeightY;
        float savedSmoothX = config.aimSmoothingX;
        float savedSmoothY = config.aimSmoothingY;
        float savedMaxPredTime = config.maxPredictionTime;
        
        // StandardPID 参数 (7个)
        float savedStdKp = config.stdKp;
        float savedStdKi = config.stdKi;
        float savedStdKd = config.stdKd;
        float savedStdDerivativeFilterAlpha = config.stdDerivativeFilterAlpha;
        float savedStdSmoothingX = config.stdSmoothingX;
        float savedStdSmoothingY = config.stdSmoothingY;
        float savedStdMaxPredTime = config.maxPredictionTime;
        
        // ChrisPID 参数 (8个)
        float savedChrisKp = config.chrisKp;
        float savedChrisKi = config.chrisKi;
        float savedChrisKd = config.chrisKd;
        float savedChrisPredWeightX = config.chrisPredWeightX;
        float savedChrisPredWeightY = config.chrisPredWeightY;
        float savedChrisDFilterAlpha = config.chrisDFilterAlpha;
        float savedChrisSmoothX = config.aimSmoothingX;
        float savedChrisSmoothY = config.aimSmoothingY;
        
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
                config.predictionWeightX = savedPredX;
                config.predictionWeightY = savedPredY;
                config.aimSmoothingX = savedSmoothX;
                config.aimSmoothingY = savedSmoothY;
                config.maxPredictionTime = savedMaxPredTime;
                break;
            case AlgorithmType::StandardPID:
                config.stdKp = savedStdKp;
                config.stdKi = savedStdKi;
                config.stdKd = savedStdKd;
                config.stdDerivativeFilterAlpha = savedStdDerivativeFilterAlpha;
                config.stdSmoothingX = savedStdSmoothingX;
                config.stdSmoothingY = savedStdSmoothingY;
                config.maxPredictionTime = savedStdMaxPredTime;
                break;
            case AlgorithmType::ChrisPID:
                config.chrisKp = savedChrisKp;
                config.chrisKi = savedChrisKi;
                config.chrisKd = savedChrisKd;
                config.chrisPredWeightX = savedChrisPredWeightX;
                config.chrisPredWeightY = savedChrisPredWeightY;
                config.chrisDFilterAlpha = savedChrisDFilterAlpha;
                config.aimSmoothingX = savedChrisSmoothX;
                config.aimSmoothingY = savedChrisSmoothY;
                break;
        }
    } else {
        config = newConfig;
    }
    
    config.bezierCurvature = std::clamp(config.bezierCurvature, 0.0f, 1.0f);
    config.bezierRandomness = std::clamp(config.bezierRandomness, 0.0f, 0.5f);
    
    // 更新DerivativePredictor参数
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
        float derivPredictedX = 0.0f, derivPredictedY = 0.0f;
        float fusionErrorX = errorX;
        float fusionErrorY = errorY;
        
        if (config.useDerivativePredictor) {
            predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
            predictor.predict(deltaTime, derivPredictedX, derivPredictedY);
            fusionErrorX += config.predictionWeightX * derivPredictedX;
            fusionErrorY += config.predictionWeightY * derivPredictedY;
        }

        float rawMoveX = calculateStandardPID(fusionErrorX, stdIntegralX, stdIntegralGainX, 
                                      stdLastErrorX, stdFilteredDeltaErrorX, deltaTime);
        float rawMoveY = calculateStandardPID(fusionErrorY, stdIntegralY, stdIntegralGainY, 
                                      stdLastErrorY, stdFilteredDeltaErrorY, deltaTime);
        
        moveX = stdPreviousMoveX * (1.0f - config.stdSmoothingX) + rawMoveX * config.stdSmoothingX;
        moveY = stdPreviousMoveY * (1.0f - config.stdSmoothingY) + rawMoveY * config.stdSmoothingY;
        stdPreviousMoveX = moveX;
        stdPreviousMoveY = moveY;
        
        static int stdLogCounter = 0;
        if (++stdLogCounter >= 30) {
            stdLogCounter = 0;
            blog(LOG_INFO, "[%s标准PID] errorX=%.1f errorY=%.1f | fusionX=%.1f fusionY=%.1f | rawMoveX=%.1f rawMoveY=%.1f | moveX=%.1f moveY=%.1f",
                 getLogPrefix(), errorX, errorY, fusionErrorX, fusionErrorY, rawMoveX, rawMoveY, moveX, moveY);
            if (config.useDerivativePredictor) {
                blog(LOG_INFO, "[%s标准PID] 预测值: derivPredX=%.2f derivPredY=%.2f | weightX=%.2f weightY=%.2f | 融合贡献X=%.2f 融合贡献Y=%.2f",
                     getLogPrefix(), derivPredictedX, derivPredictedY, config.predictionWeightX, config.predictionWeightY,
                     config.predictionWeightX * derivPredictedX, config.predictionWeightY * derivPredictedY);
            }
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
        previousErrorX = errorX;
        previousErrorY = errorY;
        filteredDeltaErrorX = 0.0f;
        filteredDeltaErrorY = 0.0f;
        integralX = 0.0f;
        integralY = 0.0f;
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
    } else if (config.algorithmType == AlgorithmType::GaussianGravity) {
        GaussianGravityConfig ggConfig;
        ggConfig.gravityStrength = config.gravityStrength;
        ggConfig.maxDistance = config.gravityMaxDistance;
        ggConfig.softEpsilon = config.gravitySoftEpsilon;
        ggConfig.maxForce = config.gravityMaxForce;
        ggConfig.smoothingFactor = config.gravitySmoothingFactor;
        ggConfig.confidenceScale = true;
        ggConfig.predictionWeight = config.predictionWeightX;
        
        gravityController_.setConfig(ggConfig);
        
        float outX = 0.0f, outY = 0.0f;
        // 计算准星在画面中的位置 (FOV中心 + 偏移)
        float crosshairX = fovCenterX - config.screenOffsetX;
        float crosshairY = fovCenterY - config.screenOffsetY;
        gravityController_.compute(targetPixelX, targetPixelY, 
                                   crosshairX, crosshairY,
                                   target ? target->confidence : 1.0f,
                                   targetVelocityX, targetVelocityY,
                                   outX, outY);
        
        moveX = outX;
        moveY = outY;
        
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
            data.currentKp = config.gravityStrength;
            data.currentKi = 0.0f;
            data.currentKd = 0.0f;
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
        float derivPredictedX = 0.0f, derivPredictedY = 0.0f;
        float fusionErrorX = errorX;
        float fusionErrorY = errorY;
        
        if (config.useDerivativePredictor) {
            predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
            predictor.predict(deltaTime, derivPredictedX, derivPredictedY);
            fusionErrorX += config.predictionWeightX * derivPredictedX;
            fusionErrorY += config.predictionWeightY * derivPredictedY;
        }

        float dynamicP = calculateDynamicP(distance) * getCurrentPGain();

        // D项基于融合误差计算（ChrisPID方式）
        float dTermX = 0.0f;
        float dTermY = 0.0f;
        if (deltaTime > 1e-6f) {
            float deltaErrorX = fusionErrorX - pidPreviousErrorX;
            float deltaErrorY = fusionErrorY - pidPreviousErrorY;
            dTermX = deltaErrorX / deltaTime * config.pidD;
            dTermY = deltaErrorY / deltaTime * config.pidD;
        }
        dTermX = std::clamp(dTermX, -50.0f, 50.0f);
        dTermY = std::clamp(dTermY, -50.0f, 50.0f);

        float alpha = config.derivativeFilterAlpha;
        filteredDeltaErrorX = alpha * dTermX + (1.0f - alpha) * filteredDeltaErrorX;
        filteredDeltaErrorY = alpha * dTermY + (1.0f - alpha) * filteredDeltaErrorY;

        // 自适应积分增益控制（基于融合误差）
        bool shouldIntegrateX = adjustIntegralGain(fusionErrorX, pidPreviousErrorX, integralGainX);
        bool shouldIntegrateY = adjustIntegralGain(fusionErrorY, pidPreviousErrorY, integralGainY);

        if (std::abs(fusionErrorX) >= config.integralDeadZone && shouldIntegrateX) {
            integralX += fusionErrorX;
            integralX = std::clamp(integralX, -config.integralLimit, config.integralLimit);
        } else {
            integralX *= 0.9f;
        }
        if (std::abs(fusionErrorY) >= config.integralDeadZone && shouldIntegrateY) {
            integralY += fusionErrorY;
            integralY = std::clamp(integralY, -config.integralLimit, config.integralLimit);
        } else {
            integralY *= 0.9f;
        }

        float integralTermX = config.pidI * integralX;
        float integralTermY = config.pidI * integralY;

        // PID输出基于融合误差（追预测位置）
        float pidOutputX = dynamicP * fusionErrorX + filteredDeltaErrorX + integralTermX;
        float pidOutputY = dynamicP * fusionErrorY + filteredDeltaErrorY + integralTermY;

        moveX = pidOutputX;
        moveY = pidOutputY;

        static int logCounter = 0;
        if (++logCounter >= 30) {
            logCounter = 0;
            blog(LOG_INFO, "[%s高级PID] dt=%.4f | errorX=%.1f errorY=%.1f | fusionX=%.1f fusionY=%.1f | dynamicP=%.3f",
                 getLogPrefix(), deltaTime, errorX, errorY, fusionErrorX, fusionErrorY, dynamicP);
            blog(LOG_INFO, "[%s高级PID] dTermX=%.2f dTermY=%.2f | filteredDX=%.2f filteredDY=%.2f | pidD=%.4f",
                 getLogPrefix(), dTermX, dTermY, filteredDeltaErrorX, filteredDeltaErrorY, config.pidD);
            blog(LOG_INFO, "[%s高级PID] integralX=%.2f integralY=%.2f | iTermX=%.2f iTermY=%.2f | iGainX=%.2f iGainY=%.2f | pidI=%.4f | outX=%.1f outY=%.1f",
                 getLogPrefix(), integralX, integralY, integralTermX, integralTermY, integralGainX, integralGainY, config.pidI, pidOutputX, pidOutputY);
            if (config.useDerivativePredictor) {
                blog(LOG_INFO, "[%s高级PID] 预测值: derivPredX=%.2f derivPredY=%.2f | weightX=%.2f weightY=%.2f | 融合贡献X=%.2f 融合贡献Y=%.2f",
                     getLogPrefix(), derivPredictedX, derivPredictedY, config.predictionWeightX, config.predictionWeightY,
                     config.predictionWeightX * derivPredictedX, config.predictionWeightY * derivPredictedY);
            }
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

        // 保存融合误差用于D项计算
        pidPreviousErrorX = fusionErrorX;
        pidPreviousErrorY = fusionErrorY;
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
        bool saturated = (std::abs(finalMoveX) >= config.maxPixelMove || 
                         std::abs(finalMoveY) >= config.maxPixelMove);
        
        SampleType sampleType = SampleType::STICKY;
        if (lockedTrackId < 0 && pendingTargetTrackId >= 0) {
            sampleType = SampleType::REACQ;
        }
        
        optimizer_.addSample(errorX, errorY, finalMoveX, finalMoveY,
                            targetVelocityX, targetVelocityY,
                            0.0f, 0.0f, saturated, sampleType);
        
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
            // 9个参数：PID核心(4) + D滤波(1) + 导数预测(2) + 平滑(2)
            params = {
                config.pidPMin,           // 0
                config.pidPMax,           // 1
                config.pidD,              // 2
                config.pidI,              // 3
                config.derivativeFilterAlpha,  // 4
                config.predictionWeightX,   // 5
                config.predictionWeightY,   // 6
                config.aimSmoothingX,      // 7
                config.aimSmoothingY       // 8
            };
            obs_log(LOG_INFO, "[AutoTune] 提取AdvancedPID参数: PMin=%.3f PMax=%.3f D=%.4f I=%.4f PredX=%.2f PredY=%.2f",
                    config.pidPMin, config.pidPMax, config.pidD, config.pidI,
                    config.predictionWeightX, config.predictionWeightY);
            break;
            
        case AlgorithmType::StandardPID:
            // 7个参数
            params = {
                config.stdKp,             // 0
                config.stdKi,             // 1
                config.stdKd,             // 2
                config.stdDerivativeFilterAlpha, // 3
                config.stdSmoothingX,     // 4
                config.stdSmoothingY,     // 5
                config.maxPredictionTime   // 6
            };
            obs_log(LOG_INFO, "[AutoTune] 提取StandardPID参数: Kp=%.3f Ki=%.4f Kd=%.4f",
                    config.stdKp, config.stdKi, config.stdKd);
            break;
            
        case AlgorithmType::ChrisPID:
            // 8个参数
            params = {
                config.chrisKp,           // 0
                config.chrisKi,           // 1
                config.chrisKd,           // 2
                config.chrisPredWeightX,  // 3
                config.chrisPredWeightY,  // 4
                config.chrisDFilterAlpha, // 5
                config.aimSmoothingX,      // 6
                config.aimSmoothingY       // 7
            };
            obs_log(LOG_INFO, "[AutoTune] 提取ChrisPID参数: Kp=%.3f Ki=%.4f Kd=%.4f PredWX=%.2f PredWY=%.2f",
                    config.chrisKp, config.chrisKi, config.chrisKd,
                    config.chrisPredWeightX, config.chrisPredWeightY);
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
                config.predictionWeightX = params[5];
                config.predictionWeightY = params[6];
                config.aimSmoothingX = params[7];
                config.aimSmoothingY = params[8];
                
                obs_log(LOG_INFO, "[AutoTune] AdvancedPID更新: PMin=%.3f PMax=%.3f D=%.4f I=%.4f PredX=%.2f SmoothX=%.2f",
                        config.pidPMin, config.pidPMax, config.pidD, config.pidI,
                        config.predictionWeightX, config.aimSmoothingX);
            }
            break;
            
        case AlgorithmType::StandardPID:
            if (params.size() >= 7) {
                config.stdKp = params[0];
                config.stdKi = params[1];
                config.stdKd = params[2];
                config.stdDerivativeFilterAlpha = params[3];
                config.stdSmoothingX = params[4];
                config.stdSmoothingY = params[5];
                config.maxPredictionTime = params[6];
                
                obs_log(LOG_INFO, "[AutoTune] StandardPID更新: Kp=%.3f Ki=%.4f Kd=%.4f SmoothX=%.2f",
                        config.stdKp, config.stdKi, config.stdKd, config.stdSmoothingX);
            }
            break;
            
        case AlgorithmType::ChrisPID:
            if (params.size() >= 8) {
                config.chrisKp = params[0];
                config.chrisKi = params[1];
                config.chrisKd = params[2];
                config.chrisPredWeightX = params[3];
                config.chrisPredWeightY = params[4];
                config.chrisDFilterAlpha = params[5];
                config.aimSmoothingX = params[6];
                config.aimSmoothingY = params[7];
                
                obs_log(LOG_INFO, "[AutoTune] ChrisPID更新: Kp=%.3f Ki=%.4f Kd=%.4f PredWX=%.2f SmoothX=%.2f",
                        config.chrisKp, config.chrisKi, config.chrisKd,
                        config.chrisPredWeightX, config.aimSmoothingX);
            }
            break;
    }
}

#endif
