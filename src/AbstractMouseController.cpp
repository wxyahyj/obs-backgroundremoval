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
    config = newConfig;
    
    config.bezierCurvature = std::clamp(config.bezierCurvature, 0.0f, 1.0f);
    config.bezierRandomness = std::clamp(config.bezierRandomness, 0.0f, 0.5f);
    config.kalmanPredictionWeightX = std::clamp(config.kalmanPredictionWeightX, 0.0f, 1.0f);
    config.kalmanPredictionWeightY = std::clamp(config.kalmanPredictionWeightY, 0.0f, 1.0f);
    
    // 更新DerivativePredictor参数
    predictor.setSmoothFactors(config.velocitySmoothFactor, config.accelerationSmoothFactor);
    predictor.setMaxPredictionTime(config.maxPredictionTime);
    
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

        float iIncrementX = fusedErrorX * deltaTime * config.pidI;
        float iIncrementY = fusedErrorY * deltaTime * config.pidI;
        
        if (std::abs(fusedErrorX) >= config.integralDeadZone) {
            integralX += iIncrementX;
            integralX = std::clamp(integralX, -config.integralLimit, config.integralLimit);
        }
        if (std::abs(fusedErrorY) >= config.integralDeadZone) {
            integralY += iIncrementY;
            integralY = std::clamp(integralY, -config.integralLimit, config.integralLimit);
        }

        float pidOutputX = dynamicP * fusedErrorX + filteredDeltaErrorX + integralX;
        float pidOutputY = dynamicP * fusedErrorY + filteredDeltaErrorY + integralY;

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
            blog(LOG_INFO, "[%s高级PID] dTermX=%.2f dTermY=%.2f | filteredDX=%.2f filteredDY=%.2f | pidD=%.3f",
                 getLogPrefix(), dTermX, dTermY, filteredDeltaErrorX, filteredDeltaErrorY, config.pidD);
            blog(LOG_INFO, "[%s高级PID] iIncX=%.3f iIncY=%.3f | integralX=%.2f integralY=%.2f | pidI=%.4f | pidOutX=%.1f pidOutY=%.1f | moveX=%.1f moveY=%.1f",
                 getLogPrefix(), iIncrementX, iIncrementY, integralX, integralY, config.pidI, pidOutputX, pidOutputY, moveX, moveY);
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
    
    moveMouse(static_cast<int>(finalMoveX), static_cast<int>(finalMoveY));
}

Detection* AbstractMouseController::selectTarget()
{
    if (currentDetections.empty()) {
        lockedTrackId = -1;
        return nullptr;
    }

    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth :
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight :
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    int fovCenterX = frameWidth / 2;
    int fovCenterY = frameHeight / 2;
    float fovRadius = static_cast<float>(config.fovRadiusPixels);

    if (lockedTrackId >= 0) {
        for (auto& det : currentDetections) {
            if (det.trackId == lockedTrackId) {
                float pixelX = det.centerX * frameWidth;
                float pixelY = det.centerY * frameHeight;
                float dx = pixelX - fovCenterX;
                float dy = pixelY - fovCenterY;
                float distSq = dx * dx + dy * dy;
                
                if (distSq <= fovRadius * fovRadius) {
                    currentTargetDistance = std::sqrt(distSq);
                    return &det;
                }
            }
        }
    }

    Detection* bestTarget = nullptr;
    float bestScore = -1.0f;

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
    }

    if (bestTarget) {
        lockedTrackId = bestTarget->trackId;
        float pixelX = bestTarget->centerX * frameWidth;
        float pixelY = bestTarget->centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        currentTargetDistance = std::sqrt(dx * dx + dy * dy);
    } else {
        lockedTrackId = -1;
        currentTargetDistance = 0.0f;
    }

    return bestTarget;
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

#endif
