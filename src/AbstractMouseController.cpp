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
    , enableNeuralPath_(false)
    , neuralPathInitialized_(false)
    , enableNeuralPathDebug_(false)
    , neuralPathIndex_(0)
    , lastNeuralTargetX_(0.0f)
    , lastNeuralTargetY_(0.0f)
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
    
    // 更新DerivativePredictor参数
    predictor.setMaxPredictionTime(config.maxPredictionTime);
    
    // 更新神经网络轨迹生成器配置
    enableNeuralPath_ = config.enableNeuralPath;
    enableNeuralPathDebug_ = config.enableNeuralPathDebug;
    initializeNeuralPathIfNeeded();
    
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
    static int targetFrameCount = 0;
    targetFrameCount++;
    if (targetFrameCount % 60 == 1 && enableNeuralPathDebug_) {
        obs_log(LOG_INFO, "[%s] TARGET SELECTED: classId=%d, conf=%.2f, center=(%.3f,%.3f), wh=(%.3f,%.3f)",
                getLogPrefix(), target->classId, target->confidence,
                target->centerX, target->centerY, target->width, target->height);
    }
    
    // 诊断：检测框稳定性追踪
    static float lastCenterX = 0, lastCenterY = 0;
    static float maxCenterDelta = 0;
    float centerDeltaX = std::abs(target->centerX - lastCenterX);
    float centerDeltaY = std::abs(target->centerY - lastCenterY);
    float maxDeltaThisFrame = std::max(centerDeltaX, centerDeltaY);
    if (maxDeltaThisFrame > maxCenterDelta) maxCenterDelta = maxDeltaThisFrame;
    lastCenterX = target->centerX;
    lastCenterY = target->centerY;
    
    if (targetFrameCount % 120 == 1 && enableNeuralPathDebug_) {
        obs_log(LOG_INFO, "[%s] DETECTION STABILITY: centerDelta=(%.4f,%.4f) maxDelta=%.4f (120帧内)",
                getLogPrefix(), centerDeltaX, centerDeltaY, maxCenterDelta);
        maxCenterDelta = 0;
    }

    float fovCenterX = config.inferenceFrameWidth / 2.0f;
    float fovCenterY = config.inferenceFrameHeight / 2.0f;
    
    // 准星位置优先：如果有准星检测结果，用准星位置作为瞄准起点
    if (aimOriginX_ >= 0.0f && aimOriginY_ >= 0.0f) {
        fovCenterX = aimOriginX_;
        fovCenterY = aimOriginY_;
    }
    
    if (targetFrameCount % 60 == 1 && enableNeuralPathDebug_) {
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
        static int neuralLogCount = 0;
        neuralLogCount++;
        if (neuralLogCount % 60 == 1 && enableNeuralPathDebug_) {
            obs_log(LOG_INFO, "[%s] NEURAL PATH ACTIVE: using neural trajectory (enabled=%d, initialized=%d)",
                    getLogPrefix(), enableNeuralPath_ ? 1 : 0, neuralPathInitialized_ ? 1 : 0);
        }
        
        // 计算目标相对位置
        double relativeTargetX = targetPixelX - fovCenterX;
        double relativeTargetY = targetPixelY - fovCenterY;
        
        static int frameCount = 0;
        frameCount++;
        if (frameCount % 30 == 1 && enableNeuralPathDebug_) {
            obs_log(LOG_INFO, "[%s] NeuralPath FRAME=%d: target=(%.1f,%.1f) fovCenter=(%.1f,%.1f) relative=(%.1f,%.1f) initialized=%d",
                    getLogPrefix(), frameCount, targetPixelX, targetPixelY, fovCenterX, fovCenterY,
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

            static int moveFrameCount = 0;
            moveFrameCount++;
            if (moveFrameCount % 10 == 1 && enableNeuralPathDebug_) {
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

    float distanceSquared = errorX * errorX + errorY * errorY;
    float deadZoneSquared = config.deadZonePixels * config.deadZonePixels;
    
    static int deadZoneFrameCount = 0;
    deadZoneFrameCount++;
    if (deadZoneFrameCount % 60 == 1 && enableNeuralPathDebug_) {
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

    if (config.algorithmType == AlgorithmType::AdvancedPID) {
        // 融合误差模式：P、I项基于融合误差，D项基于原始误差
        float fusionErrorX = errorX;
        float fusionErrorY = errorY;
        float derivPredictedX = 0.0f, derivPredictedY = 0.0f;

        if (config.useDerivativePredictor) {
            predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
            predictor.predict(deltaTime, derivPredictedX, derivPredictedY);
            fusionErrorX += config.predictionWeightX * derivPredictedX;
            fusionErrorY += config.predictionWeightY * derivPredictedY;
        }

        float dynamicP = calculateDynamicP(distance) * getCurrentPGain();

        // 动态阈值计算（基于目标宽度）
        float widthRatio = target->width;
        float dynamicCoeff = config.advMinCoefficient + (config.advMaxCoefficient - config.advMinCoefficient) /
            (1.0f + std::exp(-config.advTransitionSharpness * (widthRatio - config.advTransitionMidpoint)));
        float dynamicJudgmentThresholdX = dynamicCoeff * std::abs(fusionErrorX) + config.integralDeadZone;
        float dynamicJudgmentThresholdY = dynamicCoeff * std::abs(fusionErrorY) + config.integralDeadZone;

        // 状态机：达标判断（X轴）
        if (!advHasReachedX && std::abs(fusionErrorX) < config.advTargetThreshold) {
            advHasReachedX = true;
        } else if (std::abs(fusionErrorX) >= dynamicJudgmentThresholdX) {
            advHasReachedX = false;
            advStableCountX = 0;
        } else if (!advHasReachedX && std::abs(fusionErrorX) >= config.advTargetThreshold && std::abs(fusionErrorX) <= dynamicJudgmentThresholdX) {
            float diff = std::abs(fusionErrorX - pidPreviousErrorX);
            if (diff < config.advTargetThreshold) {
                advStableCountX++;
            } else {
                advStableCountX = 0;
            }
            if (advStableCountX >= 2) {
                advHasReachedX = true;
                advStableCountX = 0;
            }
        }

        // 状态机：达标判断（Y轴）
        if (!advHasReachedY && std::abs(fusionErrorY) < config.advTargetThreshold) {
            advHasReachedY = true;
        } else if (std::abs(fusionErrorY) >= dynamicJudgmentThresholdY) {
            advHasReachedY = false;
            advStableCountY = 0;
        } else if (!advHasReachedY && std::abs(fusionErrorY) >= config.advTargetThreshold && std::abs(fusionErrorY) <= dynamicJudgmentThresholdY) {
            float diff = std::abs(fusionErrorY - pidPreviousErrorY);
            if (diff < config.advTargetThreshold) {
                advStableCountY++;
            } else {
                advStableCountY = 0;
            }
            if (advStableCountY >= 2) {
                advHasReachedY = true;
                advStableCountY = 0;
            }
        }

        // 速度因子：达标时全速，未达标时半速
        float speedFactorX = advHasReachedX ? 1.0f : config.advSpeedFactor;
        float speedFactorY = advHasReachedY ? 1.0f : config.advSpeedFactor;

        // D项基于原始误差计算（避免双重滤波问题）
        float deltaErrorX = errorX - pidPreviousErrorX;
        float deltaErrorY = errorY - pidPreviousErrorY;

        float alpha = config.derivativeFilterAlpha;
        filteredDeltaErrorX = alpha * deltaErrorX + (1.0f - alpha) * filteredDeltaErrorX;
        filteredDeltaErrorY = alpha * deltaErrorY + (1.0f - alpha) * filteredDeltaErrorY;

        float dTermX = config.pidD * filteredDeltaErrorX;
        float dTermY = config.pidD * filteredDeltaErrorY;

        // 自适应积分基于融合误差
        bool shouldIntegrateX = adjustIntegralGain(fusionErrorX, pidPreviousErrorX, integralGainX);
        bool shouldIntegrateY = adjustIntegralGain(fusionErrorY, pidPreviousErrorY, integralGainY);

        if (std::abs(fusionErrorX) >= config.integralDeadZone && shouldIntegrateX) {
            integralX += fusionErrorX;
            integralX = std::clamp(integralX, -config.integralLimit, config.integralLimit);
        } else {
            integralX = 0.0f;
        }
        if (std::abs(fusionErrorY) >= config.integralDeadZone && shouldIntegrateY) {
            integralY += fusionErrorY;
            integralY = std::clamp(integralY, -config.integralLimit, config.integralLimit);
        } else {
            integralY = 0.0f;
        }

        float integralTermX = config.pidI * integralX;
        float integralTermY = config.pidI * integralY;

        // PID输出：P、I项基于融合误差，D项基于原始误差，应用速度因子
        float pidOutputX = dynamicP * fusionErrorX * speedFactorX + dTermX + integralTermX;
        float pidOutputY = dynamicP * fusionErrorY * speedFactorY + dTermY + integralTermY;

        // 输出整体平滑
        if (config.useOneEuroFilter) {
            // 一欧元滤波器：自适应平滑
            oneEuroX.setMinCutoff(config.oneEuroMinCutoff);
            oneEuroX.setBeta(config.oneEuroBeta);
            oneEuroX.setDCutoff(config.oneEuroDCutoff);
            oneEuroY.setMinCutoff(config.oneEuroMinCutoff);
            oneEuroY.setBeta(config.oneEuroBeta);
            oneEuroY.setDCutoff(config.oneEuroDCutoff);

            moveX = oneEuroX.filter(pidOutputX, deltaTime);
            moveY = oneEuroY.filter(pidOutputY, deltaTime);
        } else {
            // EMA平滑
            moveX = advPreviousOutputX * (1.0f - config.advOutputSmoothing) + pidOutputX * config.advOutputSmoothing;
            moveY = advPreviousOutputY * (1.0f - config.advOutputSmoothing) + pidOutputY * config.advOutputSmoothing;
            advPreviousOutputX = moveX;
            advPreviousOutputY = moveY;
        }

        static int logCounter = 0;
        if (++logCounter >= 30) {
            logCounter = 0;
            blog(LOG_INFO, "[%s高级PID-融合] dt=%.4f | errorX=%.1f errorY=%.1f | fusionX=%.1f fusionY=%.1f | dynamicP=%.3f",
                 getLogPrefix(), deltaTime, errorX, errorY, fusionErrorX, fusionErrorY, dynamicP);
            blog(LOG_INFO, "[%s高级PID-融合] reachedX=%d reachedY=%d | speedX=%.1f speedY=%.1f | threshold=%.1f",
                 getLogPrefix(), advHasReachedX ? 1 : 0, advHasReachedY ? 1 : 0, speedFactorX, speedFactorY, dynamicJudgmentThresholdX);
            blog(LOG_INFO, "[%s高级PID-融合] dTermX=%.2f dTermY=%.2f (基于原始误差) | filteredDX=%.2f filteredDY=%.2f | pidD=%.4f",
                 getLogPrefix(), dTermX, dTermY, filteredDeltaErrorX, filteredDeltaErrorY, config.pidD);
            blog(LOG_INFO, "[%s高级PID-融合] integralX=%.2f integralY=%.2f | iTermX=%.2f iTermY=%.2f | iGainX=%.2f iGainY=%.2f | pidI=%.4f | outX=%.1f outY=%.1f",
                 getLogPrefix(), integralX, integralY, integralTermX, integralTermY, integralGainX, integralGainY, config.pidI, pidOutputX, pidOutputY);
            if (config.useDerivativePredictor) {
                blog(LOG_INFO, "[%s高级PID-融合] 预测值: derivPredX=%.2f derivPredY=%.2f | weightX=%.2f weightY=%.2f | 融合贡献X=%.2f 融合贡献Y=%.2f",
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

            // 新增：P/I/D 分项值
            data.pTermX = dynamicP * fusionErrorX;
            data.pTermY = dynamicP * fusionErrorY;
            data.iTermX = integralTermX;
            data.iTermY = integralTermY;
            data.dTermX = dTermX;
            data.dTermY = dTermY;

            // 积分状态
            float iLimit = (config.integralLimit > 0.0f) ? config.integralLimit : 1000.0f;
            data.integralAbsX = std::abs(integralX);
            data.integralAbsY = std::abs(integralY);
            data.integralLimitX = iLimit;
            data.integralLimitY = iLimit;
            data.integralRatioX = std::min(1.0f, std::abs(integralX) / iLimit);
            data.integralRatioY = std::min(1.0f, std::abs(integralY) / iLimit);

            // 控制模式自动诊断
            float errDist = std::sqrt(errorX * errorX + errorY * errorY);
            float maxIRatio = std::max(data.integralRatioX, data.integralRatioY);
            bool hasTarget = (errDist > 0.5f || std::abs(targetVelocityX) > 0.1f || std::abs(targetVelocityY) > 0.1f);

            if (!hasTarget) {
                data.controlMode = 0; // IDLE
            } else if (maxIRatio > 0.8f) {
                data.controlMode = 3; // I_SATURATION
            } else if (errDist < 10.0f && maxIRatio < 0.4f) {
                data.controlMode = 2; // LOCKED
            } else {
                data.controlMode = 1; // TRACKING
            }

            data.algorithmType = 0; // AdvancedPID
            data.isFiring = isFiring;
            data.smoothingFactorX = config.aimSmoothingX;
            data.smoothingFactorY = config.aimSmoothingY;

            pidDataCallback_(data);
        }

        // 保存原始误差用于D项计算
        pidPreviousErrorX = errorX;
        pidPreviousErrorY = errorY;
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
    } else if (config.algorithmType == AlgorithmType::DynamicPID) {
        // 动态PID：基于动态阈值和状态机的PID控制器
        dynamicPidX.updateParams(config.dynamicKp, config.dynamicKi, config.dynamicKd);
        dynamicPidY.updateParams(config.dynamicKp, config.dynamicKi, config.dynamicKd);

        dynamicPidX.setBottomParams(
            config.dynamicTargetThreshold, config.dynamicSpeedMultiplier,
            config.dynamicMinCoefficient, config.dynamicMaxCoefficient,
            config.dynamicTransitionSharpness, config.dynamicTransitionMidpoint,
            config.dynamicMinDataPoints, config.dynamicErrorTolerance
        );
        dynamicPidY.setBottomParams(
            config.dynamicTargetThreshold, config.dynamicSpeedMultiplier,
            config.dynamicMinCoefficient, config.dynamicMaxCoefficient,
            config.dynamicTransitionSharpness, config.dynamicTransitionMidpoint,
            config.dynamicMinDataPoints, config.dynamicErrorTolerance
        );

        dynamicPidX.setSmoothingFactor(config.dynamicSmoothingFactor);
        dynamicPidY.setSmoothingFactor(config.dynamicSmoothingFactor);

        // 预测器融合误差（与其他算法一致）
        float dynamicErrorX = errorX;
        float dynamicErrorY = errorY;
        float dynamicDerivPredictedX = 0.0f, dynamicDerivPredictedY = 0.0f;

        if (config.useDerivativePredictor) {
            predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
            predictor.predict(deltaTime, dynamicDerivPredictedX, dynamicDerivPredictedY);
            dynamicErrorX += config.predictionWeightX * dynamicDerivPredictedX;
            dynamicErrorY += config.predictionWeightY * dynamicDerivPredictedY;
        }

        // 目标宽度（像素）用于动态阈值计算
        float targetWidthPixels = target->width * config.inferenceFrameWidth;
        float imageSize = static_cast<float>(config.inferenceFrameWidth);

        moveX = dynamicPidX.controlLoop(dynamicErrorX, deltaTime, targetWidthPixels, imageSize);
        moveY = dynamicPidY.controlLoop(dynamicErrorY, deltaTime, targetWidthPixels, imageSize);

        static int dynamicLogCounter = 0;
        if (++dynamicLogCounter >= 30) {
            dynamicLogCounter = 0;
            blog(LOG_INFO, "[%s动态PID] dt=%.4f | errorX=%.1f errorY=%.1f | dynErrX=%.1f dynErrY=%.1f | moveX=%.1f moveY=%.1f",
                 getLogPrefix(), deltaTime, errorX, errorY, dynamicErrorX, dynamicErrorY, moveX, moveY);
            blog(LOG_INFO, "[%s动态PID] pTermX=%.2f pTermY=%.2f | iTermX=%.2f iTermY=%.2f | dTermX=%.2f dTermY=%.2f",
                 getLogPrefix(),
                 dynamicPidX.getLastProportional(), dynamicPidY.getLastProportional(),
                 dynamicPidX.getLastIntegral(), dynamicPidY.getLastIntegral(),
                 dynamicPidX.getLastDerivative(), dynamicPidY.getLastDerivative());
            blog(LOG_INFO, "[%s动态PID] reachedX=%d reachedY=%d | velX=%.1f velY=%.1f | kp=%.3f ki=%.3f kd=%.3f",
                 getLogPrefix(),
                 dynamicPidX.getIsReached() ? 1 : 0, dynamicPidY.getIsReached() ? 1 : 0,
                 dynamicPidX.getVelocity(), dynamicPidY.getVelocity(),
                 config.dynamicKp, config.dynamicKi, config.dynamicKd);
            if (config.useDerivativePredictor) {
                blog(LOG_INFO, "[%s动态PID] 预测值: derivPredX=%.2f derivPredY=%.2f | weightX=%.2f weightY=%.2f | 融合贡献X=%.2f 融合贡献Y=%.2f",
                     getLogPrefix(), dynamicDerivPredictedX, dynamicDerivPredictedY, config.predictionWeightX, config.predictionWeightY,
                     config.predictionWeightX * dynamicDerivPredictedX, config.predictionWeightY * dynamicDerivPredictedY);
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
            data.currentKp = config.dynamicKp;
            data.currentKi = config.dynamicKi;
            data.currentKd = config.dynamicKd;

            data.pTermX = dynamicPidX.getLastProportional();
            data.pTermY = dynamicPidY.getLastProportional();
            data.iTermX = dynamicPidX.getLastIntegral();
            data.iTermY = dynamicPidY.getLastIntegral();
            data.dTermX = dynamicPidX.getLastDerivative();
            data.dTermY = dynamicPidY.getLastDerivative();

            float iLimitDyn = 100.0f;
            data.integralAbsX = std::abs(data.iTermX);
            data.integralAbsY = std::abs(data.iTermY);
            data.integralLimitX = iLimitDyn;
            data.integralLimitY = iLimitDyn;
            data.integralRatioX = std::min(1.0f, data.integralAbsX / iLimitDyn);
            data.integralRatioY = std::min(1.0f, data.integralAbsY / iLimitDyn);

            float errDist = std::sqrt(errorX * errorX + errorY * errorY);
            bool hasTarget = (errDist > 0.5f || std::abs(targetVelocityX) > 0.1f || std::abs(targetVelocityY) > 0.1f);
            if (!hasTarget) {
                data.controlMode = 0;
            } else if (errDist < 10.0f) {
                data.controlMode = 2;
            } else {
                data.controlMode = 1;
            }

            data.algorithmType = 3;
            data.isFiring = isFiring;
            data.smoothingFactorX = config.dynamicSmoothingFactor;
            data.smoothingFactorY = config.dynamicSmoothingFactor;

            pidDataCallback_(data);
        }

        // 重置其他算法状态
        pidPreviousErrorX = 0.0f;
        pidPreviousErrorY = 0.0f;
        previousErrorX = 0.0f;
        previousErrorY = 0.0f;
        filteredDeltaErrorX = 0.0f;
        filteredDeltaErrorY = 0.0f;
        integralX = 0.0f;
        integralY = 0.0f;
        integralGainX = 0.0f;
        integralGainY = 0.0f;
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

    // [FIX] 压枪补偿移到平滑之后，避免被 aimSmoothingY 稀释
    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;

    if (config.autoRecoilControlEnabled && firing) {
        float recoilPerMs = config.recoilStrength / static_cast<float>(config.recoilSpeed);
        float recoilThisFrame = recoilPerMs * deltaTime * 1000.0f;
        finalMoveY += recoilThisFrame;
    }

    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;
    
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
    advHasReachedX = false;
    advHasReachedY = false;
    advStableCountX = 0;
    advStableCountY = 0;
    advPreviousOutputX = 0.0f;
    advPreviousOutputY = 0.0f;
    lockedTrackId = -1;
    pendingTargetTrackId = -1;
    pendingTargetScore = 0.0f;
    currentTargetScore = 0.0f;
    chrisController_.reset();
    dynamicPidX.reset();
    dynamicPidY.reset();
    adaptiveController_.reset();
    incrementalController_.reset();
    advHasReachedX = false;
    advHasReachedY = false;
    advStableCountX = 0;
    advStableCountY = 0;
    advPreviousOutputX = 0.0f;
    advPreviousOutputY = 0.0f;
    oneEuroX.reset();
    oneEuroY.reset();
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
