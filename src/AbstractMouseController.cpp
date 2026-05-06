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

    if (config.algorithmType == AlgorithmType::AdvancedPID) {
        // 高级PID：完全按照专业PID的精确实现
        
        // 辅助函数：两位小数四舍五入（专业PID的round1）
        auto round1 = [](float value) -> float {
            return std::round(value * 100.0f) / 100.0f;
        };
        
        // 软限幅函数（专业PID的atan2Clamp）
        auto atan2Clamp = [&round1](float value, float softParam, float hardLimit) -> float {
            float angle = std::atan2(value, softParam);
            float gain = softParam - hardLimit * 0.1f;
            return round1(angle * gain);
        };
        
        // 专业PID常量
        constexpr float DEAD_ZONE = 0.3f;           // 死区阈值
        constexpr float JUMP_THRESHOLD = 30.0f;     // 跳变检测阈值
        constexpr float KP_GAIN_THRESHOLD = 1920.0f;
        // KP_GAIN_RATE 现在使用可调参数 config.adaptivePGainRate
        constexpr float INTEGRAL_GAIN_THRESHOLD = 50.0f;
        constexpr float INTEGRAL_GAIN_RATE = 0.1f;
        constexpr float LARGE_ERROR_RATE = 0.1f;
        constexpr float KF2_Q = 0.1f;
        constexpr float KF2_R = 1.0f;
        
        // 动态P增益（根据距离）
        float baseP = calculateDynamicP(distance) * getCurrentPGain();
        
        // 导数预测器：在PID计算前预测目标位置
        float predictedErrorX = errorX;
        float predictedErrorY = errorY;
        if (config.useDerivativePredictor) {
            predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
            float derivPredictedX = 0.0f, derivPredictedY = 0.0f;
            predictor.predict(deltaTime, derivPredictedX, derivPredictedY);
            predictedErrorX = errorX + config.predictionWeightX * derivPredictedX;
            predictedErrorY = errorY + config.predictionWeightY * derivPredictedY;
        }
        
        // ========== X轴处理 ==========
        float absErrorX = std::abs(predictedErrorX);
        float errorX_work = predictedErrorX;
        
        // Step 1: 死区处理
        if (absErrorX <= DEAD_ZONE) {
            errorX_work = 0.0f;
        }
        
        // Step 2: 跳变检测
        float deltaErrorX = errorX_work - pidPreviousErrorX;
        bool jumpDetectedX = std::abs(deltaErrorX) > JUMP_THRESHOLD;
        
        if (jumpDetectedX) {
            adaptivePGainX = 0.0f;
            adaptiveIGainX = 0.0f;
            integralX = 0.0f;
            lastOutputX = 0.0f;
            kf2X.Q_ = 0.0f;
            kf2X.R_ = 0.0f;
            kf3X_x = 0.0f;
            kf3X_P = 0.0f;
            pidPreviousErrorX = 0.0f;
            deltaErrorX = errorX_work;
        }
        
        // Step 3: 自适应积分增益
        {
            float ratio;
            if (absErrorX < INTEGRAL_GAIN_THRESHOLD) {
                ratio = 1.0f - (absErrorX / INTEGRAL_GAIN_THRESHOLD);
                adaptiveIGainX += (ratio - adaptiveIGainX) * INTEGRAL_GAIN_RATE;
            } else {
                ratio = INTEGRAL_GAIN_THRESHOLD / absErrorX;
                adaptiveIGainX += (ratio * adaptiveIGainX - adaptiveIGainX) * LARGE_ERROR_RATE;
            }
            adaptiveIGainX = std::clamp(adaptiveIGainX, 0.0f, 1.0f);
        }
        
        // Step 4: 自适应比例增益
        {
            float ratio;
            if (absErrorX < KP_GAIN_THRESHOLD) {
                ratio = 1.0f - (absErrorX / KP_GAIN_THRESHOLD);
                adaptivePGainX += (ratio - adaptivePGainX) * config.adaptivePGainRate;
            } else {
                ratio = KP_GAIN_THRESHOLD / absErrorX;
                adaptivePGainX += (ratio * adaptivePGainX - adaptivePGainX) * LARGE_ERROR_RATE;
            }
            adaptivePGainX = std::clamp(adaptivePGainX, 0.0f, 1.0f);
        }
        
        // Step 5: 微分计算 + kf2 卡尔曼
        float DX = round1(deltaErrorX + lastOutputX);
        kf2X.Q_ = KF2_Q;
        kf2X.R_ = KF2_R;
        float kf2OutX = kf2X.update(DX);
        DX = round1(kf2OutX);
        
        // Step 6: 精细调整
        if (absErrorX < 1.0f && std::abs(deltaErrorX) > 0.5f) {
            DX += round1(0.5f * lastOutputX + deltaErrorX);
        }
        
        // Step 7: kf3 卡尔曼滤波
        {
            float P_pred = kf3X_P + kf2X.x_;  // 与专业PID一致：直接使用kf2.x_
            float K = P_pred / (P_pred + KF2_R);
            float innov = DX - kf3X_x;
            kf3X_x = kf3X_x + K * innov;
            kf3X_P = (1.0f - K) * P_pred;
        }
        DX = round1(kf3X_x);
        
        // Step 8: D项最终处理
        if (std::abs(DX) <= 0.5f) {
            DX = 0.0f;
        }
        DX *= config.dTermScale * adaptiveIGainX;  // rate_ * integral_gain_
        
        if (config.maxPixelMove > 0.0f) {
            DX = atan2Clamp(DX, config.maxPixelMove, config.maxPixelMove);
        }
        
        // Step 9: 积分项（带限幅防止饱和）
        integralX += errorX_work * config.pidI * adaptiveIGainX * config.integralRate;
        // 应用积分限幅
        float iLimit = (config.integralLimit > 0.0f) ? config.integralLimit : 1000.0f;
        integralX = std::clamp(integralX, -iLimit, iLimit);
        float iOutX = integralX;
        
        if (config.maxPixelMove > 0.0f) {
            iOutX = atan2Clamp(iOutX, config.maxPixelMove, config.maxPixelMove);
        }
        
        // Step 10: P项
        float pOutX = round1(errorX_work * baseP);
        if (config.maxPixelMove > 0.0f) {
            pOutX = atan2Clamp(pOutX, config.maxPixelMove, config.maxPixelMove);
        }
        
        // Step 11: D2项
        float d2OutX = round1(deltaErrorX * config.pidD);
        if (config.maxPixelMove > 0.0f) {
            d2OutX = atan2Clamp(d2OutX, config.maxPixelMove, config.maxPixelMove);
        }
        
        // Step 12: 第二积分累加
        iOutX += d2OutX;
        
        if (std::abs(iOutX) <= DEAD_ZONE) {
            iOutX = 0.0f;
        }
        
        // Step 13: 总输出
        float totalX = round1(pOutX + iOutX + d2OutX);
        
        if (config.maxPixelMove > 0.0f) {
            totalX = atan2Clamp(totalX, config.maxPixelMove, config.maxPixelMove);
        }
        
        // Step 14: 比例增益权重调节
        totalX *= adaptivePGainX;
        totalX = round1(totalX);
        
        // Step 15: 保存状态
        pidPreviousErrorX = errorX_work;
        lastOutputX = totalX;
        
        moveX = totalX;
        
        // ========== Y轴处理（相同逻辑） ==========
        float absErrorY = std::abs(predictedErrorY);
        float errorY_work = predictedErrorY;
        
        if (absErrorY <= DEAD_ZONE) {
            errorY_work = 0.0f;
        }
        
        float deltaErrorY = errorY_work - pidPreviousErrorY;
        bool jumpDetectedY = std::abs(deltaErrorY) > JUMP_THRESHOLD;
        
        if (jumpDetectedY) {
            adaptivePGainY = 0.0f;
            adaptiveIGainY = 0.0f;
            integralY = 0.0f;
            lastOutputY = 0.0f;
            kf2Y.Q_ = 0.0f;
            kf2Y.R_ = 0.0f;
            kf3Y_x = 0.0f;
            kf3Y_P = 0.0f;
            pidPreviousErrorY = 0.0f;
            deltaErrorY = errorY_work;
        }
        
        {
            float ratio;
            if (absErrorY < INTEGRAL_GAIN_THRESHOLD) {
                ratio = 1.0f - (absErrorY / INTEGRAL_GAIN_THRESHOLD);
                adaptiveIGainY += (ratio - adaptiveIGainY) * INTEGRAL_GAIN_RATE;
            } else {
                ratio = INTEGRAL_GAIN_THRESHOLD / absErrorY;
                adaptiveIGainY += (ratio * adaptiveIGainY - adaptiveIGainY) * LARGE_ERROR_RATE;
            }
            adaptiveIGainY = std::clamp(adaptiveIGainY, 0.0f, 1.0f);
        }
        
        {
            float ratio;
            if (absErrorY < KP_GAIN_THRESHOLD) {
                ratio = 1.0f - (absErrorY / KP_GAIN_THRESHOLD);
                adaptivePGainY += (ratio - adaptivePGainY) * config.adaptivePGainRate;
            } else {
                ratio = KP_GAIN_THRESHOLD / absErrorY;
                adaptivePGainY += (ratio * adaptivePGainY - adaptivePGainY) * LARGE_ERROR_RATE;
            }
            adaptivePGainY = std::clamp(adaptivePGainY, 0.0f, 1.0f);
        }
        
        float DY = round1(deltaErrorY + lastOutputY);
        kf2Y.Q_ = KF2_Q;
        kf2Y.R_ = KF2_R;
        float kf2OutY = kf2Y.update(DY);
        DY = round1(kf2OutY);
        
        if (absErrorY < 1.0f && std::abs(deltaErrorY) > 0.5f) {
            DY += round1(0.5f * lastOutputY + deltaErrorY);
        }
        
        {
            float P_pred = kf3Y_P + kf2Y.x_;  // 与专业PID一致：直接使用kf2.x_
            float K = P_pred / (P_pred + KF2_R);
            float innov = DY - kf3Y_x;
            kf3Y_x = kf3Y_x + K * innov;
            kf3Y_P = (1.0f - K) * P_pred;
        }
        DY = round1(kf3Y_x);
        
        if (std::abs(DY) <= 0.5f) {
            DY = 0.0f;
        }
        DY *= config.dTermScale * adaptiveIGainY;
        
        if (config.maxPixelMove > 0.0f) {
            DY = atan2Clamp(DY, config.maxPixelMove, config.maxPixelMove);
        }
     // Step 9: 积分项（带限幅防止饱和）
        integralY += errorY_work * config.pidI * adaptiveIGainY * config.integralRate;
        // 应用积分限幅
        integralY = std::clamp(integralY, -iLimit, iLimit);
        float iOutY = integralY;
        
        if (config.maxPixelMove > 0.0f) {
            iOutY = atan2Clamp(iOutY, config.maxPixelMove, config.maxPixelMove);
        }
        
        float pOutY = round1(errorY_work * baseP);
        if (config.maxPixelMove > 0.0f) {
            pOutY = atan2Clamp(pOutY, config.maxPixelMove, config.maxPixelMove);
        }
        
        float d2OutY = round1(deltaErrorY * config.pidD);
        if (config.maxPixelMove > 0.0f) {
            d2OutY = atan2Clamp(d2OutY, config.maxPixelMove, config.maxPixelMove);
        }
        
        iOutY += d2OutY;
        
        if (std::abs(iOutY) <= DEAD_ZONE) {
            iOutY = 0.0f;
        }
        
        float totalY = round1(pOutY + iOutY + d2OutY);
        
        if (config.maxPixelMove > 0.0f) {
            totalY = atan2Clamp(totalY, config.maxPixelMove, config.maxPixelMove);
        }
        
        totalY *= adaptivePGainY;
        totalY = round1(totalY);
        
        pidPreviousErrorY = errorY_work;
        lastOutputY = totalY;
        
        moveY = totalY;
        
        static int logCounter = 0;
        if (++logCounter >= 30) {
            logCounter = 0;
            blog(LOG_INFO, "[%s高级PID] errorX=%.1f errorY=%.1f | kpGainX=%.2f kpGainY=%.2f | iGainX=%.2f iGainY=%.2f",
                 getLogPrefix(), errorX, errorY, adaptivePGainX, adaptivePGainY, adaptiveIGainX, adaptiveIGainY);
            blog(LOG_INFO, "[%s高级PID] pOutX=%.1f pOutY=%.1f | iOutX=%.1f iOutY=%.1f | d2OutX=%.1f d2OutY=%.1f",
                 getLogPrefix(), pOutX, pOutY, iOutX, iOutY, d2OutX, d2OutY);
            blog(LOG_INFO, "[%s高级PID] totalX=%.1f totalY=%.1f | kf2x=%.2f kf3x=%.2f",
                 getLogPrefix(), totalX, totalY, kf2OutX, kf3X_x);
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
            data.currentKp = baseP;
            data.currentKi = config.pidI;
            data.currentKd = config.pidD;
            data.pTermX = pOutX;
            data.pTermY = pOutY;
            data.iTermX = iOutX;
            data.iTermY = iOutY;
            data.dTermX = d2OutX;
            data.dTermY = d2OutY;

            float iLimit = (config.integralLimit > 0.0f) ? config.integralLimit : 1000.0f;
            data.integralAbsX = std::abs(integralX);
            data.integralAbsY = std::abs(integralY);
            data.integralLimitX = iLimit;
            data.integralLimitY = iLimit;
            data.integralRatioX = std::min(1.0f, std::abs(integralX) / iLimit);
            data.integralRatioY = std::min(1.0f, std::abs(integralY) / iLimit);

            float errDist = std::sqrt(errorX * errorX + errorY * errorY);
            float maxIRatio = std::max(data.integralRatioX, data.integralRatioY);
            bool hasTarget = (errDist > 0.5f || std::abs(targetVelocityX) > 0.1f || std::abs(targetVelocityY) > 0.1f);

            if (!hasTarget) {
                data.controlMode = 0;
            } else if (maxIRatio > 0.8f) {
                data.controlMode = 3;
            } else if (errDist < 10.0f && maxIRatio < 0.4f) {
                data.controlMode = 2;
            } else {
                data.controlMode = 1;
            }

            data.algorithmType = 0;
            data.isFiring = isFiring;
            pidDataCallback_(data);
        }

        pidPreviousErrorX = errorX;
        pidPreviousErrorY = errorY;
        previousErrorX = errorX;
        previousErrorY = errorY;
    } else if (config.algorithmType == AlgorithmType::ExternalPID) {
        // 外部PID库（pid_x64.lib）
        if (!externalPidInitialized_) {
            externalPidX.setName("ExternalPID_X");
            externalPidY.setName("ExternalPID_Y");
            externalPidX.init(config.externalKpX, config.externalKiX, config.externalKdX, config.externalPredictX, config.externalRateX);
            externalPidY.init(config.externalKpY, config.externalKiY, config.externalKdY, config.externalPredictY, config.externalRateY);
            externalPidX.setBase(config.externalKiMode, config.externalKpLimit, config.externalKiLimit, config.externalKdLimit, config.externalOutputLimit, config.externalKiRate, config.externalKiDeadband);
            externalPidY.setBase(config.externalKiMode, config.externalKpLimit, config.externalKiLimit, config.externalKdLimit, config.externalOutputLimit, config.externalKiRate, config.externalKiDeadband);
            externalPidInitialized_ = true;
        }

        float externalErrorX = errorX;
        float externalErrorY = errorY;

        if (config.useDerivativePredictor) {
            predictor.update(errorX, errorY, previousMoveX, previousMoveY, deltaTime);
            float derivPredictedX = 0.0f, derivPredictedY = 0.0f;
            predictor.predict(deltaTime, derivPredictedX, derivPredictedY);
            externalErrorX += config.predictionWeightX * derivPredictedX;
            externalErrorY += config.predictionWeightY * derivPredictedY;
        }

        moveX = static_cast<float>(externalPidX.update(externalErrorX));
        moveY = static_cast<float>(externalPidY.update(externalErrorY));

        static int externalLogCounter = 0;
        if (++externalLogCounter >= 30) {
            externalLogCounter = 0;
            blog(LOG_INFO, "[%s外部PID] dt=%.4f | errorX=%.1f errorY=%.1f | extErrX=%.1f extErrY=%.1f | moveX=%.1f moveY=%.1f",
                 getLogPrefix(), deltaTime, errorX, errorY, externalErrorX, externalErrorY, moveX, moveY);
            blog(LOG_INFO, "[%s外部PID] KpX=%.2f KiX=%.2f KdX=%.2f | KpY=%.2f KiY=%.2f KdY=%.2f",
                 getLogPrefix(), config.externalKpX, config.externalKiX, config.externalKdX, config.externalKpY, config.externalKiY, config.externalKdY);
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
            data.currentKp = config.externalKpX;
            data.currentKi = config.externalKiX;
            data.currentKd = config.externalKdX;
            data.algorithmType = 3;
            data.isFiring = isFiring;
            pidDataCallback_(data);
        }

        previousErrorX = errorX;
        previousErrorY = errorY;
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
            
            // 使用误差作为输入（目标相对于准心的偏移）
            float ghostOffsetX = 0.0f, ghostOffsetY = 0.0f;
            Detection* target = selectTarget();
            if (target) {
                int fw = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth :
                         ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
                int fh = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight :
                         ((config.sourceHeight > 0) ? config.sourceHeight : 1080);
                float targetW = target->width * fw;
                float targetH = target->height * fh;
                
                GhostTracker::Config ghostConfig;
                ghostConfig.enabled = true;
                ghostConfig.curvature = config.ghostCurvature;
                ghostConfig.noiseIntensity = config.ghostNoiseIntensity;
                ghostConfig.verticalSnapRatio = config.ghostVerticalSnapRatio;
                ghostConfig.noiseFreq = config.ghostNoiseFreq;
                ghostTracker.setConfig(ghostConfig);
                
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

    // 压枪补偿
    float finalMoveX = moveX;
    float finalMoveY = moveY;

    if (config.autoRecoilControlEnabled && firing) {
        float recoilPerMs = config.recoilStrength / static_cast<float>(config.recoilSpeed);
        float recoilThisFrame = recoilPerMs * deltaTime * 1000.0f;
        finalMoveY += recoilThisFrame;
    }

    // 时间相关移动：应用帧率补偿，确保移动速度不受帧率波动影响
    if (config.enableTimeBasedMovement && deltaTime > 0.0f) {
        float timeFactor = deltaTime * config.targetFrameRate;
        finalMoveX *= timeFactor;
        finalMoveY *= timeFactor;
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
