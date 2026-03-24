#ifdef _WIN32

#include "MouseController.hpp"
#include "DerivativePredictor.hpp"
#include <obs-module.h>
#include <plugin-support.h>
#include <cmath>
#include <algorithm>
#include <random>

MouseController::MouseController()
    : cachedScreenWidth(0)
    , cachedScreenHeight(0)
    , isMoving(false)
    , pidPreviousErrorX(0.0f)
    , pidPreviousErrorY(0.0f)
    , filteredDeltaErrorX(0.0f)
    , filteredDeltaErrorY(0.0f)
    , previousErrorX(0.0f)
    , previousErrorY(0.0f)
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
    , integralX(0.0f)
    , integralY(0.0f)
    , stdIntegralX(0.0f)
    , stdIntegralY(0.0f)
    , stdIntegralGainX(0.0f)
    , stdIntegralGainY(0.0f)
    , stdLastErrorX(0.0f)
    , stdLastErrorY(0.0f)
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
{
    startPos = { 0, 0 };
    targetPos = { 0, 0 };
    cachedScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    cachedScreenHeight = GetSystemMetrics(SM_CYSCREEN);
}

MouseController::~MouseController()
{
}

void MouseController::updateConfig(const MouseControllerConfig& newConfig)
{
    std::lock_guard<std::mutex> lock(mutex);
    bool configChanged = (config.enableMouseControl != newConfig.enableMouseControl ||
                          config.autoTriggerEnabled != newConfig.autoTriggerEnabled ||
                          config.autoTriggerFireDuration != newConfig.autoTriggerFireDuration ||
                          config.autoTriggerInterval != newConfig.autoTriggerInterval);
    config = newConfig;
    if (configChanged) {
        obs_log(LOG_INFO, "[MouseController] Config updated: enableMouseControl=%d, autoTriggerEnabled=%d, fireDuration=%dms, interval=%dms",
                config.enableMouseControl, config.autoTriggerEnabled, config.autoTriggerFireDuration, config.autoTriggerInterval);
    }
}

void MouseController::setDetections(const std::vector<Detection>& detections)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
}

void MouseController::setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
    config.inferenceFrameWidth = frameWidth;
    config.inferenceFrameHeight = frameHeight;
    config.cropOffsetX = cropX;
    config.cropOffsetY = cropY;
}

void MouseController::tick()
{
    std::lock_guard<std::mutex> lock(mutex);

    if (!config.enableMouseControl) {
        if (autoTriggerHolding) {
            INPUT input = {};
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
            autoTriggerHolding = false;
        }
        autoTriggerWaitingForDelay = false;
        isMoving = false;
        return;
    }

    bool hotkeyPressed = (GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000) != 0;

    // 判断是否应该瞄准
    bool shouldAim = config.continuousAimEnabled || hotkeyPressed;

    // 如果不应该瞄准，停止所有操作
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

    // 更新热键状态和Y轴解锁（仅在非持续自瞄模式下）
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
        // 持续自瞄模式下，禁用Y轴解锁
        yUnlockActive = false;
    }

    // 计算时间步长
    auto now = std::chrono::steady_clock::now();
    deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTickTime).count() / 1000.0f;
    deltaTime = std::max(0.001f, std::min(deltaTime, 0.05f));
    lastTickTime = now;

    Detection* target = selectTarget();
    if (!target) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        if (autoTriggerHolding) {
            auto now = std::chrono::steady_clock::now();
            auto fireElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - autoTriggerFireStartTime).count();
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
            }
        }
        return;
    }

    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth : 
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight : 
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    float fovCenterX = frameWidth / 2.0f;
    float fovCenterY = frameHeight / 2.0f;

    float targetPixelX = target->centerX * frameWidth;
    float targetPixelY = target->centerY * frameHeight - config.targetYOffset;

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
            auto now = std::chrono::steady_clock::now();
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
        auto now = std::chrono::steady_clock::now();

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
        // 标准PID算法
        moveX = calculateStandardPID(errorX, stdIntegralX, stdIntegralGainX, stdLastErrorX, deltaTime);
        moveY = calculateStandardPID(errorY, stdIntegralY, stdIntegralGainY, stdLastErrorY, deltaTime);
        
        // 重置高级PID状态变量，避免算法切换时状态不一致
        pidPreviousErrorX = 0.0f;
        pidPreviousErrorY = 0.0f;
        previousErrorX = 0.0f;
        previousErrorY = 0.0f;
        filteredDeltaErrorX = 0.0f;
        filteredDeltaErrorY = 0.0f;
        integralX = 0.0f;
        integralY = 0.0f;
    } else {
        // 高级PID算法（默认）
        // 更新运动预测器
        predictor.update(errorX, errorY, deltaTime);

        // 预测目标位置
        float predictedX, predictedY;
        predictor.predict(deltaTime, predictedX, predictedY);

        // 误差融合
        float fusedErrorX = errorX + config.predictionWeightX * predictedX;
        float fusedErrorY = errorY + config.predictionWeightY * predictedY;

        // 计算动态P增益，应用P-Gain Ramp
        float dynamicP = calculateDynamicP(distance) * getCurrentPGain();

        // 计算微分项
        float deltaErrorX = fusedErrorX - pidPreviousErrorX;
        float deltaErrorY = fusedErrorY - pidPreviousErrorY;

        float alpha = config.derivativeFilterAlpha;
        filteredDeltaErrorX = alpha * deltaErrorX + (1.0f - alpha) * filteredDeltaErrorX;
        filteredDeltaErrorY = alpha * deltaErrorY + (1.0f - alpha) * filteredDeltaErrorY;

        // 计算自适应D增益
        float adaptiveFactorX = 1.0f;
        float adaptiveFactorY = 1.0f;
        float adaptiveDX = calculateAdaptiveD(distance, deltaErrorX, fusedErrorX, adaptiveFactorX);
        float adaptiveDY = calculateAdaptiveD(distance, deltaErrorY, fusedErrorY, adaptiveFactorY);

        // 计算积分项
        float integralTermX = calculateIntegral(fusedErrorX, integralX, deltaTime);
        float integralTermY = calculateIntegral(fusedErrorY, integralY, deltaTime);

        // PID输出计算
        float pidOutputX = dynamicP * fusedErrorX + adaptiveDX * filteredDeltaErrorX + integralTermX;
        float pidOutputY = dynamicP * fusedErrorY + adaptiveDY * filteredDeltaErrorY + integralTermY;

        float baselineX = fusedErrorX * config.baselineCompensation;
        float baselineY = fusedErrorY * config.baselineCompensation;

        moveX = pidOutputX + baselineX;
        moveY = pidOutputY + baselineY;

        // 更新高级PID状态
        pidPreviousErrorX = fusedErrorX;
        pidPreviousErrorY = fusedErrorY;
        previousErrorX = errorX;
        previousErrorY = errorY;
        
        // 重置标准PID状态变量，避免算法切换时状态不一致
        stdIntegralX = 0.0f;
        stdIntegralY = 0.0f;
        stdIntegralGainX = 0.0f;
        stdIntegralGainY = 1.0f;
        stdLastErrorX = errorX;
        stdLastErrorY = errorY;
    }
    
    // 检测射击状态（鼠标左键按下）
    bool isFiring = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    
    // 按下左键时降低Y轴PID增益，避免与压枪对抗（压枪功能独立，不需要自瞄触发）
    if (isFiring && config.autoRecoilControlEnabled) {
        moveY *= config.recoilPidGainScale;  // 使用可配置的增益系数
    }
    
    // 限制最大移动量
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
    
    // 自动压枪逻辑：独立功能，只要开启压枪且按下左键就压枪，不需要目标识别
    if (config.autoRecoilControlEnabled && isFiring) {
        // 计算每帧应该压枪的量（基于压枪速度和强度）
        // recoilSpeed是毫秒间隔，recoilStrength是该间隔内的压枪量
        float recoilPerMs = config.recoilStrength / static_cast<float>(config.recoilSpeed);
        float recoilThisFrame = recoilPerMs * deltaTime * 1000.0f;  // deltaTime转毫秒
        
        // 累积到moveY
        moveY += recoilThisFrame;
    }

    // 平滑处理
    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;
    
    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;
    
    // 执行鼠标移动
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dx = static_cast<LONG>(finalMoveX);
    input.mi.dy = static_cast<LONG>(finalMoveY);
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.time = 0;
    input.mi.dwExtraInfo = 0;
    SendInput(1, &input, sizeof(INPUT));
}

Detection* MouseController::selectTarget()
{
    if (currentDetections.empty()) {
        currentTargetTrackId = -1;
        currentTargetDistance = 0.0f;
        return nullptr;
    }

    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth : 
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight : 
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    int fovCenterX = frameWidth / 2;
    int fovCenterY = frameHeight / 2;
    float fovRadiusSquared = static_cast<float>(config.fovRadiusPixels * config.fovRadiusPixels);

    Detection* bestTarget = nullptr;
    float bestScore = -1.0f;
    int bestTrackId = -1;
    float bestDistance = std::numeric_limits<float>::max();

    for (auto& det : currentDetections) {
        int targetX = static_cast<int>(det.centerX * frameWidth);
        int targetY = static_cast<int>(det.centerY * frameHeight);
        
        float dx = static_cast<float>(targetX - fovCenterX);
        float dy = static_cast<float>(targetY - fovCenterY);
        float distanceSquared = dx * dx + dy * dy;

        if (distanceSquared <= fovRadiusSquared) {
            float distance = std::sqrt(distanceSquared);
            
            // 计算目标评分，考虑距离、大小和置信度
            float sizeScore = std::min(det.width * det.height, 0.1f); // 目标大小
            float confidenceScore = det.confidence; // 置信度
            float distanceScore = 1.0f / (1.0f + distance); // 距离越近分数越高
            
            // 综合评分 - 增加距离因素权重
            float score = 0.7f * distanceScore + 0.2f * confidenceScore + 0.1f * sizeScore;
            
            if (score > bestScore) {
                bestScore = score;
                bestTarget = &det;
                bestTrackId = det.trackId;
                bestDistance = distance;
            }
        }
    }

    if (!bestTarget) {
        currentTargetTrackId = -1;
        currentTargetDistance = 0.0f;
        return nullptr;
    }

    auto now = std::chrono::steady_clock::now();

    if (currentTargetTrackId == -1) {
        currentTargetTrackId = bestTrackId;
        targetLockStartTime = now;
        currentTargetDistance = bestDistance;
        return bestTarget;
    }

    if (bestTrackId == currentTargetTrackId) {
        currentTargetDistance = bestDistance;
        return bestTarget;
    }

    auto lockElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetLockStartTime).count();
    
    if (lockElapsed < config.targetSwitchDelayMs) {
        // 检查当前目标是否在FOV内
        bool currentTargetInFOV = false;
        for (auto& det : currentDetections) {
            if (det.trackId == currentTargetTrackId) {
                int targetX = static_cast<int>(det.centerX * frameWidth);
                int targetY = static_cast<int>(det.centerY * frameHeight);
                float dx = static_cast<float>(targetX - fovCenterX);
                float dy = static_cast<float>(targetY - fovCenterY);
                float distSq = dx * dx + dy * dy;
                if (distSq <= fovRadiusSquared) {
                    currentTargetInFOV = true;
                    return &det;
                }
            }
        }
        
        // 如果当前目标不在FOV内，仍然等待延迟时间后再切换
        if (lockElapsed < config.targetSwitchDelayMs) {
            // 延迟时间未到，返回nullptr，保持当前目标的锁定状态
            return nullptr;
        } else {
            // 延迟时间已过，切换到新目标
            currentTargetTrackId = bestTrackId;
            targetLockStartTime = now;
            currentTargetDistance = bestDistance;
            return bestTarget;
        }
    }

    if (currentTargetDistance > 0.0f && config.targetSwitchTolerance > 0.0f) {
        float improvement = (currentTargetDistance - bestDistance) / currentTargetDistance;
        if (improvement < config.targetSwitchTolerance) {
            for (auto& det : currentDetections) {
                if (det.trackId == currentTargetTrackId) {
                    int targetX = static_cast<int>(det.centerX * frameWidth);
                    int targetY = static_cast<int>(det.centerY * frameHeight);
                    float dx = static_cast<float>(targetX - fovCenterX);
                    float dy = static_cast<float>(targetY - fovCenterY);
                    float distSq = dx * dx + dy * dy;
                    if (distSq <= fovRadiusSquared) {
                        return &det;
                    }
                }
            }
        }
    }

    currentTargetTrackId = bestTrackId;
    targetLockStartTime = now;
    currentTargetDistance = bestDistance;
    return bestTarget;
}

POINT MouseController::convertToScreenCoordinates(const Detection& det)
{
    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth : 
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight : 
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    float screenPixelX = det.centerX * frameWidth + config.screenOffsetX;
    float screenPixelY = det.centerY * frameHeight - config.targetYOffset + config.screenOffsetY;

    POINT result;
    result.x = static_cast<LONG>(screenPixelX);
    result.y = static_cast<LONG>(screenPixelY);

    LONG maxX = static_cast<LONG>(cachedScreenWidth - 1);
    LONG maxY = static_cast<LONG>(cachedScreenHeight - 1);
    
    result.x = std::max(0L, std::min(result.x, maxX));
    result.y = std::max(0L, std::min(result.y, maxY));

    return result;
}

void MouseController::moveMouseTo(const POINT& pos)
{
    POINT currentPos;
    GetCursorPos(&currentPos);
    
    long deltaX = pos.x - currentPos.x;
    long deltaY = pos.y - currentPos.y;

    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dx = deltaX;
    input.mi.dy = deltaY;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.time = 0;
    input.mi.dwExtraInfo = 0;

    SendInput(1, &input, sizeof(INPUT));
}

void MouseController::startMouseMovement(const POINT& target)
{
    GetCursorPos(&startPos);
    targetPos = target;
    isMoving = true;
    resetPidState();
    resetMotionState();
}

float MouseController::calculateIntegral(float error, float& integral, float deltaTime)
{
    if (std::abs(error) < config.integralDeadZone) {
        return 0.0f;
    }
    
    if (std::abs(error) > config.integralSeparationThreshold) {
        return 0.0f;
    }
    
    integral += error * deltaTime;
    integral = std::max(-config.integralLimit, std::min(integral, config.integralLimit));
    
    return integral;
}

float MouseController::getCurrentPGain()
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetLockStartTime).count() / 1000.0f;
    
    float rampFactor = std::min(elapsed / config.pGainRampDuration, 1.0f);
    float currentScale = config.pGainRampInitialScale + (1.0f - config.pGainRampInitialScale) * rampFactor;
    
    return currentScale;
}

void MouseController::resetPidState()
{
    pidPreviousErrorX = 0.0f;
    pidPreviousErrorY = 0.0f;
    filteredDeltaErrorX = 0.0f;
    filteredDeltaErrorY = 0.0f;
    integralX = 0.0f;
    integralY = 0.0f;
    predictor.reset();
    // 重置标准PID状态
    stdIntegralX = 0.0f;
    stdIntegralY = 0.0f;
    stdIntegralGainX = 0.0f;
    stdIntegralGainY = 1.0f;
    stdLastErrorX = 0.0f;
    stdLastErrorY = 0.0f;
}

float MouseController::calculateDynamicP(float distance)
{
    float normalizedDistance = distance / static_cast<float>(config.fovRadiusPixels);
    normalizedDistance = std::max(0.0f, std::min(1.0f, normalizedDistance));
    float distancePower = std::pow(normalizedDistance, config.pidPSlope);
    float p = config.pidPMin + (config.pidPMax - config.pidPMin) * distancePower;
    return std::max(config.pidPMin, std::min(config.pidPMax, p));
}

float MouseController::calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor)
{
    UNUSED_PARAMETER(distance);
    UNUSED_PARAMETER(deltaError);
    UNUSED_PARAMETER(error);
    adaptiveFactor = 1.0f;
    return config.pidD;
}

// 标准PID计算函数
float MouseController::calculateStandardPID(float error, float& integral, float& integralGain, float& lastError, float deltaTime)
{
    // 死区处理
    if (std::abs(error) < config.stdDeadZone) {
        error = 0.0f;
    }

    // 比例计算
    float kp = config.stdKp * error;

    // 积分计算配合积分分离
    if (adjustStandardIntegral(error, lastError, integralGain))
    {
        // 积分累加
        integral += error;

        // 积分限幅
        integral = std::clamp(integral, -config.stdIntegralLimit, config.stdIntegralLimit);
    }
    else
    {
        // 积分清零
        integral = 0;
    }

    // 积分计算 + 死区处理
    float ki = (std::abs(integral) > config.stdIntegralDeadzone) ? config.stdKi * integral : 0;

    // 微分计算
    float kd = config.stdKd * (error - lastError);

    // 总输出计算
    float output = kp + ki + kd;

    // 总输出限幅
    output = std::clamp(output, -config.stdOutputLimit, config.stdOutputLimit);

    // 状态更新
    lastError = error;

    // 返回输出
    return output;
}

bool MouseController::adjustStandardIntegral(float error, float lastError, float& integralGain)
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
    return integralGain > 0.01f;  // 只有当积分增益大于0.01时才进行积分
}

void MouseController::resetMotionState()
{
    currentVelocityX = 0.0f;
    currentVelocityY = 0.0f;
    currentAccelerationX = 0.0f;
    currentAccelerationY = 0.0f;
    previousMoveX = 0.0f;
    previousMoveY = 0.0f;
}

int MouseController::getRandomDelay()
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

int MouseController::getRandomDuration()
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

void MouseController::performAutoClick()
{
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));
    autoTriggerHolding = true;
    isFiring = true;
    autoTriggerFireStartTime = std::chrono::steady_clock::now();
    currentFireDuration = config.autoTriggerFireDuration + getRandomDuration();
}

void MouseController::releaseAutoTrigger()
{
    if (autoTriggerHolding) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
        autoTriggerHolding = false;
    }
    isFiring = false;
    autoTriggerWaitingForDelay = false;
}

void MouseController::setCurrentWeapon(const std::string& weaponName)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentWeapon = weaponName;
}

std::string MouseController::getCurrentWeapon() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return currentWeapon;
}

#endif
