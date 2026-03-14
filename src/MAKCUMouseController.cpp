#ifdef _WIN32

#include "MAKCUMouseController.hpp"
#include "DerivativePredictor.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <obs-module.h>
#include "plugin-support.h"

MAKCUMouseController::MAKCUMouseController()
    : cachedScreenWidth(0)
    , cachedScreenHeight(0)
    , hSerial(INVALID_HANDLE_VALUE)
    , serialConnected(false)
    , portName("COM5")
    , baudRate(4000000)
    , isMoving(false)
    , pidPreviousErrorX(0.0f)
    , pidPreviousErrorY(0.0f)
    , filteredDeltaErrorX(0.0f)
    , filteredDeltaErrorY(0.0f)
    , previousErrorX(0.0f)
    , previousErrorY(0.0f)
    , errorSignChangeCount(0)
    , lastSignChangeTime(std::chrono::steady_clock::now())
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
    , integralX(0.0f)
    , integralY(0.0f)
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
    cachedScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    cachedScreenHeight = GetSystemMetrics(SM_CYSCREEN);
    connectSerial();
    
    if (serialConnected) {
        move(0, 0);
    }
}

MAKCUMouseController::MAKCUMouseController(const std::string& port, int baud)
    : cachedScreenWidth(0)
    , cachedScreenHeight(0)
    , hSerial(INVALID_HANDLE_VALUE)
    , serialConnected(false)
    , portName(port)
    , baudRate(baud)
    , isMoving(false)
    , pidPreviousErrorX(0.0f)
    , pidPreviousErrorY(0.0f)
    , filteredDeltaErrorX(0.0f)
    , filteredDeltaErrorY(0.0f)
    , previousErrorX(0.0f)
    , previousErrorY(0.0f)
    , errorSignChangeCount(0)
    , lastSignChangeTime(std::chrono::steady_clock::now())
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
    , integralX(0.0f)
    , integralY(0.0f)
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
    cachedScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    cachedScreenHeight = GetSystemMetrics(SM_CYSCREEN);
    connectSerial();
    
    if (serialConnected) {
        move(0, 0);
    }
}

MAKCUMouseController::~MAKCUMouseController()
{
    disconnectSerial();
}

bool MAKCUMouseController::connectSerial()
{
    if (serialConnected) {
        return true;
    }

    std::wstring wPortName(portName.begin(), portName.end());
    hSerial = CreateFileW(
        wPortName.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hSerial == INVALID_HANDLE_VALUE) {
        return false;
    }

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);

    if (!GetCommState(hSerial, &dcbSerialParams)) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    dcbSerialParams.BaudRate = baudRate;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;

    if (!SetCommState(hSerial, &dcbSerialParams)) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    COMMTIMEOUTS timeouts = { 0 };
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;

    if (!SetCommTimeouts(hSerial, &timeouts)) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        return false;
    }

    serialConnected = true;
    return true;
}

void MAKCUMouseController::disconnectSerial()
{
    if (serialConnected && hSerial != INVALID_HANDLE_VALUE) {
        CloseHandle(hSerial);
        hSerial = INVALID_HANDLE_VALUE;
        serialConnected = false;
    }
}

bool MAKCUMouseController::sendSerialCommand(const std::string& command)
{
    if (!serialConnected || hSerial == INVALID_HANDLE_VALUE) {
        return false;
    }

    DWORD bytesWritten;
    std::string cmd = command + "\r\n";
    bool success = WriteFile(hSerial, cmd.c_str(), static_cast<DWORD>(cmd.length()), &bytesWritten, NULL);
    if (success && bytesWritten == static_cast<DWORD>(cmd.length())) {
        // 读取设备响应（可选）
        char buffer[256];
        DWORD bytesRead;
        DWORD events;
        if (WaitCommEvent(hSerial, &events, NULL)) {
            if (events & EV_RXCHAR) {
                ReadFile(hSerial, buffer, sizeof(buffer) - 1, &bytesRead, NULL);
            }
        }
        
        return true;
    } else {
        return false;
    }
}

void MAKCUMouseController::move(int dx, int dy)
{
    char cmd[64];
    sprintf_s(cmd, sizeof(cmd), "km.move(%d,%d)", dx, dy);
    sendSerialCommand(cmd);
}

void MAKCUMouseController::moveTo(int x, int y)
{
    char cmd[64];
    sprintf_s(cmd, sizeof(cmd), "km.moveTo(%d,%d)", x, y);
    sendSerialCommand(cmd);
}

void MAKCUMouseController::click(bool left)
{
    sendSerialCommand(left ? "km.left(1)" : "km.left(0)");
}

void MAKCUMouseController::clickDown()
{
    obs_log(LOG_INFO, "[MAKCU] clickDown: sending km.left(1)");
    bool success = sendSerialCommand("km.left(1)");
    obs_log(LOG_INFO, "[MAKCU] clickDown result: %d, serialConnected=%d", success, serialConnected);
}

void MAKCUMouseController::clickUp()
{
    obs_log(LOG_INFO, "[MAKCU] clickUp: sending km.left(0)");
    bool success = sendSerialCommand("km.left(0)");
    obs_log(LOG_INFO, "[MAKCU] clickUp result: %d, serialConnected=%d", success, serialConnected);
}

void MAKCUMouseController::wheel(int delta)
{
    char cmd[64];
    sprintf_s(cmd, sizeof(cmd), "km.wheel(%d)", delta);
    sendSerialCommand(cmd);
}

void MAKCUMouseController::updateConfig(const MouseControllerConfig& newConfig)
{
    std::lock_guard<std::mutex> lock(mutex);
    
    bool configChanged = (config.enableMouseControl != newConfig.enableMouseControl ||
                          config.autoTriggerEnabled != newConfig.autoTriggerEnabled ||
                          config.autoTriggerFireDuration != newConfig.autoTriggerFireDuration ||
                          config.autoTriggerInterval != newConfig.autoTriggerInterval ||
                          config.targetSwitchDelayMs != newConfig.targetSwitchDelayMs ||
                          config.targetSwitchTolerance != newConfig.targetSwitchTolerance);
    
    bool portChanged = (newConfig.makcuPort != portName);
    bool baudChanged = (newConfig.makcuBaudRate != baudRate);
    
    config = newConfig;
    
    if (configChanged) {
        obs_log(LOG_INFO, "[MAKCU] Config updated: enableMouseControl=%d, autoTriggerEnabled=%d, fireDuration=%dms, interval=%dms, targetSwitchDelay=%dms, targetSwitchTolerance=%.2f",
                newConfig.enableMouseControl, newConfig.autoTriggerEnabled, newConfig.autoTriggerFireDuration, newConfig.autoTriggerInterval,
                newConfig.targetSwitchDelayMs, newConfig.targetSwitchTolerance);
    }
    
    if (portChanged || baudChanged) {
        portName = newConfig.makcuPort;
        baudRate = newConfig.makcuBaudRate;
        
        disconnectSerial();
        
        connectSerial();
    }
}

void MAKCUMouseController::setDetections(const std::vector<Detection>& detections)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
}

void MAKCUMouseController::setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
    config.inferenceFrameWidth = frameWidth;
    config.inferenceFrameHeight = frameHeight;
    config.cropOffsetX = cropX;
    config.cropOffsetY = cropY;
}

void MAKCUMouseController::tick()
{
    std::lock_guard<std::mutex> lock(mutex);

    if (!config.enableMouseControl) {
        if (autoTriggerHolding) {
            releaseAutoTrigger();
        }
        autoTriggerWaitingForDelay = false;
        isMoving = false;
        return;
    }

    bool hotkeyPressed = (GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000) != 0;

    if (!hotkeyPressed) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        yUnlockActive = false;
        releaseAutoTrigger();
        return;
    }

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
    
    float moveX = pidOutputX + baselineX;
    float moveY = pidOutputY + baselineY;
    
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
    
    // 平滑处理
    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;
    
    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;
    
    if (serialConnected) {
        move(static_cast<int>(finalMoveX), static_cast<int>(finalMoveY));
    } else {
        connectSerial();
    }
    
    // 更新状态
    pidPreviousErrorX = fusedErrorX;
    pidPreviousErrorY = fusedErrorY;
    previousErrorX = errorX;
    previousErrorY = errorY;
}

Detection* MAKCUMouseController::selectTarget()
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

POINT MAKCUMouseController::convertToScreenCoordinates(const Detection& det)
{
    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth : 
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight : 
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    float screenPixelX = det.centerX * frameWidth + config.screenOffsetX;
    float screenPixelY = det.centerY * frameHeight - config.targetYOffset + config.screenOffsetY;

    static bool loggedOnce = false;
    if (!loggedOnce) {
        obs_log(LOG_INFO, "[MAKCU] 坐标转换调试信息:");
        obs_log(LOG_INFO, "[MAKCU]   屏幕尺寸: %dx%d", cachedScreenWidth, cachedScreenHeight);
        obs_log(LOG_INFO, "[MAKCU]   推理帧尺寸: %dx%d", frameWidth, frameHeight);
        obs_log(LOG_INFO, "[MAKCU]   检测中心(归一化): %.4f, %.4f", det.centerX, det.centerY);
        obs_log(LOG_INFO, "[MAKCU]   屏幕偏移: %d, %d", config.screenOffsetX, config.screenOffsetY);
        obs_log(LOG_INFO, "[MAKCU]   最终屏幕坐标: %.1f, %.1f", screenPixelX, screenPixelY);
        loggedOnce = true;
    }

    POINT result;
    result.x = static_cast<LONG>(screenPixelX);
    result.y = static_cast<LONG>(screenPixelY);

    LONG maxX = static_cast<LONG>(cachedScreenWidth - 1);
    LONG maxY = static_cast<LONG>(cachedScreenHeight - 1);
    
    result.x = std::max(0L, std::min(result.x, maxX));
    result.y = std::max(0L, std::min(result.y, maxY));

    return result;
}

float MAKCUMouseController::calculateDynamicP(float distance)
{
    float normalizedDistance = distance / static_cast<float>(config.fovRadiusPixels);
    normalizedDistance = std::max(0.0f, std::min(1.0f, normalizedDistance));
    float distancePower = std::pow(normalizedDistance, config.pidPSlope);
    float p = config.pidPMin + (config.pidPMax - config.pidPMin) * distancePower;
    return std::max(config.pidPMin, std::min(config.pidPMax, p));
}

float MAKCUMouseController::calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor)
{
    if (!config.adaptiveDEnabled) {
        adaptiveFactor = 1.0f;
        return config.pidD;
    }
    
    float baseD = config.pidD;
    
    float jitterFactor = std::abs(deltaError) / config.dJitterThreshold;
    jitterFactor = std::min(jitterFactor, 1.0f);
    
    float distanceFactor = 1.0f - std::min(distance / 200.0f, 1.0f);
    
    float overshootFactor = 0.0f;
    auto now = std::chrono::steady_clock::now();
    
    if ((error > 0 && previousErrorX < 0) || (error < 0 && previousErrorX > 0)) {
        auto timeSinceLastChange = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lastSignChangeTime).count();
        
        if (timeSinceLastChange < 500) {
            errorSignChangeCount++;
        } else {
            errorSignChangeCount = 1;
        }
        lastSignChangeTime = now;
        
        if (errorSignChangeCount >= 3) {
            overshootFactor = 1.0f;
        } else if (errorSignChangeCount >= 2) {
            overshootFactor = 0.5f;
        }
    }
    
    adaptiveFactor = (1.0f - config.dAdaptiveStrength) + 
                     config.dAdaptiveStrength * (jitterFactor * 0.5f + 
                                               distanceFactor * 0.3f + 
                                               overshootFactor * 0.2f);
    
    float finalD = baseD * adaptiveFactor;
    finalD = std::max(config.pidDMin, std::min(finalD, config.pidDMax));
    
    static int logCounter = 0;
    if (logCounter++ % 100 == 0) {
        obs_log(LOG_INFO, "[MAKCU-AdaptiveD] jitter=%.2f, dist=%.2f, over=%.2f, factor=%.3f, baseD=%.4f, finalD=%.4f",
                jitterFactor, distanceFactor, overshootFactor, adaptiveFactor, baseD, finalD);
    }
    
    return finalD;
}

float MAKCUMouseController::calculateIntegral(float error, float& integral, float deltaTime)
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

float MAKCUMouseController::getCurrentPGain()
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - targetLockStartTime).count() / 1000.0f;
    
    float rampFactor = std::min(elapsed / config.pGainRampDuration, 1.0f);
    float currentScale = config.pGainRampInitialScale + (1.0f - config.pGainRampInitialScale) * rampFactor;
    
    return currentScale;
}

void MAKCUMouseController::resetPidState()
{
    pidPreviousErrorX = 0.0f;
    pidPreviousErrorY = 0.0f;
    filteredDeltaErrorX = 0.0f;
    filteredDeltaErrorY = 0.0f;
    integralX = 0.0f;
    integralY = 0.0f;
    predictor.reset();
}

void MAKCUMouseController::resetMotionState()
{
    currentVelocityX = 0.0f;
    currentVelocityY = 0.0f;
    currentAccelerationX = 0.0f;
    currentAccelerationY = 0.0f;
    previousMoveX = 0.0f;
    previousMoveY = 0.0f;
}

int MAKCUMouseController::getRandomDelay()
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

int MAKCUMouseController::getRandomDuration()
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

void MAKCUMouseController::performAutoClick()
{
    clickDown();
    autoTriggerHolding = true;
    autoTriggerFireStartTime = std::chrono::steady_clock::now();
    currentFireDuration = config.autoTriggerFireDuration + getRandomDuration();
}

void MAKCUMouseController::releaseAutoTrigger()
{
    obs_log(LOG_INFO, "[MAKCU-AutoTrigger] releaseAutoTrigger called, holding=%d", autoTriggerHolding);
    if (autoTriggerHolding) {
        obs_log(LOG_INFO, "[MAKCU-AutoTrigger] Sending km.left(0) to release");
        clickUp();
        autoTriggerHolding = false;
        obs_log(LOG_INFO, "[MAKCU-AutoTrigger] Released successfully");
    }
    autoTriggerWaitingForDelay = false;
}

bool MAKCUMouseController::isConnected()
{
    return serialConnected;
}

bool MAKCUMouseController::testCommunication()
{
    if (!serialConnected || hSerial == INVALID_HANDLE_VALUE) {
        return false;
    }

    std::string testCommand = "km.echo(1)";
    bool success = sendSerialCommand(testCommand);
    
    return success;
}

void MAKCUMouseController::setCurrentWeapon(const std::string& weaponName)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentWeapon_ = weaponName;
}

std::string MAKCUMouseController::getCurrentWeapon() const
{
    std::lock_guard<std::mutex> lock(mutex);
    return currentWeapon_;
}

#endif
