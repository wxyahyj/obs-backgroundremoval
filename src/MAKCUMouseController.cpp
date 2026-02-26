#ifdef _WIN32

#include "MAKCUMouseController.hpp"
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
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
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
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , previousMoveX(0.0f)
    , previousMoveY(0.0f)
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
    
    obs_log(LOG_INFO, "[MAKCU] Config updated: enableMouseControl=%d, autoTriggerEnabled=%d, fireDuration=%dms, interval=%dms, targetSwitchDelay=%dms, targetSwitchTolerance=%.2f",
            newConfig.enableMouseControl, newConfig.autoTriggerEnabled, newConfig.autoTriggerFireDuration, newConfig.autoTriggerInterval,
            newConfig.targetSwitchDelayMs, newConfig.targetSwitchTolerance);
    
    bool portChanged = (newConfig.makcuPort != portName);
    bool baudChanged = (newConfig.makcuBaudRate != baudRate);
    
    config = newConfig;
    
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
            obs_log(LOG_INFO, "[MAKCU-AutoTrigger] Releasing because enableMouseControl=false");
            releaseAutoTrigger();
        }
        autoTriggerWaitingForDelay = false;
        isMoving = false;
        return;
    }

    bool hotkeyPressed = (GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000) != 0;

    if (!hotkeyPressed) {
        if (autoTriggerHolding) {
            obs_log(LOG_INFO, "[MAKCU-AutoTrigger] Releasing because hotkey released");
        }
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
    }

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
            obs_log(LOG_INFO, "[MAKCU-AutoTrigger] Holding: fireElapsed=%lldms, currentFireDuration=%dms", 
                    fireElapsed, currentFireDuration);
            if (fireElapsed >= currentFireDuration) {
                releaseAutoTrigger();
                lastAutoTriggerTime = now;
                obs_log(LOG_INFO, "[MAKCU-AutoTrigger] Released after %lldms", fireElapsed);
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
                        obs_log(LOG_INFO, "[MAKCU-AutoTrigger] Firing: delay=%lldms, cooldown=%lldms, fireDuration=%dms", 
                                delayElapsed, cooldownElapsed, config.autoTriggerFireDuration);
                        performAutoClick();
                    }
                }
            } else {
                autoTriggerWaitingForDelay = false;
            }
        }
    }

    isMoving = true;
    
    float dynamicP = calculateDynamicP(distance);
    
    float deltaErrorX = errorX - pidPreviousErrorX;
    float deltaErrorY = errorY - pidPreviousErrorY;
    
    float alpha = config.derivativeFilterAlpha;
    filteredDeltaErrorX = alpha * deltaErrorX + (1.0f - alpha) * filteredDeltaErrorX;
    filteredDeltaErrorY = alpha * deltaErrorY + (1.0f - alpha) * filteredDeltaErrorY;
    
    float pdOutputX = dynamicP * errorX + config.pidD * filteredDeltaErrorX;
    float pdOutputY = dynamicP * errorY + config.pidD * filteredDeltaErrorY;
    
    float baselineX = errorX * config.baselineCompensation;
    float baselineY = errorY * config.baselineCompensation;
    
    float moveX = pdOutputX + baselineX;
    float moveY = pdOutputY + baselineY;
    
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
    
    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;
    
    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;
    
    if (serialConnected) {
        move(static_cast<int>(finalMoveX), static_cast<int>(finalMoveY));
    } else {
        connectSerial();
    }
    
    pidPreviousErrorX = errorX;
    pidPreviousErrorY = errorY;
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
    float minDistanceSquared = std::numeric_limits<float>::max();
    int bestTrackId = -1;

    for (auto& det : currentDetections) {
        int targetX = static_cast<int>(det.centerX * frameWidth);
        int targetY = static_cast<int>(det.centerY * frameHeight);
        
        float dx = static_cast<float>(targetX - fovCenterX);
        float dy = static_cast<float>(targetY - fovCenterY);
        float distanceSquared = dx * dx + dy * dy;

        if (distanceSquared <= fovRadiusSquared && distanceSquared < minDistanceSquared) {
            minDistanceSquared = distanceSquared;
            bestTarget = &det;
            bestTrackId = det.trackId;
        }
    }

    if (!bestTarget) {
        currentTargetTrackId = -1;
        currentTargetDistance = 0.0f;
        return nullptr;
    }

    float bestDistance = std::sqrt(minDistanceSquared);
    auto now = std::chrono::steady_clock::now();

    static bool loggedOnce = false;
    if (!loggedOnce) {
        obs_log(LOG_INFO, "[MAKCU-TargetSwitch] targetSwitchDelayMs=%dms, targetSwitchTolerance=%.2f", 
                config.targetSwitchDelayMs, config.targetSwitchTolerance);
        loggedOnce = true;
    }

    if (currentTargetTrackId == -1) {
        obs_log(LOG_INFO, "[MAKCU-TargetSwitch] First target: trackId=%d, distance=%.1f", bestTrackId, bestDistance);
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
    obs_log(LOG_INFO, "[MAKCU-TargetSwitch] New target found: currentTrackId=%d, newTrackId=%d, lockElapsed=%lldms, delay=%dms", 
            currentTargetTrackId, bestTrackId, lockElapsed, config.targetSwitchDelayMs);
    
    if (lockElapsed < config.targetSwitchDelayMs) {
        obs_log(LOG_INFO, "[MAKCU-TargetSwitch] Delaying switch, keeping current target");
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
        obs_log(LOG_INFO, "[MAKCU-TargetSwitch] Current target lost, switching to new");
        currentTargetTrackId = bestTrackId;
        targetLockStartTime = now;
        currentTargetDistance = bestDistance;
        return bestTarget;
    }

    if (currentTargetDistance > 0.0f && config.targetSwitchTolerance > 0.0f) {
        float improvement = (currentTargetDistance - bestDistance) / currentTargetDistance;
        obs_log(LOG_INFO, "[MAKCU-TargetSwitch] Tolerance check: improvement=%.2f, tolerance=%.2f", improvement, config.targetSwitchTolerance);
        if (improvement < config.targetSwitchTolerance) {
            obs_log(LOG_INFO, "[MAKCU-TargetSwitch] Improvement too small, keeping current target");
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

    obs_log(LOG_INFO, "[MAKCU-TargetSwitch] Switching to new target");
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

void MAKCUMouseController::resetPidState()
{
    pidPreviousErrorX = 0.0f;
    pidPreviousErrorY = 0.0f;
    filteredDeltaErrorX = 0.0f;
    filteredDeltaErrorY = 0.0f;
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
