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
    , yUnlockActive(false)
    , lastAutoTriggerTime(std::chrono::steady_clock::now())
{
    hotkeyPressStartTime = std::chrono::steady_clock::now();
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
    , yUnlockActive(false)
    , lastAutoTriggerTime(std::chrono::steady_clock::now())
{
    hotkeyPressStartTime = std::chrono::steady_clock::now();
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
    sendSerialCommand(left ? "km.left(1)" : "km.right(1)");
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
    
    // 检查MAKCU配置是否变化
    bool portChanged = (newConfig.makcuPort != portName);
    bool baudChanged = (newConfig.makcuBaudRate != baudRate);
    
    config = newConfig;
    
    // 如果配置变化，重新连接串口
    if (portChanged || baudChanged) {
        portName = newConfig.makcuPort;
        baudRate = newConfig.makcuBaudRate;
        
        // 先断开旧连接
        disconnectSerial();
        
        // 重新连接
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
        return;
    }

    float distance = std::sqrt(distanceSquared);

    if (config.autoTriggerEnabled) {
        if (distance < config.autoTriggerRadius) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastAutoTriggerTime).count();
            if (elapsed >= config.autoTriggerCooldownMs) {
                click(true);
                lastAutoTriggerTime = now;
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
        return nullptr;
    }

    // 使用推理帧的实际尺寸来计算 FOV 中心
    // 这确保了 FOV 计算与检测坐标使用相同的基准
    int frameWidth = (config.inferenceFrameWidth > 0) ? config.inferenceFrameWidth : 
                     ((config.sourceWidth > 0) ? config.sourceWidth : 1920);
    int frameHeight = (config.inferenceFrameHeight > 0) ? config.inferenceFrameHeight : 
                      ((config.sourceHeight > 0) ? config.sourceHeight : 1080);

    int fovCenterX = frameWidth / 2;
    int fovCenterY = frameHeight / 2;
    float fovRadiusSquared = static_cast<float>(config.fovRadiusPixels * config.fovRadiusPixels);

    Detection* bestTarget = nullptr;
    float minDistanceSquared = std::numeric_limits<float>::max();

    for (auto& det : currentDetections) {
        // 检测坐标已经是相对于推理帧的归一化坐标
        // 所以我们需要使用推理帧的尺寸来计算像素位置
        int targetX = static_cast<int>(det.centerX * frameWidth);
        int targetY = static_cast<int>(det.centerY * frameHeight);
        
        float dx = static_cast<float>(targetX - fovCenterX);
        float dy = static_cast<float>(targetY - fovCenterY);
        float distanceSquared = dx * dx + dy * dy;

        if (distanceSquared <= fovRadiusSquared && distanceSquared < minDistanceSquared) {
            minDistanceSquared = distanceSquared;
            bestTarget = &det;
        }
    }

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
