#ifdef _WIN32

#include "MAKCUMouseController.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include "plugin-support.h"

MAKCUMouseController::MAKCUMouseController()
    : hSerial(INVALID_HANDLE_VALUE)
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
{
    connectSerial();
    
    // 初始化MAKCU板子状态
    if (serialConnected) {
        // 发送一个小幅度的移动命令来重置板子状态
        move(0, 0);
    }
}

MAKCUMouseController::MAKCUMouseController(const std::string& port, int baud)
    : hSerial(INVALID_HANDLE_VALUE)
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
{
    connectSerial();
    
    // 初始化MAKCU板子状态
    if (serialConnected) {
        // 发送一个小幅度的移动命令来重置板子状态
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
    sprintf_s(cmd, sizeof(cmd), "km.move(%d,%d)", x, y);
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

void MAKCUMouseController::tick()
{
    std::lock_guard<std::mutex> lock(mutex);

    if (!config.enableMouseControl) {
        return;
    }

    if (!(GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000)) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        return;
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

    POINT targetScreenPos = convertToScreenCoordinates(*target);
    
    POINT currentPos;
    GetCursorPos(&currentPos);
    
    float errorX = static_cast<float>(targetScreenPos.x - currentPos.x);
    float errorY = static_cast<float>(targetScreenPos.y - currentPos.y);
    
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

    isMoving = true;
    
    float distance = std::sqrt(distanceSquared);
    
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
    
    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;
    
    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;
    
    // 检查连接状态后再发送命令
    if (serialConnected) {
        move(static_cast<int>(finalMoveX), static_cast<int>(finalMoveY));
    } else {
        // 尝试重新连接
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

    // 安全检查：确保sourceWidth和sourceHeight不为0
    int safeSourceWidth = (config.sourceWidth > 0) ? config.sourceWidth : 1920;
    int safeSourceHeight = (config.sourceHeight > 0) ? config.sourceHeight : 1080;

    int fovCenterX = safeSourceWidth / 2;
    int fovCenterY = safeSourceHeight / 2;
    float fovRadiusSquared = static_cast<float>(config.fovRadiusPixels * config.fovRadiusPixels);

    Detection* bestTarget = nullptr;
    float minDistanceSquared = std::numeric_limits<float>::max();

    for (auto& det : currentDetections) {
        int targetX = static_cast<int>(det.centerX * safeSourceWidth);
        int targetY = static_cast<int>(det.centerY * safeSourceHeight);
        
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
    int fullScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    int fullScreenHeight = GetSystemMetrics(SM_CYSCREEN);

    // 安全检查：确保sourceWidth和sourceHeight不为0
    int safeSourceWidth = (config.sourceWidth > 0) ? config.sourceWidth : fullScreenWidth;
    int safeSourceHeight = (config.sourceHeight > 0) ? config.sourceHeight : fullScreenHeight;

    float sourcePixelX = det.centerX * safeSourceWidth;
    float sourcePixelY = det.centerY * safeSourceHeight - config.targetYOffset;

    // 当screenWidth或screenHeight为0时，使用实际屏幕分辨率作为目标
    int targetScreenWidth = (config.screenWidth > 0) ? config.screenWidth : fullScreenWidth;
    int targetScreenHeight = (config.screenHeight > 0) ? config.screenHeight : fullScreenHeight;

    float screenScaleX = (float)targetScreenWidth / safeSourceWidth;
    float screenScaleY = (float)targetScreenHeight / safeSourceHeight;

    float screenPixelX = config.screenOffsetX + sourcePixelX * screenScaleX;
    float screenPixelY = config.screenOffsetY + sourcePixelY * screenScaleY;

    POINT result;
    result.x = static_cast<LONG>(screenPixelX);
    result.y = static_cast<LONG>(screenPixelY);

    LONG maxX = static_cast<LONG>(fullScreenWidth - 1);
    LONG maxY = static_cast<LONG>(fullScreenHeight - 1);
    
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

    // 发送echo命令测试通信
    std::string testCommand = "km.echo(1)";
    bool success = sendSerialCommand(testCommand);
    
    return success;
}

#endif
