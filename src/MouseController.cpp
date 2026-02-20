#ifdef _WIN32

#include "MouseController.hpp"
#include <cmath>
#include <algorithm>

MouseController::MouseController()
    : isMoving(false)
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
    startPos = { 0, 0 };
    targetPos = { 0, 0 };
}

MouseController::~MouseController()
{
}

void MouseController::updateConfig(const MouseControllerConfig& newConfig)
{
    std::lock_guard<std::mutex> lock(mutex);
    config = newConfig;
}

void MouseController::setDetections(const std::vector<Detection>& detections)
{
    std::lock_guard<std::mutex> lock(mutex);
    currentDetections = detections;
}

void MouseController::tick()
{
    std::lock_guard<std::mutex> lock(mutex);

    // 快速检查：鼠标控制是否启用
    if (!config.enableMouseControl) {
        return;
    }

    // 快速检查：热键是否按下
    if (!(GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000)) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        return;
    }

    // 快速检查：是否有目标
    Detection* target = selectTarget();
    if (!target) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        return;
    }

    // 计算目标屏幕位置
    POINT targetScreenPos = convertToScreenCoordinates(*target);
    
    // 获取当前鼠标位置
    POINT currentPos;
    GetCursorPos(&currentPos);
    
    // 计算误差
    float errorX = static_cast<float>(targetScreenPos.x - currentPos.x);
    float errorY = static_cast<float>(targetScreenPos.y - currentPos.y);
    
    // 计算距离（使用平方距离避免平方根）
    float distanceSquared = errorX * errorX + errorY * errorY;
    float deadZoneSquared = config.deadZonePixels * config.deadZonePixels;
    
    // 检查是否在死区内
    if (distanceSquared < deadZoneSquared) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        return;
    }

    isMoving = true;
    
    // 计算实际距离（用于动态P值计算）
    float distance = std::sqrt(distanceSquared);
    
    // 计算动态P值
    float dynamicP = calculateDynamicP(distance);
    
    // 计算误差差值
    float deltaErrorX = errorX - pidPreviousErrorX;
    float deltaErrorY = errorY - pidPreviousErrorY;
    
    // 应用一阶低通滤波
    float alpha = config.derivativeFilterAlpha;
    filteredDeltaErrorX = alpha * deltaErrorX + (1.0f - alpha) * filteredDeltaErrorX;
    filteredDeltaErrorY = alpha * deltaErrorY + (1.0f - alpha) * filteredDeltaErrorY;
    
    // 计算PID输出
    float pdOutputX = dynamicP * errorX + config.pidD * filteredDeltaErrorX;
    float pdOutputY = dynamicP * errorY + config.pidD * filteredDeltaErrorY;
    
    // 计算基线补偿
    float baselineX = errorX * config.baselineCompensation;
    float baselineY = errorY * config.baselineCompensation;
    
    // 计算最终移动量
    float moveX = pdOutputX + baselineX;
    float moveY = pdOutputY + baselineY;
    
    // 限制最大移动量
    float moveDistSquared = moveX * moveX + moveY * moveY;
    float maxMoveSquared = config.maxPixelMove * config.maxPixelMove;
    if (moveDistSquared > maxMoveSquared && moveDistSquared > 0.0f) {
        float scale = config.maxPixelMove / std::sqrt(moveDistSquared);
        moveX *= scale;
        moveY *= scale;
    }
    
    // 应用平滑处理
    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;
    
    // 更新历史值
    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;
    
    // 计算新的鼠标位置
    float newPosX = static_cast<float>(currentPos.x) + finalMoveX;
    float newPosY = static_cast<float>(currentPos.y) + finalMoveY;
    
    // 转换为整数坐标
    POINT newPos;
    newPos.x = static_cast<LONG>(newPosX);
    newPos.y = static_cast<LONG>(newPosY);
    
    // 移动鼠标
    moveMouseTo(newPos);
    
    // 更新PID历史误差
    pidPreviousErrorX = errorX;
    pidPreviousErrorY = errorY;
}

Detection* MouseController::selectTarget()
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

POINT MouseController::convertToScreenCoordinates(const Detection& det)
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

void MouseController::resetPidState()
{
    pidPreviousErrorX = 0.0f;
    pidPreviousErrorY = 0.0f;
    filteredDeltaErrorX = 0.0f;
    filteredDeltaErrorY = 0.0f;
}

float MouseController::calculateDynamicP(float distance)
{
    float normalizedDistance = distance / static_cast<float>(config.fovRadiusPixels);
    normalizedDistance = std::max(0.0f, std::min(1.0f, normalizedDistance));
    float distancePower = std::pow(normalizedDistance, config.pidPSlope);
    float p = config.pidPMin + (config.pidPMax - config.pidPMin) * distancePower;
    return std::max(config.pidPMin, std::min(config.pidPMax, p));
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

#endif
