#ifdef _WIN32

#include "MouseController.hpp"
#include <plugin-support.h>
#include <cmath>
#include <algorithm>

MouseController::MouseController()
    : cachedScreenWidth(0)
    , cachedScreenHeight(0)
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
    config = newConfig;
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

    static bool loggedOnce = false;
    if (!loggedOnce) {
        obs_log(LOG_INFO, "[MouseController] 坐标调试:");
        obs_log(LOG_INFO, "  推理帧尺寸: %dx%d", frameWidth, frameHeight);
        obs_log(LOG_INFO, "  目标归一化中心: (%.4f, %.4f)", target->centerX, target->centerY);
        obs_log(LOG_INFO, "  目标像素位置: (%.1f, %.1f)", targetPixelX, targetPixelY);
        obs_log(LOG_INFO, "  FOV中心: (%.1f, %.1f)", fovCenterX, fovCenterY);
        obs_log(LOG_INFO, "  误差(移动量): (%.1f, %.1f)", errorX, errorY);
        obs_log(LOG_INFO, "  屏幕偏移: (%d, %d)", config.screenOffsetX, config.screenOffsetY);
        loggedOnce = true;
    }

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
    
    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dx = static_cast<LONG>(finalMoveX);
    input.mi.dy = static_cast<LONG>(finalMoveY);
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.time = 0;
    input.mi.dwExtraInfo = 0;
    SendInput(1, &input, sizeof(INPUT));
    
    pidPreviousErrorX = errorX;
    pidPreviousErrorY = errorY;
}

Detection* MouseController::selectTarget()
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
