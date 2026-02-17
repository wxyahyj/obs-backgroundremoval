#ifdef _WIN32

#include "MouseController.hpp"
#include <cmath>
#include <algorithm>

MouseController::MouseController()
    : isMoving(false)
    , pidPreviousErrorX(0.0f)
    , pidPreviousErrorY(0.0f)
    , currentVelocityX(0.0f)
    , currentVelocityY(0.0f)
    , currentAccelerationX(0.0f)
    , currentAccelerationY(0.0f)
    , sCurveProgress(0.0f)
    , sCurveTotalTime(0.0f)
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
    
    float distance = std::sqrt(errorX * errorX + errorY * errorY);
    
    if (distance < config.deadZonePixels) {
        if (isMoving) {
            isMoving = false;
            resetPidState();
            resetMotionState();
        }
        return;
    }

    isMoving = true;
    
    float dynamicP = calculateDynamicP(distance);
    float pdOutputX = dynamicP * errorX + config.pidD * (errorX - pidPreviousErrorX);
    float pdOutputY = dynamicP * errorY + config.pidD * (errorY - pidPreviousErrorY);
    
    float moveX = pdOutputX + baselineX;
    float moveY = pdOutputY + baselineY;
    
    float moveDist = std::sqrt(moveX * moveX + moveY * moveY);
    if (moveDist > config.maxPixelMove && moveDist > 0.0f) {
        float scale = config.maxPixelMove / moveDist;
        moveX *= scale;
        moveY *= scale;
    }
    
    float finalMoveX = previousMoveX * (1.0f - config.aimSmoothingX) + moveX * config.aimSmoothingX;
    float finalMoveY = previousMoveY * (1.0f - config.aimSmoothingY) + moveY * config.aimSmoothingY;
    
    previousMoveX = finalMoveX;
    previousMoveY = finalMoveY;
    
    float newPosX = static_cast<float>(currentPos.x) + finalMoveX;
    float newPosY = static_cast<float>(currentPos.y) + finalMoveY;
    
    POINT newPos;
    newPos.x = static_cast<LONG>(newPosX);
    newPos.y = static_cast<LONG>(newPosY);
    
    moveMouseTo(newPos);
    
    pidPreviousErrorX = errorX;
    pidPreviousErrorY = errorY;
}

Detection* MouseController::selectTarget()
{
    if (currentDetections.empty()) {
        return nullptr;
    }

    float fovCenterX = 0.5f;
    float fovCenterY = 0.5f;

    Detection* bestTarget = nullptr;
    float minDistance = std::numeric_limits<float>::max();

    for (auto& det : currentDetections) {
        float dx = det.centerX - fovCenterX;
        float dy = det.centerY - fovCenterY;
        float distance = std::sqrt(dx * dx + dy * dy);

        float normalizedRadius = static_cast<float>(config.fovRadiusPixels) / 
                                 std::min(config.sourceWidth, config.sourceHeight);
        
        if (distance <= normalizedRadius && distance < minDistance) {
            minDistance = distance;
            bestTarget = &det;
        }
    }

    return bestTarget;
}

POINT MouseController::convertToScreenCoordinates(const Detection& det)
{
    int fullScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    int fullScreenHeight = GetSystemMetrics(SM_CYSCREEN);

    float sourcePixelX = det.centerX * config.sourceWidth;
    float sourcePixelY = det.centerY * config.sourceHeight;

    float screenScaleX = (config.screenWidth > 0) ? (float)config.screenWidth / config.sourceWidth : 1.0f;
    float screenScaleY = (config.screenHeight > 0) ? (float)config.screenHeight / config.sourceHeight : 1.0f;

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
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    INPUT input = {};
    input.type = INPUT_MOUSE;
    input.mi.dx = static_cast<LONG>((pos.x * 65535.0f) / (screenWidth - 1));
    input.mi.dy = static_cast<LONG>((pos.y * 65535.0f) / (screenHeight - 1));
    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
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
}

float MouseController::sCurve(float t)
{
    t = std::max(0.0f, std::min(1.0f, t));
    return 3.0f * t * t - 2.0f * t * t * t;
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
    sCurveProgress = 0.0f;
    sCurveTotalTime = 0.0f;
    previousMoveX = 0.0f;
    previousMoveY = 0.0f;
}

#endif
