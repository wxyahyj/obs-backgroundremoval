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

    if (isMoving) {
        if (!(GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000)) {
            isMoving = false;
            resetPidState();
            resetMotionState();
            return;
        }
        
        POINT currentPos;
        GetCursorPos(&currentPos);
        
        float errorX = static_cast<float>(targetPos.x - currentPos.x);
        float errorY = static_cast<float>(targetPos.y - currentPos.y);
        
        float distance = std::sqrt(errorX * errorX + errorY * errorY);
        
        if (distance < config.deadZonePixels) {
            isMoving = false;
            resetPidState();
            resetMotionState();
            return;
        }
        
        float dynamicP = calculateDynamicP(distance);
        float targetVelX = dynamicP * errorX + config.pidD * (errorX - pidPreviousErrorX);
        float targetVelY = dynamicP * errorY + config.pidD * (errorY - pidPreviousErrorY);
        
        float targetSpeed = std::sqrt(targetVelX * targetVelX + targetVelY * targetVelY);
        if (targetSpeed > 0.0f && targetSpeed < config.baselineCompensation && distance > config.deadZonePixels) {
            float scale = config.baselineCompensation / targetSpeed;
            targetVelX *= scale;
            targetVelY *= scale;
        }
        
        targetSpeed = std::sqrt(targetVelX * targetVelX + targetVelY * targetVelY);
        if (targetSpeed > config.maxSpeedPixelsPerSec && targetSpeed > 0.0f) {
            float scale = config.maxSpeedPixelsPerSec / targetSpeed;
            targetVelX *= scale;
            targetVelY *= scale;
        }
        
        float deltaTime = 11.11f / 1000.0f;
        
        float initialVelX = currentVelocityX;
        float initialVelY = currentVelocityY;
        
        float accelTime = config.sCurveTime;
        float sCurveT = sCurveProgress / accelTime;
        sCurveT = std::max(0.0f, std::min(1.0f, sCurveT));
        float sCurveFactor = sCurve(sCurveT);
        
        float blendedVelX = initialVelX + (targetVelX - initialVelX) * sCurveFactor;
        float blendedVelY = initialVelY + (targetVelY - initialVelY) * sCurveFactor;
        
        float desiredAccelX = (blendedVelX - currentVelocityX) / deltaTime;
        float desiredAccelY = (blendedVelY - currentVelocityY) / deltaTime;
        
        float desiredAccel = std::sqrt(desiredAccelX * desiredAccelX + desiredAccelY * desiredAccelY);
        if (desiredAccel > config.maxAcceleration && desiredAccel > 0.0f) {
            float scale = config.maxAcceleration / desiredAccel;
            desiredAccelX *= scale;
            desiredAccelY *= scale;
        }
        
        float jerkX = (desiredAccelX - currentAccelerationX) / deltaTime;
        float jerkY = (desiredAccelY - currentAccelerationY) / deltaTime;
        
        float jerk = std::sqrt(jerkX * jerkX + jerkY * jerkY);
        if (jerk > config.maxJerk && jerk > 0.0f) {
            float scale = config.maxJerk / jerk;
            jerkX *= scale;
            jerkY *= scale;
        }
        
        currentAccelerationX += jerkX * deltaTime;
        currentAccelerationY += jerkY * deltaTime;
        
        currentVelocityX += currentAccelerationX * deltaTime;
        currentVelocityY += currentAccelerationY * deltaTime;
        
        float currentSpeed = std::sqrt(currentVelocityX * currentVelocityX + currentVelocityY * currentVelocityY);
        if (currentSpeed > config.maxSpeedPixelsPerSec && currentSpeed > 0.0f) {
            float scale = config.maxSpeedPixelsPerSec / currentSpeed;
            currentVelocityX *= scale;
            currentVelocityY *= scale;
        }
        
        float newPosX = static_cast<float>(currentPos.x) + currentVelocityX * deltaTime;
        float newPosY = static_cast<float>(currentPos.y) + currentVelocityY * deltaTime;
        
        POINT newPos;
        newPos.x = static_cast<LONG>(newPosX);
        newPos.y = static_cast<LONG>(newPosY);
        
        sCurveProgress += deltaTime;
        if (sCurveProgress > accelTime) {
            sCurveProgress = accelTime;
        }
        
        moveMouseTo(newPos);
        
        pidPreviousErrorX = errorX;
        pidPreviousErrorY = errorY;
        
        return;
    }

    if (GetAsyncKeyState(config.hotkeyVirtualKey) & 0x8000) {
        Detection* target = selectTarget();
        if (target) {
            POINT screenPos = convertToScreenCoordinates(*target);
            startMouseMovement(screenPos);
        }
    }
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
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    float sourcePixelX = det.centerX * config.sourceWidth;
    float sourcePixelY = det.centerY * config.sourceHeight;

    float canvasPixelX = config.sourceCanvasPosX + sourcePixelX * config.sourceCanvasScaleX;
    float canvasPixelY = config.sourceCanvasPosY + sourcePixelY * config.sourceCanvasScaleY;

    POINT result;
    result.x = static_cast<LONG>(canvasPixelX);
    result.y = static_cast<LONG>(canvasPixelY);

    LONG maxX = static_cast<LONG>(screenWidth - 1);
    LONG maxY = static_cast<LONG>(screenHeight - 1);
    
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
    float p = config.pidPMin + (config.pidPMax - config.pidPMin) * (1.0f - std::exp(-distance / config.pidPSlope));
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
}

#endif
