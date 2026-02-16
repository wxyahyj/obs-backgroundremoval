#ifdef _WIN32

#include "MouseController.hpp"
#include <cmath>
#include <algorithm>

MouseController::MouseController()
    : isMoving(false)
    , currentT(0.0f)
    , bezierT(0.0f)
    , rng(std::random_device{}())
    , pidPreviousErrorX(0.0f)
    , pidPreviousErrorY(0.0f)
    , pidIntegralX(0.0f)
    , pidIntegralY(0.0f)
{
    startPos = { 0, 0 };
    targetPos = { 0, 0 };
    controlPoint = { 0, 0 };
    bezierPathPoint = { 0, 0 };
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
        if (config.algorithmType == 0) {
            currentT += 1.0f / (config.moveDurationMs / 16.666f);
            if (currentT >= 1.0f) {
                moveMouseTo(targetPos);
                isMoving = false;
                currentT = 0.0f;
            } else {
                float easedT = easeOut(currentT);
                POINT currentPos = calculateBezierPoint(easedT, startPos, controlPoint, targetPos);
                moveMouseTo(currentPos);
            }
        } else if (config.algorithmType == 1) {
            POINT currentPos;
            GetCursorPos(&currentPos);
            float errorX = static_cast<float>(currentPos.x - targetPos.x);
            float errorY = static_cast<float>(currentPos.y - targetPos.y);
            if (std::abs(errorX) < 1.0f && std::abs(errorY) < 1.0f) {
                moveMouseTo(targetPos);
                isMoving = false;
                resetPidState();
            } else {
                float derivativeX = errorX - pidPreviousErrorX;
                float derivativeY = errorY - pidPreviousErrorY;
                pidIntegralX += errorX;
                pidIntegralY += errorY;
                float outputX = config.pidP * errorX + config.pidI * pidIntegralX + config.pidD * derivativeX;
                float outputY = config.pidP * errorY + config.pidI * pidIntegralY + config.pidD * derivativeY;
                POINT newPos;
                newPos.x = static_cast<LONG>(currentPos.x - outputX);
                newPos.y = static_cast<LONG>(currentPos.y - outputY);
                moveMouseTo(newPos);
                pidPreviousErrorX = errorX;
                pidPreviousErrorY = errorY;
            }
        } else if (config.algorithmType == 2) {
            bezierT += 1.0f / (config.moveDurationMs / 16.666f);
            if (bezierT >= 1.0f) {
                moveMouseTo(targetPos);
                isMoving = false;
                bezierT = 0.0f;
                resetPidState();
            } else {
                float easedT = easeOut(bezierT);
                bezierPathPoint = calculateBezierPoint(easedT, startPos, controlPoint, targetPos);
                POINT currentPos;
                GetCursorPos(&currentPos);
                float errorX = static_cast<float>(currentPos.x - bezierPathPoint.x);
                float errorY = static_cast<float>(currentPos.y - bezierPathPoint.y);
                float derivativeX = errorX - pidPreviousErrorX;
                float derivativeY = errorY - pidPreviousErrorY;
                pidIntegralX += errorX;
                pidIntegralY += errorY;
                float outputX = config.pidP * errorX + config.pidI * pidIntegralX + config.pidD * derivativeX;
                float outputY = config.pidP * errorY + config.pidI * pidIntegralY + config.pidD * derivativeY;
                POINT newPos;
                newPos.x = static_cast<LONG>(currentPos.x - outputX);
                newPos.y = static_cast<LONG>(currentPos.y - outputY);
                moveMouseTo(newPos);
                pidPreviousErrorX = errorX;
                pidPreviousErrorY = errorY;
            }
        }
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

POINT MouseController::calculateBezierPoint(float t, const POINT& p0, const POINT& p1, const POINT& p2)
{
    float t2 = t * t;
    float mt = 1.0f - t;
    float mt2 = mt * mt;

    POINT result;
    result.x = static_cast<int>(mt2 * p0.x + 2 * mt * t * p1.x + t2 * p2.x);
    result.y = static_cast<int>(mt2 * p0.y + 2 * mt * t * p1.y + t2 * p2.y);

    return result;
}

float MouseController::easeOut(float t)
{
    return 1.0f - std::pow(1.0f - t, 2.0f);
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

    if (config.algorithmType == 0 || config.algorithmType == 2) {
        currentT = 0.0f;
        bezierT = 0.0f;
        int dx = targetPos.x - startPos.x;
        int dy = targetPos.y - startPos.y;
        float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy));

        std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.1415926f);
        std::uniform_real_distribution<float> radiusDist(distance * config.bezierMinRadius, distance * config.bezierMaxRadius);

        float angle = angleDist(rng);
        float radius = radiusDist(rng);

        float midX = (startPos.x + targetPos.x) / 2.0f;
        float midY = (startPos.y + targetPos.y) / 2.0f;

        controlPoint.x = static_cast<int>(midX + std::cos(angle) * radius);
        controlPoint.y = static_cast<int>(midY + std::sin(angle) * radius);
    }
}

void MouseController::resetPidState()
{
    pidPreviousErrorX = 0.0f;
    pidPreviousErrorY = 0.0f;
    pidIntegralX = 0.0f;
    pidIntegralY = 0.0f;
}

#endif
