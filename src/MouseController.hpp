#ifndef MOUSE_CONTROLLER_HPP
#define MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include <windows.h>
#include <vector>
#include <mutex>
#include <random>
#include "models/Detection.h"

struct MouseControllerConfig {
    bool enableMouseControl;
    int hotkeyVirtualKey;
    int moveDurationMs;
    int fovRadiusPixels;
    float sourceCanvasPosX;
    float sourceCanvasPosY;
    float sourceCanvasScaleX;
    float sourceCanvasScaleY;
    int sourceWidth;
    int sourceHeight;
    int algorithmType;
    float pidP;
    float pidI;
    float pidD;
    float bezierMinRadius;
    float bezierMaxRadius;
};

class MouseController {
public:
    MouseController();
    ~MouseController();

    void updateConfig(const MouseControllerConfig& config);
    
    void setDetections(const std::vector<Detection>& detections);
    
    void tick();

private:
    std::mutex mutex;
    MouseControllerConfig config;
    std::vector<Detection> currentDetections;
    
    bool isMoving;
    float currentT;
    POINT startPos;
    POINT targetPos;
    POINT controlPoint;
    POINT bezierPathPoint;
    float bezierT;
    
    std::mt19937 rng;
    
    float pidPreviousErrorX;
    float pidPreviousErrorY;
    float pidIntegralX;
    float pidIntegralY;

    Detection* selectTarget();
    POINT convertToScreenCoordinates(const Detection& det);
    POINT calculateBezierPoint(float t, const POINT& p0, const POINT& p1, const POINT& p2);
    float easeOut(float t);
    void moveMouseTo(const POINT& pos);
    void startMouseMovement(const POINT& target);
    void resetPidState();
};

#endif

#endif
