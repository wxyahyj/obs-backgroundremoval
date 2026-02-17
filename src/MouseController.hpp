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
    int fovRadiusPixels;
    float sourceCanvasPosX;
    float sourceCanvasPosY;
    float sourceCanvasScaleX;
    float sourceCanvasScaleY;
    int sourceWidth;
    int sourceHeight;
    float pidPMin;
    float pidPMax;
    float pidPSlope;
    float pidD;
    float baselineCompensation;
    float maxSpeedPixelsPerSec;
    float deadZonePixels;
    float maxAcceleration;
    float maxJerk;
    float sCurveTime;
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
    POINT startPos;
    POINT targetPos;
    
    float currentVelocityX;
    float currentVelocityY;
    float currentAccelerationX;
    float currentAccelerationY;
    
    float sCurveProgress;
    float sCurveTotalTime;
    
    float pidPreviousErrorX;
    float pidPreviousErrorY;

    float sCurve(float t);
    float calculateDynamicP(float distance);
    Detection* selectTarget();
    POINT convertToScreenCoordinates(const Detection& det);
    void moveMouseTo(const POINT& pos);
    void startMouseMovement(const POINT& target);
    void resetPidState();
    void resetMotionState();
};

#endif

#endif
