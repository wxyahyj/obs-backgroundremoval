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
    int screenOffsetX;
    int screenOffsetY;
    int screenWidth;
    int screenHeight;
    float pidPMin;
    float pidPMax;
    float pidPSlope;
    float pidD;
    float baselineCompensation;
    float aimSmoothingX;
    float aimSmoothingY;
    float maxPixelMove;
    float deadZonePixels;
    float targetYOffset;
    float derivativeFilterAlpha; // 一阶低通滤波系数
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
    
    float previousMoveX;
    float previousMoveY;
    
    float pidPreviousErrorX;
    float pidPreviousErrorY;
    float filteredDeltaErrorX; // 滤波后的X轴误差差值
    float filteredDeltaErrorY; // 滤波后的Y轴误差差值
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
