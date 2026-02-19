#ifndef MOUSE_CONTROLLER_INTERFACE_HPP
#define MOUSE_CONTROLLER_INTERFACE_HPP

#include <vector>
#include <string>
#include "models/Detection.h"

enum class ControllerType {
    WindowsAPI,
    MAKCU
};

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
    float derivativeFilterAlpha;
    ControllerType controllerType;
    std::string makcuPort;
    int makcuBaudRate;
};

class MouseControllerInterface {
public:
    virtual ~MouseControllerInterface() = default;

    virtual void updateConfig(const MouseControllerConfig& config) = 0;

    virtual void setDetections(const std::vector<Detection>& detections) = 0;

    virtual void tick() = 0;
};

#endif
