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
    int inferenceFrameWidth;
    int inferenceFrameHeight;
    int cropOffsetX;
    int cropOffsetY;
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
    int yUnlockDelayMs;
    bool yUnlockEnabled;
    bool autoTriggerEnabled;
    int autoTriggerRadius;
    int autoTriggerCooldownMs;
    int autoTriggerFireDelay;
    int autoTriggerFireDuration;
    int autoTriggerInterval;
    bool autoTriggerDelayRandomEnabled;
    int autoTriggerDelayRandomMin;
    int autoTriggerDelayRandomMax;
    bool autoTriggerDurationRandomEnabled;
    int autoTriggerDurationRandomMin;
    int autoTriggerDurationRandomMax;
    int autoTriggerMoveCompensation;
    
    int targetSwitchDelayMs;
    float targetSwitchTolerance;
};

class MouseControllerInterface {
public:
    virtual ~MouseControllerInterface() = default;

    virtual void updateConfig(const MouseControllerConfig& config) = 0;

    virtual void setDetections(const std::vector<Detection>& detections) = 0;

    virtual void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) = 0;

    virtual void tick() = 0;
    
    virtual void setCurrentWeapon(const std::string& weaponName) = 0;
    
    virtual std::string getCurrentWeapon() const = 0;
    
    virtual ControllerType getControllerType() const = 0;
};

#endif
