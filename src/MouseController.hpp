#ifndef MOUSE_CONTROLLER_HPP
#define MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include <windows.h>
#include <vector>
#include <mutex>
#include <random>
#include <chrono>
#include "MouseControllerInterface.hpp"

class MouseController : public MouseControllerInterface {
public:
    MouseController();
    ~MouseController();

    void updateConfig(const MouseControllerConfig& config) override;
    
    void setDetections(const std::vector<Detection>& detections) override;

    void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) override;
    
    void tick() override;
    
    void setCurrentWeapon(const std::string& weaponName);
    
    std::string getCurrentWeapon() const;

private:
    mutable std::mutex mutex;
    MouseControllerConfig config;
    std::vector<Detection> currentDetections;
    
    int cachedScreenWidth;
    int cachedScreenHeight;
    
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
    float filteredDeltaErrorX;
    float filteredDeltaErrorY;
    float calculateDynamicP(float distance);
    Detection* selectTarget();
    POINT convertToScreenCoordinates(const Detection& det);
    void moveMouseTo(const POINT& pos);
    void startMouseMovement(const POINT& target);
    void resetPidState();
    void resetMotionState();
    void performAutoClick();
    
    std::chrono::steady_clock::time_point hotkeyPressStartTime;
    bool yUnlockActive;
    std::chrono::steady_clock::time_point lastAutoTriggerTime;
    
    std::string currentWeapon_;
    int recoilPatternIndex_;
    std::chrono::steady_clock::time_point recoilStartTime_;
    bool recoilActive_;
    
    void applyRecoilCompensation(float& moveX, float& moveY);
    void resetRecoilState();
};

#endif

#endif
