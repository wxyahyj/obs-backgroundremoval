#ifndef ABSTRACT_MOUSE_CONTROLLER_HPP
#define ABSTRACT_MOUSE_CONTROLLER_HPP

#ifdef _WIN32

#include <windows.h>
#include <vector>
#include <mutex>
#include <random>
#include <chrono>
#include <string>
#include "MouseControllerInterface.hpp"
#include "DerivativePredictor.hpp"
#include "KalmanFilter.hpp"
#include "DopaPIDController.hpp"
#include "ChrisPIDController.hpp"

class AbstractMouseController : public MouseControllerInterface {
protected:
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
    float previousErrorX;
    float previousErrorY;
    
    float previousTargetX;
    float previousTargetY;
    float targetVelocityX;
    float targetVelocityY;

    float integralX;
    float integralY;
    float integralGainX;
    float integralGainY;
    
    float stdIntegralX;
    float stdIntegralY;
    float stdIntegralGainX;
    float stdIntegralGainY;
    float stdLastErrorX;
    float stdLastErrorY;
    float stdFilteredDeltaErrorX;
    float stdFilteredDeltaErrorY;
    float stdPreviousMoveX;
    float stdPreviousMoveY;
    
    int lockedTrackId;
    
    std::chrono::steady_clock::time_point lastRecoilTime;
    bool isFiring;
    
    DerivativePredictor predictor;
    KalmanFilter kalmanFilter;
    bool kalmanFilterInitialized;
    
    DopaDualAxisPID dopaController_;
    ChrisAimController chrisController_;

    std::chrono::steady_clock::time_point lastTickTime;
    float deltaTime;
    
    std::chrono::steady_clock::time_point hotkeyPressStartTime;
    bool yUnlockActive;
    std::chrono::steady_clock::time_point lastAutoTriggerTime;
    std::chrono::steady_clock::time_point autoTriggerFireStartTime;
    std::chrono::steady_clock::time_point autoTriggerDelayStartTime;
    bool autoTriggerHolding;
    bool autoTriggerWaitingForDelay;
    int currentFireDuration;
    std::mt19937 randomGenerator;
    
    int currentTargetTrackId;
    std::chrono::steady_clock::time_point targetLockStartTime;
    float currentTargetDistance;
    
    // 延迟转火相关
    int pendingTargetTrackId;
    std::chrono::steady_clock::time_point pendingTargetStartTime;
    float pendingTargetScore;
    float currentTargetScore;
    
    std::string currentWeapon;
    
    float bezierPhase;
    
    PidDataCallback pidDataCallback_;

    virtual void moveMouse(int dx, int dy) = 0;
    virtual void performClickDown() = 0;
    virtual void performClickUp() = 0;
    virtual bool checkFiring() = 0;
    
    float calculateDynamicP(float distance);
    float calculateAdaptiveD(float distance, float deltaError, float error, float& adaptiveFactor);
    float calculateIntegral(float error, float& integral, float& integralGain, float lastError, float deltaTime);
    bool adjustIntegralGain(float error, float lastError, float& integralGain);
    float calculateStandardPID(float error, float& integral, float& integralGain, 
                               float& lastError, float& filteredDeltaError, float deltaTime);
    bool adjustStandardIntegral(float error, float lastError, float& integralGain);
    Detection* selectTarget();
    POINT convertToScreenCoordinates(const Detection& det);
    void resetPidState();
    void resetMotionState();
    void performAutoClick();
    void releaseAutoTrigger();
    int getRandomDelay();
    int getRandomDuration();
    float getCurrentPGain();
    
    virtual const char* getLogPrefix() const;

public:
    AbstractMouseController();
    virtual ~AbstractMouseController() = default;

    void updateConfig(const MouseControllerConfig& config) override;
    void setDetections(const std::vector<Detection>& detections) override;
    void setDetectionsWithFrameSize(const std::vector<Detection>& detections, int frameWidth, int frameHeight, int cropX, int cropY) override;
    void tick() override;
    void setCurrentWeapon(const std::string& weaponName) override;
    std::string getCurrentWeapon() const override;
    bool getKalmanPrediction(float& predX, float& predY) const override;
    void setPidDataCallback(PidDataCallback callback) override;
};

#endif

#endif
