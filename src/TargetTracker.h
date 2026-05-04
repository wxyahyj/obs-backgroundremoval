#pragma once

#include <vector>
#include <mutex>
#include <chrono>
#include <opencv2/core.hpp>
#include "models/Detection.h"
#include "HungarianAlgorithm.hpp"

#ifdef _WIN32
#include "KalmanFilter.hpp"
#endif

class TargetTracker {
public:
    struct Config {
        int maxLostFrames = 10;
        float iouThreshold = 0.3f;
        bool useKalmanTracker = false;
        
        float trackingWeightIou = 0.4f;
        float trackingWeightCenter = 0.3f;
        float trackingWeightAspect = 0.15f;
        float trackingWeightArea = 0.15f;
        
        int maxReidentifyFrames = 30;
        float reidentifyCenterThreshold = 0.1f;
        
        int kalmanGenerateThreshold = 2;
        int kalmanTerminateCount = 5;
        int kalmanPredictionFrames = 5;
    };

    struct LostTarget {
        int trackId;
        float x, y, width, height;
        float centerX, centerY;
        int lostFrames;
        std::chrono::steady_clock::time_point lostTime;
    };

#ifdef _WIN32
    struct KalmanPrediction {
        float x, y, width, height;
        int trackId;
    };
#endif

    explicit TargetTracker(const Config& config);
    ~TargetTracker() = default;

    TargetTracker(const TargetTracker&) = delete;
    TargetTracker& operator=(const TargetTracker&) = delete;
    TargetTracker(TargetTracker&&) noexcept = default;
    TargetTracker& operator=(TargetTracker&&) noexcept = default;

    std::vector<Detection> update(const std::vector<Detection>& newDetections);
    std::vector<Detection> getTrackedTargets() const;
    void reset();
    void updateConfig(const Config& config);

#ifdef _WIN32
    std::vector<KalmanPrediction> getKalmanPredictions() const;
    std::vector<std::vector<std::pair<float, float>>> getKalmanTrajectories() const;
#endif

private:
    std::vector<Detection> updateWithKalman(const std::vector<Detection>& newDetections);
    std::vector<Detection> updateWithHungarian(const std::vector<Detection>& newDetections);

    Config config_;
    std::vector<Detection> trackedTargets_;
    mutable std::mutex targetsMutex_;
    int nextTrackId_ = 0;
    
    std::vector<LostTarget> lostTargets_;
    mutable std::mutex lostTargetsMutex_;
    
#ifdef _WIN32
    KalmanP kalmanTracker_;
    std::vector<KalmanPrediction> kalmanPredictions_;
    mutable std::mutex kalmanPredictionsMutex_;
    std::vector<std::vector<std::pair<float, float>>> kalmanTrajectories_;
    mutable std::mutex kalmanTrajectoriesMutex_;
#endif
};
