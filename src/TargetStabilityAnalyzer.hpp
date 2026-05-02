#ifndef TARGET_STABILITY_ANALYZER_H
#define TARGET_STABILITY_ANALYZER_H

#include <deque>
#include <cmath>
#include <chrono>
#include "models/Detection.h"

class TargetStabilityAnalyzer {
public:
    struct StabilityResult {
        bool isStable = false;
        float stabilityScore = 0.0f;
        int stableFrameCount = 0;
        Detection smoothedTarget;
        int64_t stableDurationMs = 0;
    };

    TargetStabilityAnalyzer() = default;
    ~TargetStabilityAnalyzer() = default;

    void setConfig(int requiredFrames, float positionThreshold, float sizeThreshold) {
        requiredStableFrames_ = requiredFrames;
        positionThreshold_ = positionThreshold;
        sizeThreshold_ = sizeThreshold;
    }

    void setEnabled(bool enabled) {
        enabled_ = enabled;
        if (!enabled) {
            reset();
        }
    }

    bool isEnabled() const { return enabled_; }

    StabilityResult analyze(const Detection* currentTarget, int frameWidth, int frameHeight) {
        StabilityResult result;
        result.isStable = false;
        result.stableFrameCount = 0;
        result.stabilityScore = 0.0f;

        if (!enabled_) {
            if (currentTarget) {
                result.isStable = true;
                result.stabilityScore = 1.0f;
                result.smoothedTarget = *currentTarget;
            }
            return result;
        }

        if (!currentTarget) {
            reset();
            return result;
        }

        auto now = std::chrono::steady_clock::now();
        
        recentDetections_.push_back(*currentTarget);
        if (recentDetections_.size() > static_cast<size_t>(maxHistorySize_)) {
            recentDetections_.pop_front();
        }

        if (recentDetections_.size() < static_cast<size_t>(requiredStableFrames_)) {
            result.smoothedTarget = *currentTarget;
            return result;
        }

        float positionVariance = calculatePositionVariance(frameWidth, frameHeight);
        float sizeVariance = calculateSizeVariance();

        if (positionVariance <= positionThreshold_ && sizeVariance <= sizeThreshold_) {
            if (stableFrameCount_ == 0) {
                stableStartTime_ = now;
            }
            stableFrameCount_++;
            result.isStable = (stableFrameCount_ >= requiredStableFrames_);
            result.stableFrameCount = stableFrameCount_;
            
            if (result.isStable) {
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - stableStartTime_).count();
                result.stableDurationMs = duration;
            }
        } else {
            stableFrameCount_ = 0;
            stableStartTime_ = now;
        }

        if (positionThreshold_ > 0) {
            result.stabilityScore = std::max(0.0f, 1.0f - (positionVariance / positionThreshold_));
        }
        result.stabilityScore = std::min(1.0f, std::max(0.0f, result.stabilityScore));

        result.smoothedTarget = calculateSmoothedTarget();

        return result;
    }

    void reset() {
        recentDetections_.clear();
        stableFrameCount_ = 0;
        stableStartTime_ = std::chrono::steady_clock::now();
    }

    int getStableFrameCount() const { return stableFrameCount_; }
    float getPositionThreshold() const { return positionThreshold_; }
    int getRequiredStableFrames() const { return requiredStableFrames_; }

private:
    bool enabled_ = false;
    int requiredStableFrames_ = 3;
    float positionThreshold_ = 5.0f;
    float sizeThreshold_ = 0.1f;
    int maxHistorySize_ = 10;

    std::deque<Detection> recentDetections_;
    int stableFrameCount_ = 0;
    std::chrono::steady_clock::time_point stableStartTime_;

    float calculatePositionVariance(int frameWidth, int frameHeight) const {
        if (recentDetections_.size() < 2) return 0.0f;

        float sumX = 0.0f, sumY = 0.0f;
        for (const auto& det : recentDetections_) {
            sumX += det.centerX * frameWidth;
            sumY += det.centerY * frameHeight;
        }
        float meanX = sumX / recentDetections_.size();
        float meanY = sumY / recentDetections_.size();

        float variance = 0.0f;
        for (const auto& det : recentDetections_) {
            float dx = det.centerX * frameWidth - meanX;
            float dy = det.centerY * frameHeight - meanY;
            variance += dx * dx + dy * dy;
        }
        variance /= recentDetections_.size();

        return std::sqrt(variance);
    }

    float calculateSizeVariance() const {
        if (recentDetections_.size() < 2) return 0.0f;

        float sumArea = 0.0f;
        for (const auto& det : recentDetections_) {
            sumArea += det.width * det.height;
        }
        float meanArea = sumArea / recentDetections_.size();

        float variance = 0.0f;
        for (const auto& det : recentDetections_) {
            float diff = det.width * det.height - meanArea;
            variance += diff * diff;
        }
        variance /= recentDetections_.size();

        return std::sqrt(variance) / (meanArea > 0 ? meanArea : 1.0f);
    }

    Detection calculateSmoothedTarget() const {
        if (recentDetections_.empty()) {
            return Detection{};
        }

        Detection smoothed = recentDetections_.back();
        
        float sumX = 0.0f, sumY = 0.0f;
        float sumW = 0.0f, sumH = 0.0f;
        float sumCX = 0.0f, sumCY = 0.0f;
        
        for (const auto& det : recentDetections_) {
            sumX += det.x;
            sumY += det.y;
            sumW += det.width;
            sumH += det.height;
            sumCX += det.centerX;
            sumCY += det.centerY;
        }
        
        size_t n = recentDetections_.size();
        smoothed.x = sumX / n;
        smoothed.y = sumY / n;
        smoothed.width = sumW / n;
        smoothed.height = sumH / n;
        smoothed.centerX = sumCX / n;
        smoothed.centerY = sumCY / n;
        
        return smoothed;
    }
};

#endif
