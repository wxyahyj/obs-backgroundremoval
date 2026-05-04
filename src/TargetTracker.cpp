#include "TargetTracker.h"
#include <algorithm>
#include <cmath>

TargetTracker::TargetTracker(const Config& config) : config_(config) {
#ifdef _WIN32
    if (config_.useKalmanTracker) {
        kalmanTracker_.init(config_.kalmanGenerateThreshold, config_.kalmanTerminateCount);
    }
#endif
}

std::vector<Detection> TargetTracker::update(const std::vector<Detection>& newDetections) {
#ifdef _WIN32
    if (config_.useKalmanTracker) {
        return updateWithKalman(newDetections);
    } else {
        return updateWithHungarian(newDetections);
    }
#else
    return updateWithHungarian(newDetections);
#endif
}

std::vector<Detection> TargetTracker::getTrackedTargets() const {
    std::lock_guard<std::mutex> lock(targetsMutex_);
    return trackedTargets_;
}

void TargetTracker::reset() {
    std::lock_guard<std::mutex> lock(targetsMutex_);
    trackedTargets_.clear();
    nextTrackId_ = 0;
    
    std::lock_guard<std::mutex> lostLock(lostTargetsMutex_);
    lostTargets_.clear();
    
#ifdef _WIN32
    std::lock_guard<std::mutex> predLock(kalmanPredictionsMutex_);
    kalmanPredictions_.clear();
    
    std::lock_guard<std::mutex> trajLock(kalmanTrajectoriesMutex_);
    kalmanTrajectories_.clear();
    
    kalmanTracker_.reset();
#endif
}

void TargetTracker::updateConfig(const Config& config) {
    config_ = config;
#ifdef _WIN32
    if (config_.useKalmanTracker) {
        kalmanTracker_.init(config_.kalmanGenerateThreshold, config_.kalmanTerminateCount);
    }
#endif
}

#ifdef _WIN32
std::vector<TargetTracker::KalmanPrediction> TargetTracker::getKalmanPredictions() const {
    std::lock_guard<std::mutex> lock(kalmanPredictionsMutex_);
    return kalmanPredictions_;
}

std::vector<std::vector<std::pair<float, float>>> TargetTracker::getKalmanTrajectories() const {
    std::lock_guard<std::mutex> lock(kalmanTrajectoriesMutex_);
    return kalmanTrajectories_;
}

std::vector<Detection> TargetTracker::updateWithKalman(const std::vector<Detection>& newDetections) {
    std::vector<KalmanDetail::DetectionObject> kalmanDets;
    for (const auto& det : newDetections) {
        KalmanDetail::DetectionObject kdet;
        kdet.bbox.x = det.x;
        kdet.bbox.y = det.y;
        kdet.bbox.width = det.width;
        kdet.bbox.height = det.height;
        kdet.label = det.classId;
        kdet.prob = det.confidence;
        kdet.track_id = -1;
        kalmanDets.push_back(kdet);
    }
    
    std::vector<KalmanDetail::DetectionObject> kalmanTracked = kalmanTracker_.predict(kalmanDets);
    
    {
        std::lock_guard<std::mutex> predLock(kalmanPredictionsMutex_);
        kalmanPredictions_.clear();
        std::vector<KalmanDetail::DetectionObject> predictions = kalmanTracker_.getPredictions();
        for (const auto& pred : predictions) {
            KalmanPrediction kp;
            kp.x = pred.bbox.x;
            kp.y = pred.bbox.y;
            kp.width = pred.bbox.width;
            kp.height = pred.bbox.height;
            kp.trackId = pred.track_id;
            kalmanPredictions_.push_back(kp);
        }
    }
    
    {
        std::lock_guard<std::mutex> trajLock(kalmanTrajectoriesMutex_);
        kalmanTrajectories_ = kalmanTracker_.getMultiFramePredictions(config_.kalmanPredictionFrames);
    }
    
    std::vector<Detection> trackedDetections;
    for (const auto& kt : kalmanTracked) {
        Detection det;
        det.x = kt.bbox.x;
        det.y = kt.bbox.y;
        det.width = kt.bbox.width;
        det.height = kt.bbox.height;
        det.centerX = det.x + det.width / 2.0f;
        det.centerY = det.y + det.height / 2.0f;
        det.classId = kt.label;
        det.confidence = kt.prob;
        det.trackId = kt.track_id;
        det.lostFrames = 0;
        trackedDetections.push_back(det);
    }
    
    std::lock_guard<std::mutex> lock(targetsMutex_);
    trackedTargets_ = trackedDetections;
    return trackedDetections;
}
#endif

std::vector<Detection> TargetTracker::updateWithHungarian(const std::vector<Detection>& newDetections) {
    std::lock_guard<std::mutex> lock(targetsMutex_);
    
    std::vector<Detection> trackedDetections;
    
    if (newDetections.empty() && trackedTargets_.empty()) {
        return trackedDetections;
    }
    
    if (trackedTargets_.empty()) {
        for (const auto& det : newDetections) {
            Detection newDet = det;
            newDet.trackId = nextTrackId_++;
            newDet.lostFrames = 0;
            trackedDetections.push_back(newDet);
        }
    } else {
        int n = static_cast<int>(newDetections.size());
        int m = static_cast<int>(trackedTargets_.size());
        
        std::vector<std::vector<float>> costMatrix(n, std::vector<float>(m, 1.0f));
        
        for (int i = 0; i < n; ++i) {
            cv::Rect2f detBox(
                newDetections[i].x,
                newDetections[i].y,
                newDetections[i].width,
                newDetections[i].height
            );
            cv::Point2f detCenter(newDetections[i].centerX, newDetections[i].centerY);
            
            for (int j = 0; j < m; ++j) {
                cv::Rect2f trackBox(
                    trackedTargets_[j].x,
                    trackedTargets_[j].y,
                    trackedTargets_[j].width,
                    trackedTargets_[j].height
                );
                cv::Point2f trackCenter(trackedTargets_[j].centerX, trackedTargets_[j].centerY);
                
                costMatrix[i][j] = HungarianAlgorithm::calculateFusedDistance(
                    detBox, trackBox, detCenter, trackCenter,
                    config_.trackingWeightIou,
                    config_.trackingWeightCenter,
                    config_.trackingWeightAspect,
                    config_.trackingWeightArea);
            }
        }
        
        std::vector<int> assignment = HungarianAlgorithm::solve(costMatrix);
        
        std::vector<bool> detectionMatched(n, false);
        std::vector<bool> trackMatched(m, false);
        
        for (int i = 0; i < n; ++i) {
            int j = assignment[i];
            if (j >= 0 && j < m && costMatrix[i][j] < (1.0f - config_.iouThreshold)) {
                Detection newDet = newDetections[i];
                newDet.trackId = trackedTargets_[j].trackId;
                newDet.lostFrames = 0;
                trackedDetections.push_back(newDet);
                detectionMatched[i] = true;
                trackMatched[j] = true;
            }
        }
        
        for (int i = 0; i < n; ++i) {
            if (!detectionMatched[i]) {
                Detection newDet = newDetections[i];
                newDet.trackId = nextTrackId_++;
                newDet.lostFrames = 0;
                trackedDetections.push_back(newDet);
            }
        }
        
        for (int j = 0; j < m; ++j) {
            if (!trackMatched[j]) {
                trackedTargets_[j].lostFrames++;
                if (trackedTargets_[j].lostFrames <= config_.maxLostFrames) {
                    trackedDetections.push_back(trackedTargets_[j]);
                } else {
                    std::lock_guard<std::mutex> lostLock(lostTargetsMutex_);
                    LostTarget lost;
                    lost.trackId = trackedTargets_[j].trackId;
                    lost.x = trackedTargets_[j].x;
                    lost.y = trackedTargets_[j].y;
                    lost.width = trackedTargets_[j].width;
                    lost.height = trackedTargets_[j].height;
                    lost.centerX = trackedTargets_[j].centerX;
                    lost.centerY = trackedTargets_[j].centerY;
                    lost.lostFrames = 0;
                    lost.lostTime = std::chrono::steady_clock::now();
                    
                    bool found = false;
                    for (auto& existing : lostTargets_) {
                        if (existing.trackId == lost.trackId) {
                            existing = lost;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        lostTargets_.push_back(lost);
                    }
                }
            }
        }
        
        {
            std::lock_guard<std::mutex> lostLock(lostTargetsMutex_);
            auto now = std::chrono::steady_clock::now();
            for (auto it = lostTargets_.begin(); it != lostTargets_.end(); ) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->lostTime).count();
                if (elapsed > config_.maxReidentifyFrames * 33) {
                    it = lostTargets_.erase(it);
                    continue;
                }
                
                for (int i = 0; i < n; ++i) {
                    if (detectionMatched[i]) continue;
                    
                    float dx = newDetections[i].centerX - it->centerX;
                    float dy = newDetections[i].centerY - it->centerY;
                    float centerDist = std::sqrt(dx * dx + dy * dy);
                    
                    if (centerDist < config_.reidentifyCenterThreshold) {
                        Detection newDet = newDetections[i];
                        newDet.trackId = it->trackId;
                        newDet.lostFrames = 0;
                        trackedDetections.push_back(newDet);
                        detectionMatched[i] = true;
                        it = lostTargets_.erase(it);
                        break;
                    }
                }
                
                if (it != lostTargets_.end()) {
                    ++it;
                }
            }
        }
    }
    
    trackedTargets_ = trackedDetections;
    return trackedDetections;
}
