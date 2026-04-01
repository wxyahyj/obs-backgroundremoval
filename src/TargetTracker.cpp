#ifdef _WIN32

#include "TargetTracker.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>

// TrackedTarget 实现
TrackedTarget::TrackedTarget()
    : trackId(-1)
    , persistentId(-1)
    , centerX(0.0f)
    , centerY(0.0f)
    , width(0.0f)
    , height(0.0f)
    , confidence(0.0f)
    , lostFrames(0)
    , totalSeenFrames(0)
    , kalmanInitialized(false)
    , historyIndex(0)
{
    lastSeenTime = std::chrono::steady_clock::now();
    std::fill(std::begin(historyX), std::end(historyX), 0.0f);
    std::fill(std::begin(historyY), std::end(historyY), 0.0f);
}

void TrackedTarget::update(const Detection& det, float deltaTime)
{
    trackId = det.trackId;
    centerX = det.centerX;
    centerY = det.centerY;
    width = det.width;
    height = det.height;
    confidence = det.confidence;
    lostFrames = 0;
    totalSeenFrames++;
    lastSeenTime = std::chrono::steady_clock::now();
    
    // 更新历史位置
    historyX[historyIndex] = centerX;
    historyY[historyIndex] = centerY;
    historyIndex = (historyIndex + 1) % HISTORY_SIZE;
    
    // 更新卡尔曼滤波器
    if (!kalmanInitialized) {
        kalmanFilter.init(centerX, centerY);
        kalmanInitialized = true;
    } else {
        kalmanFilter.predict(deltaTime);
        kalmanFilter.update(centerX, centerY, confidence);
    }
}

void TrackedTarget::predict(float deltaTime, float& predX, float& predY)
{
    if (kalmanInitialized) {
        // 使用卡尔曼滤波器预测
        kalmanFilter.predict(deltaTime);
        float x, y, vx, vy;
        kalmanFilter.getState(x, y, vx, vy);
        predX = x + vx * deltaTime;
        predY = y + vy * deltaTime;
    } else {
        // 使用历史平均速度预测
        float vx, vy;
        getAverageVelocity(vx, vy, deltaTime);
        predX = centerX + vx * deltaTime;
        predY = centerY + vy * deltaTime;
    }
}

float TrackedTarget::getIOU(const Detection& det) const
{
    // 计算两个框的IOU
    float x1 = centerX - width / 2.0f;
    float y1 = centerY - height / 2.0f;
    float x2 = centerX + width / 2.0f;
    float y2 = centerY + height / 2.0f;
    
    float detX1 = det.x;
    float detY1 = det.y;
    float detX2 = det.x + det.width;
    float detY2 = det.y + det.height;
    
    // 计算交集
    float interX1 = std::max(x1, detX1);
    float interY1 = std::max(y1, detY1);
    float interX2 = std::min(x2, detX2);
    float interY2 = std::min(y2, detY2);
    
    if (interX2 <= interX1 || interY2 <= interY1) {
        return 0.0f;
    }
    
    float interArea = (interX2 - interX1) * (interY2 - interY1);
    float area1 = width * height;
    float area2 = det.width * det.height;
    float unionArea = area1 + area2 - interArea;
    
    return unionArea > 0.0f ? interArea / unionArea : 0.0f;
}

float TrackedTarget::getDistanceToDetection(const Detection& det, int frameWidth, int frameHeight) const
{
    float dx = (centerX - det.centerX) * frameWidth;
    float dy = (centerY - det.centerY) * frameHeight;
    return std::sqrt(dx * dx + dy * dy);
}

bool TrackedTarget::isValid(int maxLostFrames) const
{
    return lostFrames < maxLostFrames;
}

void TrackedTarget::markLost()
{
    lostFrames++;
}

void TrackedTarget::getAverageVelocity(float& vx, float& vy, float deltaTime) const
{
    if (totalSeenFrames < 2 || deltaTime <= 0.0f) {
        vx = 0.0f;
        vy = 0.0f;
        return;
    }
    
    // 使用历史位置计算平均速度
    int validSamples = std::min(totalSeenFrames, HISTORY_SIZE);
    if (validSamples < 2) {
        vx = 0.0f;
        vy = 0.0f;
        return;
    }
    
    float sumVx = 0.0f;
    float sumVy = 0.0f;
    int count = 0;
    
    for (int i = 1; i < validSamples; i++) {
        int idx1 = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
        int idx2 = (historyIndex - i - 1 + HISTORY_SIZE) % HISTORY_SIZE;
        
        float dx = historyX[idx1] - historyX[idx2];
        float dy = historyY[idx1] - historyY[idx2];
        
        sumVx += dx / deltaTime;
        sumVy += dy / deltaTime;
        count++;
    }
    
    vx = count > 0 ? sumVx / count : 0.0f;
    vy = count > 0 ? sumVy / count : 0.0f;
}

// TargetTracker 实现
TargetTracker::TargetTracker()
    : nextPersistentId(0)
    , currentPersistentId(-1)
    , iouThreshold(0.3f)
    , distanceThreshold(100.0f)
    , maxLostFrames(30)
    , kalmanProcessNoise(0.01f)
    , kalmanMeasurementNoise(1.0f)
{
}

void TargetTracker::update(const std::vector<Detection>& detections, float deltaTime, int frameWidth, int frameHeight)
{
    // 1. 数据关联
    std::vector<int> matchedDetectionIndices;
    std::vector<int> matchedTargetIds;
    std::vector<int> unmatchedDetectionIndices;
    
    associateDetections(detections, matchedDetectionIndices, matchedTargetIds,
                        unmatchedDetectionIndices, frameWidth, frameHeight, deltaTime);
    
    // 2. 更新匹配的跟踪目标
    for (size_t i = 0; i < matchedDetectionIndices.size(); i++) {
        int detIdx = matchedDetectionIndices[i];
        int targetId = matchedTargetIds[i];
        
        auto it = trackedTargets.find(targetId);
        if (it != trackedTargets.end()) {
            it->second.update(detections[detIdx], deltaTime);
        }
    }
    
    // 3. 为未匹配的检测创建新目标
    for (int detIdx : unmatchedDetectionIndices) {
        createNewTarget(detections[detIdx]);
    }
    
    // 4. 标记未匹配的跟踪目标为丢失
    for (auto& pair : trackedTargets) {
        bool matched = false;
        for (int targetId : matchedTargetIds) {
            if (targetId == pair.first) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            pair.second.markLost();
        }
    }
    
    // 5. 清理丢失太久的目标
    cleanupLostTargets();
}

TrackedTarget* TargetTracker::getLockedTarget()
{
    if (currentPersistentId < 0) {
        return nullptr;
    }
    
    auto it = trackedTargets.find(currentPersistentId);
    if (it != trackedTargets.end() && it->second.isValid(maxLostFrames)) {
        return &(it->second);
    }
    
    return nullptr;
}

void TargetTracker::lockTarget(int persistentId)
{
    auto it = trackedTargets.find(persistentId);
    if (it != trackedTargets.end() && it->second.isValid(maxLostFrames)) {
        currentPersistentId = persistentId;
    }
}

void TargetTracker::unlock()
{
    currentPersistentId = -1;
}

TrackedTarget* TargetTracker::getBestTarget(int frameWidth, int frameHeight, 
                                            int fovCenterX, int fovCenterY, float fovRadius)
{
    TrackedTarget* bestTarget = nullptr;
    float bestScore = -1.0f;
    float fovRadiusSq = fovRadius * fovRadius;
    
    for (auto& pair : trackedTargets) {
        TrackedTarget& target = pair.second;
        
        if (!target.isValid(maxLostFrames)) {
            continue;
        }
        
        // 计算到FOV中心的距离
        float pixelX = target.centerX * frameWidth;
        float pixelY = target.centerY * frameHeight;
        float dx = pixelX - fovCenterX;
        float dy = pixelY - fovCenterY;
        float distSq = dx * dx + dy * dy;
        
        if (distSq > fovRadiusSq) {
            continue;
        }
        
        // 计算评分
        float distance = std::sqrt(distSq);
        float distanceScore = 1.0f / (1.0f + distance * 0.01f);
        float confidenceScore = target.confidence;
        float stabilityScore = std::min(target.totalSeenFrames / 10.0f, 1.0f);
        
        float score = 0.5f * distanceScore + 0.3f * confidenceScore + 0.2f * stabilityScore;
        
        if (score > bestScore) {
            bestScore = score;
            bestTarget = &target;
        }
    }
    
    return bestTarget;
}

void TargetTracker::reset()
{
    trackedTargets.clear();
    nextPersistentId = 0;
    currentPersistentId = -1;
}

bool TargetTracker::isTargetMatched(const std::unordered_map<int, bool>& targetMatched, int targetId)
{
    return targetMatched.count(targetId) && targetMatched.at(targetId);
}

void TargetTracker::associateDetections(const std::vector<Detection>& detections,
                                        std::vector<int>& matchedDetectionIndices,
                                        std::vector<int>& matchedTargetIds,
                                        std::vector<int>& unmatchedDetectionIndices,
                                        int frameWidth, int frameHeight, float deltaTime)
{
    matchedDetectionIndices.clear();
    matchedTargetIds.clear();
    unmatchedDetectionIndices.clear();

    std::vector<bool> detectionMatched(detections.size(), false);
    // 使用unordered_map来安全地跟踪匹配状态，避免persistentId作为数组索引的越界问题
    std::unordered_map<int, bool> targetMatched;

    // 第一阶段：使用IOU匹配
    for (size_t detIdx = 0; detIdx < detections.size(); detIdx++) {
        float bestIOU = iouThreshold;
        int bestTargetId = -1;

        for (auto& pair : trackedTargets) {
            // 检查目标是否已匹配
            if (isTargetMatched(targetMatched, pair.first)) continue;

            float iou = pair.second.getIOU(detections[detIdx]);
            if (iou > bestIOU) {
                bestIOU = iou;
                bestTargetId = pair.first;
            }
        }

        if (bestTargetId >= 0) {
            matchedDetectionIndices.push_back(static_cast<int>(detIdx));
            matchedTargetIds.push_back(bestTargetId);
            detectionMatched[detIdx] = true;
            targetMatched[bestTargetId] = true;
        }
    }

    // 第二阶段：对未匹配的检测，使用卡尔曼预测位置+距离匹配
    for (size_t detIdx = 0; detIdx < detections.size(); detIdx++) {
        if (detectionMatched[detIdx]) continue;

        float bestDistance = distanceThreshold;
        int bestTargetId = -1;

        for (auto& pair : trackedTargets) {
            // 检查目标是否已匹配
            if (isTargetMatched(targetMatched, pair.first)) continue;
            if (!pair.second.kalmanInitialized) continue;

            // 使用卡尔曼滤波器预测位置
            float predX, predY;
            pair.second.predict(deltaTime, predX, predY);

            // 计算预测位置与检测的距离
            float dx = (predX - detections[detIdx].centerX) * frameWidth;
            float dy = (predY - detections[detIdx].centerY) * frameHeight;
            float distance = std::sqrt(dx * dx + dy * dy);

            if (distance < bestDistance) {
                bestDistance = distance;
                bestTargetId = pair.first;
            }
        }

        if (bestTargetId >= 0) {
            matchedDetectionIndices.push_back(static_cast<int>(detIdx));
            matchedTargetIds.push_back(bestTargetId);
            detectionMatched[detIdx] = true;
            targetMatched[bestTargetId] = true;
        }
    }

    // 收集未匹配的检测
    for (size_t i = 0; i < detections.size(); i++) {
        if (!detectionMatched[i]) {
            unmatchedDetectionIndices.push_back(static_cast<int>(i));
        }
    }
}

int TargetTracker::createNewTarget(const Detection& det)
{
    int persistentId = nextPersistentId++;
    TrackedTarget target;
    target.persistentId = persistentId;
    target.trackId = det.trackId;
    target.centerX = det.centerX;
    target.centerY = det.centerY;
    target.width = det.width;
    target.height = det.height;
    target.confidence = det.confidence;
    target.totalSeenFrames = 1;
    target.lostFrames = 0;
    target.lastSeenTime = std::chrono::steady_clock::now();
    target.kalmanInitialized = false;
    target.historyIndex = 0;
    
    std::fill(std::begin(target.historyX), std::end(target.historyX), det.centerX);
    std::fill(std::begin(target.historyY), std::end(target.historyY), det.centerY);
    
    trackedTargets[persistentId] = target;
    return persistentId;
}

void TargetTracker::cleanupLostTargets()
{
    for (auto it = trackedTargets.begin(); it != trackedTargets.end();) {
        if (it->second.lostFrames >= maxLostFrames) {
            // 如果当前锁定的目标被清理，解锁
            if (it->first == currentPersistentId) {
                currentPersistentId = -1;
            }
            it = trackedTargets.erase(it);
        } else {
            ++it;
        }
    }
}

#endif // _WIN32
