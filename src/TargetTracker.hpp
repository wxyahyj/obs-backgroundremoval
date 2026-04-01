#ifndef TARGET_TRACKER_HPP
#define TARGET_TRACKER_HPP

#include <vector>
#include <map>
#include <chrono>
#include "models/Detection.h"
#include "KalmanFilter.hpp"

// 跟踪目标的状态信息
struct TrackedTarget {
    int trackId;                           // 当前跟踪ID
    int persistentId;                      // 持久化ID（即使trackId变化也保持不变）
    float centerX, centerY;                // 归一化中心坐标
    float width, height;                   // 归一化宽高
    float confidence;                      // 置信度
    int lostFrames;                        // 连续丢失帧数
    int totalSeenFrames;                   // 总共看到帧数
    std::chrono::steady_clock::time_point lastSeenTime;  // 最后看到时间
    KalmanFilter kalmanFilter;             // 每个目标独立的卡尔曼滤波器
    bool kalmanInitialized;                // 卡尔曼滤波器是否已初始化
    
    // 历史位置（用于计算速度）
    static constexpr int HISTORY_SIZE = 5;
    float historyX[HISTORY_SIZE];
    float historyY[HISTORY_SIZE];
    int historyIndex;
    
    TrackedTarget();
    
    // 更新目标状态
    void update(const Detection& det, float deltaTime);
    
    // 预测位置
    void predict(float deltaTime, float& predX, float& predY);
    
    // 获取IOU（交并比）
    float getIOU(const Detection& det) const;
    
    // 获取预测位置与检测的距离
    float getDistanceToDetection(const Detection& det, int frameWidth, int frameHeight) const;
    
    // 检查目标是否有效（未丢失太久）
    bool isValid(int maxLostFrames = 30) const;
    
    // 标记为丢失
    void markLost();
    
    // 获取平均速度
    void getAverageVelocity(float& vx, float& vy, float deltaTime) const;
};

// 增强的目标跟踪器
class TargetTracker {
private:
    std::map<int, TrackedTarget> trackedTargets;  // persistentId -> TrackedTarget
    int nextPersistentId;                          // 下一个持久化ID
    int currentPersistentId;                       // 当前锁定的目标
    
    // 配置参数
    float iouThreshold;           // IOU匹配阈值
    float distanceThreshold;      // 距离匹配阈值（像素）
    int maxLostFrames;           // 最大允许丢失帧数
    float kalmanProcessNoise;    // 卡尔曼过程噪声
    float kalmanMeasurementNoise; // 卡尔曼测量噪声
    
public:
    TargetTracker();
    
    // 更新跟踪器（每帧调用）
    void update(const std::vector<Detection>& detections, float deltaTime, int frameWidth, int frameHeight);
    
    // 获取当前锁定的目标
    TrackedTarget* getLockedTarget();
    
    // 锁定指定目标
    void lockTarget(int persistentId);
    
    // 解锁当前目标
    void unlock();
    
    // 获取最佳目标（用于自动选择）
    TrackedTarget* getBestTarget(int frameWidth, int frameHeight, int fovCenterX, int fovCenterY, float fovRadius);
    
    // 获取所有跟踪目标
    const std::map<int, TrackedTarget>& getAllTargets() const { return trackedTargets; }
    
    // 获取当前锁定的持久化ID
    int getCurrentPersistentId() const { return currentPersistentId; }
    
    // 设置配置参数
    void setIOUThreshold(float threshold) { iouThreshold = threshold; }
    void setDistanceThreshold(float threshold) { distanceThreshold = threshold; }
    void setMaxLostFrames(int frames) { maxLostFrames = frames; }
    void setKalmanProcessNoise(float noise) { kalmanProcessNoise = noise; }
    void setKalmanMeasurementNoise(float noise) { kalmanMeasurementNoise = noise; }
    
    // 重置跟踪器
    void reset();
    
private:
    // 数据关联：将检测与现有跟踪目标匹配
    void associateDetections(const std::vector<Detection>& detections,
                            std::vector<int>& matchedDetectionIndices,
                            std::vector<int>& matchedTargetIds,
                            std::vector<int>& unmatchedDetectionIndices,
                            int frameWidth, int frameHeight, float deltaTime);

    // 辅助函数：检查目标是否已匹配
    static bool isTargetMatched(const std::unordered_map<int, bool>& targetMatched, int targetId);

    // 创建新目标
    int createNewTarget(const Detection& det);

    // 清理丢失太久的目标
    void cleanupLostTargets();
};

#endif // TARGET_TRACKER_HPP
