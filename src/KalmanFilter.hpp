#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <Eigen/Dense>

class KalmanFilter {
private:
    // 状态向量 [x, y, vx, vy, ax, ay] - 增强型6维状态
    Eigen::Vector6f state;

    // 误差协方差矩阵 (6x6)
    Eigen::Matrix6f covariance;

    // 过程噪声协方差
    Eigen::Matrix6f processNoise;

    // 测量噪声协方差
    Eigen::Matrix2f measurementNoise;

    // 状态转移矩阵
    Eigen::Matrix6f stateTransition;

    // 观测矩阵
    Eigen::Matrix<float, 2, 6> measurementMatrix;

    // 时间步长
    float dt;

    // 自适应参数
    float baseProcessNoise;
    float baseMeasurementNoise;
    float confidenceScale;

    // 目标ID管理
    static int nextId;
    int id;

    // 目标丢失追踪
    int lostFrameCount;
    int maxLostFrames;

    // 初始化矩阵
    void initializeMatrices();

    // 更新状态转移矩阵（根据dt）
    void updateStateTransition(float deltaTime);

public:
    KalmanFilter();

    // 初始化滤波器
    void init(float x, float y);

    // 预测步骤
    void predict(float deltaTime);

    // 更新步骤
    void update(float measuredX, float measuredY, float confidence = 1.0f);

    // 标记目标丢失（无测量时调用）
    void markLost();

    // 标记目标检测到
    void markDetected();

    // 检查目标是否仍然存活
    bool isAlive() const;

    // 获取丢失帧数
    int getLostFrameCount() const;

    // 设置最大丢失帧数
    void setMaxLostFrames(int maxFrames);

    // 获取目标ID
    int getId() const;

    // 获取预测位置
    void getPrediction(float predictionTime, float& predX, float& predY) const;

    // 获取预测偏移量（相对于当前测量位置的偏移，可直接用于PID误差融合）
    void getPredictionOffset(float predictionTime, float currentX, float currentY, float& offsetX, float& offsetY) const;

    // 获取当前状态估计
    void getState(float& x, float& y, float& vx, float& vy);
    void getStateFull(float& x, float& y, float& vx, float& vy, float& ax, float& ay);

    // 获取当前位置
    void getPosition(float& x, float& y);

    // 获取当前速度
    void getVelocity(float& vx, float& vy);

    // 获取当前加速度
    void getAcceleration(float& ax, float& ay);

    // 重置滤波器
    void reset();

    // 设置噪声参数
    void setProcessNoise(float q);
    void setMeasurementNoise(float r);
    void setConfidenceScale(float scale);

    // 自适应噪声调整（根据速度和置信度）
    void adaptNoise(float velocity, float confidence);

    // 获取噪声参数
    float getProcessNoise() const { return baseProcessNoise; }
    float getMeasurementNoise() const { return baseMeasurementNoise; }
    float getConfidenceScale() const { return confidenceScale; }

    // 检查滤波器是否已初始化
    bool isInitialized() const;
};

#endif // KALMAN_FILTER_HPP
