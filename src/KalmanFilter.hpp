#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <Eigen/Dense>

class KalmanFilter {
private:
    // 状态向量 [x, y, vx, vy]
    Eigen::Vector4f state;

    // 误差协方差矩阵 (4x4)
    Eigen::Matrix4f covariance;

    // 过程噪声协方差
    Eigen::Matrix4f processNoise;

    // 测量噪声协方差
    Eigen::Matrix2f measurementNoise;

    // 状态转移矩阵
    Eigen::Matrix4f stateTransition;

    // 观测矩阵
    Eigen::Matrix<float, 2, 4> measurementMatrix;

    // 时间步长
    float dt;

    // 自适应参数
    float baseProcessNoise;
    float baseMeasurementNoise;
    float confidenceScale;

    // 初始化矩阵
    void initializeMatrices();

public:
    KalmanFilter();

    // 初始化滤波器
    void init(float x, float y);

    // 预测步骤
    void predict(float deltaTime);

    // 更新步骤
    void update(float measuredX, float measuredY, float confidence = 1.0f);

    // 获取预测位置
    void getPrediction(float predictionTime, float& predX, float& predY) const;

    // 获取预测偏移量（相对于当前测量位置的偏移，可直接用于PID误差融合）
    void getPredictionOffset(float predictionTime, float currentX, float currentY, float& offsetX, float& offsetY);

    // 获取当前状态估计
    void getState(float& x, float& y, float& vx, float& vy);

    // 获取当前位置
    void getPosition(float& x, float& y);

    // 获取当前速度
    void getVelocity(float& vx, float& vy);

    // 重置滤波器
    void reset();

    // 设置噪声参数
    void setProcessNoise(float q);
    void setMeasurementNoise(float r);
    void setConfidenceScale(float scale);

    // 获取噪声参数
    float getProcessNoise() const { return baseProcessNoise; }
    float getMeasurementNoise() const { return baseMeasurementNoise; }
    float getConfidenceScale() const { return confidenceScale; }

    // 检查滤波器是否已初始化
    bool isInitialized() const;
};

#endif // KALMAN_FILTER_HPP
