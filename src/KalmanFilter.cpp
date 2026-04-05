#include "KalmanFilter.hpp"
#include <cmath>
#include <obs-module.h>
#include <plugin-support.h>

// 静态ID计数器初始化
int KalmanFilter::nextId = 0;

KalmanFilter::KalmanFilter()
    : state(Eigen::Vector4f::Zero())
    , covariance(Eigen::Matrix4f::Identity() * 100.0f)
    , processNoise(Eigen::Matrix4f::Zero())
    , measurementNoise(Eigen::Matrix2f::Identity())
    , stateTransition(Eigen::Matrix4f::Identity())
    , measurementMatrix(Eigen::Matrix<float, 2, 4>::Zero())
    , dt(0.016f)
    , baseProcessNoise(0.01f)
    , baseMeasurementNoise(1.0f)
    , confidenceScale(1.0f)
    , id(nextId++)
    , lostFrameCount(0)
    , maxLostFrames(3)
{
    initializeMatrices();
}

void KalmanFilter::initializeMatrices()
{
    // 状态转移矩阵 F (恒定速度模型)
    // [1  0  dt 0 ]
    // [0  1  0  dt]
    // [0  0  1  0 ]
    // [0  0  0  1 ]
    stateTransition = Eigen::Matrix4f::Identity();
    stateTransition(0, 2) = dt;
    stateTransition(1, 3) = dt;

    // 观测矩阵 H (只观测位置)
    // [1  0  0  0]
    // [0  1  0  0]
    measurementMatrix.setZero();
    measurementMatrix(0, 0) = 1.0f;
    measurementMatrix(1, 1) = 1.0f;

    // 过程噪声协方差 Q
    processNoise.setZero();
    processNoise(0, 0) = baseProcessNoise * 0.1f;
    processNoise(1, 1) = baseProcessNoise * 0.1f;
    processNoise(2, 2) = baseProcessNoise;
    processNoise(3, 3) = baseProcessNoise;

    // 测量噪声协方差 R
    measurementNoise = Eigen::Matrix2f::Identity() * baseMeasurementNoise;
}

void KalmanFilter::init(float x, float y)
{
    // 初始化状态向量
    state << x, y, 0.0f, 0.0f;

    // 重置协方差矩阵
    covariance = Eigen::Matrix4f::Identity() * 100.0f;

    // 重新初始化矩阵
    initializeMatrices();

    // 重置丢失计数
    lostFrameCount = 0;
}

void KalmanFilter::predict(float deltaTime)
{
    dt = deltaTime;

    // 更新状态转移矩阵中的时间项
    stateTransition(0, 2) = dt;
    stateTransition(1, 3) = dt;

    // 状态预测: x_pred = F * x
    state = stateTransition * state;

    // 协方差预测: P_pred = F * P * F^T + Q
    covariance = stateTransition * covariance * stateTransition.transpose() + processNoise;
}

void KalmanFilter::update(float measuredX, float measuredY, float confidence)
{
    // 标记目标检测到
    markDetected();

    // 根据置信度调整测量噪声
    float adjustedR = baseMeasurementNoise / (confidenceScale * confidence + 0.1f);
    Eigen::Matrix2f currentMeasurementNoise = Eigen::Matrix2f::Identity() * adjustedR;

    // 计算卡尔曼增益 K = P * H^T * (H * P * H^T + R)^(-1)
    Eigen::Matrix2f s = measurementMatrix * covariance * measurementMatrix.transpose() + currentMeasurementNoise;
    
    // 检查矩阵是否可逆（行列式不能太小）
    float det = s.determinant();
    if (std::abs(det) < 1e-6f) {
        obs_log(LOG_WARNING, "[KalmanFilter] Matrix is nearly singular (det=%.2e), skipping update", det);
        return;
    }
    
    Eigen::Matrix<float, 4, 2> gain = covariance * measurementMatrix.transpose() * s.inverse();

    // 计算观测残差 y = z - H * x
    Eigen::Vector2f residual;
    residual << measuredX, measuredY;
    residual -= measurementMatrix * state;

    // 状态更新 x = x + K * y
    state += gain * residual;

    // 协方差更新 P = (I - K * H) * P
    Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
    covariance = (identity - gain * measurementMatrix) * covariance;
}

void KalmanFilter::markLost()
{
    lostFrameCount++;
}

void KalmanFilter::markDetected()
{
    lostFrameCount = 0;
}

bool KalmanFilter::isAlive() const
{
    return lostFrameCount < maxLostFrames;
}

int KalmanFilter::getLostFrameCount() const
{
    return lostFrameCount;
}

void KalmanFilter::setMaxLostFrames(int maxFrames)
{
    maxLostFrames = std::max(1, maxFrames);
}

int KalmanFilter::getId() const
{
    return id;
}

void KalmanFilter::getPrediction(float predictionTime, float& predX, float& predY) const
{
    // 基于当前状态进行预测
    // x_pred = x + vx * t
    // y_pred = y + vy * t
    predX = state(0) + state(2) * predictionTime;
    predY = state(1) + state(3) * predictionTime;
}

void KalmanFilter::getPredictionOffset(float predictionTime, float currentX, float currentY, float& offsetX, float& offsetY) const
{
    float predX, predY;
    getPrediction(predictionTime, predX, predY);
    
    // 计算预测位置相对于当前测量位置的偏移
    offsetX = predX - currentX;
    offsetY = predY - currentY;
}

void KalmanFilter::getState(float& x, float& y, float& vx, float& vy)
{
    x = state(0);
    y = state(1);
    vx = state(2);
    vy = state(3);
}

void KalmanFilter::getPosition(float& x, float& y)
{
    x = state(0);
    y = state(1);
}

void KalmanFilter::getVelocity(float& vx, float& vy)
{
    vx = state(2);
    vy = state(3);
}

void KalmanFilter::reset()
{
    state = Eigen::Vector4f::Zero();
    covariance = Eigen::Matrix4f::Identity() * 100.0f;
    initializeMatrices();
    lostFrameCount = 0;
    // 注意：reset不重置ID，ID是永久分配的
}

void KalmanFilter::setProcessNoise(float q)
{
    baseProcessNoise = std::max(0.001f, q);
    processNoise.setZero();
    processNoise(0, 0) = baseProcessNoise * 0.1f;
    processNoise(1, 1) = baseProcessNoise * 0.1f;
    processNoise(2, 2) = baseProcessNoise;
    processNoise(3, 3) = baseProcessNoise;
}

void KalmanFilter::setMeasurementNoise(float r)
{
    baseMeasurementNoise = std::max(0.001f, r);
    measurementNoise = Eigen::Matrix2f::Identity() * baseMeasurementNoise;
}

void KalmanFilter::setConfidenceScale(float scale)
{
    confidenceScale = std::max(0.1f, scale);
}

bool KalmanFilter::isInitialized() const
{
    // 检查协方差矩阵是否还是初始值
    return covariance(0, 0) < 99.0f;
}
