#include "KalmanFilter.hpp"
#include <cmath>
#include <algorithm>

int KalmanFilter::nextId = 0;

KalmanFilter::KalmanFilter()
    : state(Eigen::Vector6f::Zero())
    , covariance(Eigen::Matrix6f::Identity() * 100.0f)
    , processNoise(Eigen::Matrix6f::Identity())
    , measurementNoise(Eigen::Matrix2f::Identity())
    , stateTransition(Eigen::Matrix6f::Identity())
    , measurementMatrix(Eigen::Matrix<float, 2, 6>::Zero())
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
    // 状态转移矩阵 F (恒定速度+加速度模型)
    // 状态向量: [x, y, vx, vy, ax, ay]
    // x' = x + vx*dt + 0.5*ax*dt^2
    // y' = y + vy*dt + 0.5*ay*dt^2
    // vx' = vx + ax*dt
    // vy' = vy + ay*dt
    // ax' = ax
    // ay' = ay
    stateTransition = Eigen::Matrix6f::Identity();
    // 位置更新
    stateTransition(0, 2) = dt;   // x += vx*dt
    stateTransition(1, 3) = dt;   // y += vy*dt
    stateTransition(0, 4) = 0.5f * dt * dt;  // x += 0.5*ax*dt^2
    stateTransition(1, 5) = 0.5f * dt * dt;  // y += 0.5*ay*dt^2
    stateTransition(2, 4) = dt;   // vx += ax*dt
    stateTransition(3, 5) = dt;   // vy += ay*dt

    // 观测矩阵 H (只观测位置)
    // [1  0  0  0  0  0]
    // [0  1  0  0  0  0]
    measurementMatrix.setZero();
    measurementMatrix(0, 0) = 1.0f;
    measurementMatrix(1, 1) = 1.0f;

    // 过程噪声协方差 Q
    // 对位置、速度、加速度设置不同的噪声
    processNoise = Eigen::Matrix6f::Identity();
    processNoise(0, 0) = baseProcessNoise;       // x
    processNoise(1, 1) = baseProcessNoise;       // y
    processNoise(2, 2) = baseProcessNoise * 0.1f; // vx
    processNoise(3, 3) = baseProcessNoise * 0.1f; // vy
    processNoise(4, 4) = baseProcessNoise * 0.01f; // ax
    processNoise(5, 5) = baseProcessNoise * 0.01f; // ay

    // 测量噪声协方差 R
    measurementNoise = Eigen::Matrix2f::Identity() * baseMeasurementNoise;
}

void KalmanFilter::updateStateTransition(float deltaTime)
{
    dt = deltaTime;
    
    // 更新状态转移矩阵中的时间相关项
    stateTransition(0, 2) = dt;
    stateTransition(1, 3) = dt;
    stateTransition(0, 4) = 0.5f * dt * dt;
    stateTransition(1, 5) = 0.5f * dt * dt;
    stateTransition(2, 4) = dt;
    stateTransition(3, 5) = dt;
}

void KalmanFilter::init(float x, float y)
{
    // 初始化状态向量 [x, y, vx, vy, ax, ay]
    state << x, y, 0.0f, 0.0f, 0.0f, 0.0f;

    // 重置协方差矩阵
    covariance = Eigen::Matrix6f::Identity() * 100.0f;

    // 重新初始化矩阵
    initializeMatrices();

    // 重置丢失计数
    lostFrameCount = 0;
}

void KalmanFilter::predict(float deltaTime)
{
    // 更新状态转移矩阵
    updateStateTransition(deltaTime);

    // 状态预测: x_pred = F * x
    state = stateTransition * state;

    // 协方差预测: P_pred = F * P * F^T + Q
    covariance = stateTransition * covariance * stateTransition.transpose() + processNoise;
}

void KalmanFilter::update(float measuredX, float measuredY, float confidence)
{
    // 标记目标检测到
    markDetected();

    // 自适应测量噪声（根据置信度）
    float adaptedR = baseMeasurementNoise / std::max(0.1f, confidence * confidenceScale);
    measurementNoise(0, 0) = adaptedR;
    measurementNoise(1, 1) = adaptedR;

    // 测量向量
    Eigen::Vector2f measurement;
    measurement << measuredX, measuredY;

    // 计算残差: y - H * x
    Eigen::Vector2f residual = measurement - measurementMatrix * state;

    // 计算残差协方差: S = H * P * H^T + R
    Eigen::Matrix2f S = measurementMatrix * covariance * measurementMatrix.transpose() + measurementNoise;

    // 计算卡尔曼增益: K = P * H^T * S^(-1)
    Eigen::Matrix<float, 6, 2> gain = covariance * measurementMatrix.transpose() * S.inverse();

    // 状态更新 x = x + K * y
    state += gain * residual;

    // 协方差更新 P = (I - K * H) * P
    Eigen::Matrix6f identity = Eigen::Matrix6f::Identity();
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
    // 使用当前状态进行预测
    // x_pred = x + vx*t + 0.5*ax*t^2
    // y_pred = y + vy*t + 0.5*ay*t^2
    predX = state(0) + state(2) * predictionTime + 0.5f * state(4) * predictionTime * predictionTime;
    predY = state(1) + state(3) * predictionTime + 0.5f * state(5) * predictionTime * predictionTime;
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

void KalmanFilter::getStateFull(float& x, float& y, float& vx, float& vy, float& ax, float& ay)
{
    x = state(0);
    y = state(1);
    vx = state(2);
    vy = state(3);
    ax = state(4);
    ay = state(5);
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

void KalmanFilter::getAcceleration(float& ax, float& ay)
{
    ax = state(4);
    ay = state(5);
}

void KalmanFilter::reset()
{
    state = Eigen::Vector6f::Zero();
    covariance = Eigen::Matrix6f::Identity() * 100.0f;
    initializeMatrices();
    lostFrameCount = 0;
}

void KalmanFilter::setProcessNoise(float q)
{
    baseProcessNoise = q;
    // 设置对角过程噪声
    processNoise(0, 0) = q;
    processNoise(1, 1) = q;
    processNoise(2, 2) = q * 0.1f;
    processNoise(3, 3) = q * 0.1f;
    processNoise(4, 4) = q * 0.01f;
    processNoise(5, 5) = q * 0.01f;
}

void KalmanFilter::setMeasurementNoise(float r)
{
    baseMeasurementNoise = r;
    measurementNoise(0, 0) = r;
    measurementNoise(1, 1) = r;
}

void KalmanFilter::setConfidenceScale(float scale)
{
    confidenceScale = scale;
}

void KalmanFilter::adaptNoise(float velocity, float confidence)
{
    // 根据速度自适应调整过程噪声
    // 速度越快，过程噪声越大（目标运动越不确定）
    float velocityFactor = 1.0f + std::min(velocity * 0.1f, 2.0f);
    
    // 根据置信度自适应调整测量噪声
    // 置信度越低，测量噪声越大（检测越不可靠）
    float confidenceFactor = 1.0f / std::max(confidence, 0.1f);
    
    // 更新过程噪声
    float adaptedQ = baseProcessNoise * velocityFactor;
    processNoise(0, 0) = adaptedQ;
    processNoise(1, 1) = adaptedQ;
    processNoise(2, 2) = adaptedQ * 0.1f;
    processNoise(3, 3) = adaptedQ * 0.1f;
    processNoise(4, 4) = adaptedQ * 0.01f;
    processNoise(5, 5) = adaptedQ * 0.01f;
    
    // 更新测量噪声
    float adaptedR = baseMeasurementNoise * confidenceFactor / confidenceScale;
    measurementNoise(0, 0) = adaptedR;
    measurementNoise(1, 1) = adaptedR;
}

bool KalmanFilter::isInitialized() const
{
    return state(0) != 0.0f || state(1) != 0.0f;
}
