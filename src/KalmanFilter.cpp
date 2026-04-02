#ifdef _WIN32

#include "KalmanFilter.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>

KalmanFilter::KalmanFilter()
    : dt(0.016f)
    , baseProcessNoise(0.01f)
    , baseMeasurementNoise(1.0f)
    , confidenceScale(1.0f)
{
    // 初始化所有矩阵为零
    std::memset(state, 0, sizeof(state));
    std::memset(covariance, 0, sizeof(covariance));
    std::memset(processNoise, 0, sizeof(processNoise));
    std::memset(measurementNoise, 0, sizeof(measurementNoise));
    std::memset(stateTransition, 0, sizeof(stateTransition));
    std::memset(measurementMatrix, 0, sizeof(measurementMatrix));

    // 初始化矩阵
    initializeMatrices();
}

void KalmanFilter::initializeMatrices()
{
    // 状态转移矩阵 F (恒定速度模型)
    // [1, 0, dt, 0 ]
    // [0, 1, 0,  dt]
    // [0, 0, 1,  0 ]
    // [0, 0, 0,  1 ]
    stateTransition[0][0] = 1.0f;
    stateTransition[0][2] = dt;
    stateTransition[1][1] = 1.0f;
    stateTransition[1][3] = dt;
    stateTransition[2][2] = 1.0f;
    stateTransition[3][3] = 1.0f;

    // 观测矩阵 H (只观测位置)
    // [1, 0, 0, 0]
    // [0, 1, 0, 0]
    measurementMatrix[0][0] = 1.0f;
    measurementMatrix[1][1] = 1.0f;

    // 过程噪声协方差 Q
    // 对角矩阵，速度和位置有各自的噪声
    processNoise[0][0] = baseProcessNoise * 0.1f;
    processNoise[1][1] = baseProcessNoise * 0.1f;
    processNoise[2][2] = baseProcessNoise;
    processNoise[3][3] = baseProcessNoise;

    // 测量噪声协方差 R
    measurementNoise[0][0] = baseMeasurementNoise;
    measurementNoise[1][1] = baseMeasurementNoise;

    // 初始协方差矩阵 P (较大的初始不确定性)
    for (int i = 0; i < 4; i++) {
        covariance[i][i] = 100.0f;
    }
}

void KalmanFilter::init(float x, float y)
{
    // 初始化状态向量
    state[0] = x;
    state[1] = y;
    state[2] = 0.0f;  // 初始速度为0
    state[3] = 0.0f;

    // 重置协方差矩阵
    std::memset(covariance, 0, sizeof(covariance));
    for (int i = 0; i < 4; i++) {
        covariance[i][i] = 100.0f;
    }

    // 重新初始化矩阵
    initializeMatrices();
}

void KalmanFilter::predict(float deltaTime)
{
    dt = deltaTime;

    // 更新状态转移矩阵中的时间项
    stateTransition[0][2] = dt;
    stateTransition[1][3] = dt;

    // 状态预测: x_pred = F * x
    float newState[4];
    newState[0] = stateTransition[0][0] * state[0] + stateTransition[0][2] * state[2];
    newState[1] = stateTransition[1][1] * state[1] + stateTransition[1][3] * state[3];
    newState[2] = state[2];
    newState[3] = state[3];
    std::memcpy(state, newState, sizeof(state));

    // 协方差预测: P_pred = F * P * F^T + Q
    float temp[4][4];
    float ft[4][4];

    // 计算 F * P
    matMul4x4(stateTransition, covariance, temp);

    // 计算 F^T
    matTranspose4x4(stateTransition, ft);

    // 计算 F * P * F^T
    matMul4x4(temp, ft, covariance);

    // 加上过程噪声 Q
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            covariance[i][j] += processNoise[i][j];
        }
    }
}

void KalmanFilter::update(float measuredX, float measuredY, float confidence)
{
    // 根据置信度调整测量噪声
    // 置信度越高，测量噪声越小
    float adjustedR = baseMeasurementNoise / (confidenceScale * confidence + 0.1f);
    float currentMeasurementNoise[2][2];
    currentMeasurementNoise[0][0] = adjustedR;
    currentMeasurementNoise[0][1] = 0.0f;
    currentMeasurementNoise[1][0] = 0.0f;
    currentMeasurementNoise[1][1] = adjustedR;

    // 计算 S = H * P * H^T + R
    float ht[4][2];
    float ph[4][2];
    float s[2][2];

    // H^T
    matTranspose2x4(measurementMatrix, ht);

    // P * H^T
    matMul4x4_4x2(covariance, ht, ph);

    // H * P * H^T
    matMul2x4_4x2(measurementMatrix, ph, s);

    // 加上测量噪声 R
    s[0][0] += currentMeasurementNoise[0][0];
    s[1][1] += currentMeasurementNoise[1][1];

    // 计算 S 的逆矩阵
    float sInv[2][2];
    if (!matInverse2x2(s, sInv)) {
        // 矩阵不可逆，跳过更新
        return;
    }

    // 计算卡尔曼增益 K = P * H^T * S^(-1)
    float gain[4][2];
    matMul4x4_4x2(covariance, ht, ph);  // P * H^T

    // (P * H^T) * S^(-1)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            gain[i][j] = ph[i][0] * sInv[0][j] + ph[i][1] * sInv[1][j];
        }
    }

    // 计算观测残差 y = z - H * x
    float residual[2];
    residual[0] = measuredX - (measurementMatrix[0][0] * state[0] + measurementMatrix[0][1] * state[1]);
    residual[1] = measuredY - (measurementMatrix[1][0] * state[0] + measurementMatrix[1][1] * state[1]);

    // 状态更新 x = x + K * y
    for (int i = 0; i < 4; i++) {
        state[i] += gain[i][0] * residual[0] + gain[i][1] * residual[1];
    }

    // 协方差更新 P = (I - K * H) * P
    float kh[4][4];
    float identity[4][4];

    // 初始化单位矩阵
    std::memset(identity, 0, sizeof(identity));
    for (int i = 0; i < 4; i++) {
        identity[i][i] = 1.0f;
    }

    // K * H
    std::memset(kh, 0, sizeof(kh));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            kh[i][j] = gain[i][0] * measurementMatrix[0][j] + gain[i][1] * measurementMatrix[1][j];
        }
    }

    // I - K * H
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            kh[i][j] = identity[i][j] - kh[i][j];
        }
    }

    // (I - K * H) * P
    float newCovariance[4][4];
    matMul4x4(kh, covariance, newCovariance);
    std::memcpy(covariance, newCovariance, sizeof(covariance));
}

void KalmanFilter::getPrediction(float predictionTime, float& predX, float& predY)
{
    // 基于当前状态进行预测
    // x_pred = x + vx * t
    // y_pred = y + vy * t
    predX = state[0] + state[2] * predictionTime;
    predY = state[1] + state[3] * predictionTime;
}

void KalmanFilter::getPredictionOffset(float predictionTime, float currentX, float currentY, float& offsetX, float& offsetY)
{
    // 获取绝对坐标的预测位置
    float predX, predY;
    getPrediction(predictionTime, predX, predY);

    // 转换为相对于当前位置的偏移量
    offsetX = predX - currentX;
    offsetY = predY - currentY;
}

void KalmanFilter::getState(float& x, float& y, float& vx, float& vy)
{
    x = state[0];
    y = state[1];
    vx = state[2];
    vy = state[3];
}

void KalmanFilter::getPosition(float& x, float& y)
{
    x = state[0];
    y = state[1];
}

void KalmanFilter::getVelocity(float& vx, float& vy)
{
    vx = state[2];
    vy = state[3];
}

void KalmanFilter::reset()
{
    std::memset(state, 0, sizeof(state));
    std::memset(covariance, 0, sizeof(covariance));

    for (int i = 0; i < 4; i++) {
        covariance[i][i] = 100.0f;
    }

    initializeMatrices();
}

void KalmanFilter::setProcessNoise(float q)
{
    baseProcessNoise = std::max(0.001f, q);

    // 更新过程噪声矩阵
    processNoise[0][0] = baseProcessNoise * 0.1f;
    processNoise[1][1] = baseProcessNoise * 0.1f;
    processNoise[2][2] = baseProcessNoise;
    processNoise[3][3] = baseProcessNoise;
}

void KalmanFilter::setMeasurementNoise(float r)
{
    baseMeasurementNoise = std::max(0.1f, r);

    // 更新测量噪声矩阵
    measurementNoise[0][0] = baseMeasurementNoise;
    measurementNoise[1][1] = baseMeasurementNoise;
}

void KalmanFilter::setConfidenceScale(float scale)
{
    confidenceScale = std::max(0.1f, scale);
}

bool KalmanFilter::isInitialized() const
{
    // 检查协方差矩阵是否还是初始值
    return covariance[0][0] < 99.0f;
}

// 矩阵运算辅助函数实现

void KalmanFilter::matMul4x4(const float a[4][4], const float b[4][4], float result[4][4])
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void KalmanFilter::matMul4x4_4x2(const float a[4][4], const float b[4][2], float result[4][2])
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            result[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void KalmanFilter::matMul2x4_4x4(const float a[2][4], const float b[4][4], float result[2][4])
{
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            result[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void KalmanFilter::matMul2x4_4x2(const float a[2][4], const float b[4][2], float result[2][2])
{
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void KalmanFilter::matTranspose4x4(const float mat[4][4], float result[4][4])
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[j][i] = mat[i][j];
        }
    }
}

void KalmanFilter::matTranspose4x2(const float mat[4][2], float result[2][4])
{
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            result[j][i] = mat[i][j];
        }
    }
}

void KalmanFilter::matTranspose2x4(const float mat[2][4], float result[4][2])
{
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            result[j][i] = mat[i][j];
        }
    }
}

bool KalmanFilter::matInverse2x2(const float mat[2][2], float result[2][2])
{
    float det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];

    if (std::abs(det) < 1e-10f) {
        return false;  // 矩阵奇异，不可逆
    }

    float invDet = 1.0f / det;
    result[0][0] = mat[1][1] * invDet;
    result[0][1] = -mat[0][1] * invDet;
    result[1][0] = -mat[1][0] * invDet;
    result[1][1] = mat[0][0] * invDet;

    return true;
}

#endif // _WIN32
