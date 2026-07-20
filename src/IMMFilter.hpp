#ifndef IMM_FILTER_HPP
#define IMM_FILTER_HPP

#include <cmath>
#include <algorithm>
#include <array>

class IMMFilter {
public:
    struct Config {
        float processNoisePos = 0.1f;
        float processNoiseVel = 0.5f;
        float processNoiseAcc = 1.0f;
        float processNoiseTurn = 0.1f;
        float measurementNoiseX = 1.0f;
        float measurementNoiseY = 1.0f;
        float dt = 0.016f;
        bool enabled = true;
        size_t activeModels = 3;
    };

    IMMFilter();
    void reset();
    void setConfig(const Config& cfg);
    void predict(float dt);
    void update(float measuredX, float measuredY);
    void getState(float& estX, float& estY, float& velX, float& velY);
    void getPrediction(float predictDt, float& predX, float& predY);

private:
    Config cfg_;

    static constexpr size_t MODEL_CV = 0;
    static constexpr size_t MODEL_CA = 1;
    static constexpr size_t MODEL_CT = 2;
    static constexpr size_t NUM_MODELS = 3;
    static constexpr float TINY = 1e-6f;

    void predictCV(float dt);
    void updateCV(float mx, float my);
    void predictCA(float dt);
    void updateCA(float mx, float my);
    void predictCT(float dt);
    void updateCT(float mx, float my);
    void interact(float dt);
    void updateModelProbabilities(float mx, float my);
    void combine();

    float xCV_[4];
    float PxCV_[4][4];
    float xCA_[6];
    float PxCA_[6][6];
    float xCT_[5];
    float PxCT_[5][5];

    float probCV_;
    float probCA_;
    float probCT_;

    float transMatrix_[3][3];

    bool initialized_;

    static constexpr size_t CX = 4;
    static constexpr size_t CAX = 6;
    static constexpr size_t CTX = 5;
};

inline IMMFilter::IMMFilter()
    : probCV_(1.0f / 3.0f)
    , probCA_(1.0f / 3.0f)
    , probCT_(1.0f / 3.0f)
    , initialized_(false)
{
    transMatrix_[0][0] = 0.90f; transMatrix_[0][1] = 0.05f; transMatrix_[0][2] = 0.05f;
    transMatrix_[1][0] = 0.10f; transMatrix_[1][1] = 0.85f; transMatrix_[1][2] = 0.05f;
    transMatrix_[2][0] = 0.10f; transMatrix_[2][1] = 0.05f; transMatrix_[2][2] = 0.85f;
    reset();
}

inline void IMMFilter::reset()
{
    for (size_t i = 0; i < CX; i++) xCV_[i] = 0.0f;
    for (size_t i = 0; i < CX; i++)
        for (size_t j = 0; j < CX; j++)
            PxCV_[i][j] = (i == j) ? 1000.0f : 0.0f;

    for (size_t i = 0; i < CAX; i++) xCA_[i] = 0.0f;
    for (size_t i = 0; i < CAX; i++)
        for (size_t j = 0; j < CAX; j++)
            PxCA_[i][j] = (i == j) ? 1000.0f : 0.0f;

    for (size_t i = 0; i < CTX; i++) xCT_[i] = 0.0f;
    for (size_t i = 0; i < CTX; i++)
        for (size_t j = 0; j < CTX; j++)
            PxCT_[i][j] = (i == j) ? 1000.0f : 0.0f;

    probCV_ = 1.0f / 3.0f;
    probCA_ = 1.0f / 3.0f;
    probCT_ = 1.0f / 3.0f;
    initialized_ = false;
}

inline void IMMFilter::setConfig(const Config& cfg)
{
    cfg_ = cfg;
}

inline void IMMFilter::predictCV(float dt)
{
    float F[CX][CX] = {
        {1, dt, 0,  0},
        {0,  1, 0,  0},
        {0,  0, 1, dt},
        {0,  0, 0,  1}
    };

    float Qpos = cfg_.processNoisePos;
    float Qvel = cfg_.processNoiseVel;
    float Q[CX][CX] = {};
    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    Q[0][0] = Qpos * dt3 / 3; Q[0][1] = Qpos * dt2 / 2;
    Q[1][0] = Qpos * dt2 / 2; Q[1][1] = Qpos * dt;
    Q[2][2] = Qpos * dt3 / 3; Q[2][3] = Qpos * dt2 / 2;
    Q[3][2] = Qpos * dt2 / 2; Q[3][3] = Qpos * dt;

    float xPred[CX];
    float Pp[CX][CX];
    for (size_t i = 0; i < CX; i++) {
        xPred[i] = 0;
        for (size_t j = 0; j < CX; j++) xPred[i] += F[i][j] * xCV_[j];
    }
    for (size_t i = 0; i < CX; i++) {
        for (size_t j = 0; j < CX; j++) {
            float sum = 0;
            for (size_t k = 0; k < CX; k++) sum += F[i][k] * PxCV_[k][j];
            sum += Q[i][j];
            float sum2 = 0;
            for (size_t k = 0; k < CX; k++) sum2 += sum * F[j][k];
            Pp[i][j] = sum2;
        }
    }
    for (size_t i = 0; i < CX; i++) xCV_[i] = xPred[i];
    for (size_t i = 0; i < CX; i++)
        for (size_t j = 0; j < CX; j++)
            PxCV_[i][j] = Pp[i][j];
}

inline void IMMFilter::updateCV(float mx, float my)
{
    float meas[2] = {mx, my};
    float Hx = 1, Hy = 0;
    float Sx = PxCV_[0][0] + cfg_.measurementNoiseX;
    float Kx = PxCV_[0][0] / std::max(Sx, TINY);
    float Sy = PxCV_[2][2] + cfg_.measurementNoiseY;
    float Ky = PxCV_[2][2] / std::max(Sy, TINY);
    xCV_[0] += Kx * (meas[0] - xCV_[0]);
    xCV_[1] += (PxCV_[1][0] / std::max(Sx, TINY)) * (meas[0] - xCV_[0]);
    xCV_[2] += Ky * (meas[1] - xCV_[2]);
    xCV_[3] += (PxCV_[3][2] / std::max(Sy, TINY)) * (meas[1] - xCV_[2]);
    PxCV_[0][0] = (1 - Kx) * PxCV_[0][0];
    PxCV_[2][2] = (1 - Ky) * PxCV_[2][2];
}

inline void IMMFilter::predictCA(float dt)
{
    float halfDt2 = dt * dt * 0.5f;
    float F[CAX][CAX] = {
        {1, dt, halfDt2, 0,  0,       0},
        {0,  1,      dt, 0,  0,       0},
        {0,  0,       1, 0,  0,       0},
        {0,  0,       0, 1, dt, halfDt2},
        {0,  0,       0, 0,  1,      dt},
        {0,  0,       0, 0,  0,       1}
    };

    float Qacc = cfg_.processNoiseAcc;
    float Q[CAX][CAX] = {};
    for (size_t d = 0; d < 2; d++) {
        size_t i0 = d * 3;
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 3; j++)
                Q[i0 + i][i0 + j] = Qacc * (i + j >= 2 ? 0.01f : 0.05f);
    }

    float xPred[CAX] = {};
    float Pp[CAX][CAX] = {};
    for (size_t i = 0; i < CAX; i++)
        for (size_t j = 0; j < CAX; j++) xPred[i] += F[i][j] * xCA_[j];
    for (size_t i = 0; i < CAX; i++)
        for (size_t j = 0; j < CAX; j++) {
            float sum = 0;
            for (size_t k = 0; k < CAX; k++) sum += F[i][k] * PxCA_[k][j];
            sum += Q[i][j];
            float sum2 = 0;
            for (size_t k = 0; k < CAX; k++) sum2 += sum * F[j][k];
            Pp[i][j] = sum2;
        }
    for (size_t i = 0; i < CAX; i++) xCA_[i] = xPred[i];
    for (size_t i = 0; i < CAX; i++)
        for (size_t j = 0; j < CAX; j++)
            PxCA_[i][j] = Pp[i][j];
}

inline void IMMFilter::updateCA(float mx, float my)
{
    float meas[2] = {mx, my};
    float Sx = PxCA_[0][0] + cfg_.measurementNoiseX;
    float Kx = PxCA_[0][0] / std::max(Sx, TINY);
    float Sy = PxCA_[3][3] + cfg_.measurementNoiseY;
    float Ky = PxCA_[3][3] / std::max(Sy, TINY);
    float innovX = meas[0] - xCA_[0];
    float innovY = meas[1] - xCA_[3];
    xCA_[0] += Kx * innovX;
    xCA_[1] += (PxCA_[1][0] / std::max(Sx, TINY)) * innovX;
    xCA_[2] += (PxCA_[2][0] / std::max(Sx, TINY)) * innovX;
    xCA_[3] += Ky * innovY;
    xCA_[4] += (PxCA_[4][3] / std::max(Sy, TINY)) * innovY;
    xCA_[5] += (PxCA_[5][3] / std::max(Sy, TINY)) * innovY;
    PxCA_[0][0] = (1 - Kx) * PxCA_[0][0];
    PxCA_[3][3] = (1 - Ky) * PxCA_[3][3];
}

inline void IMMFilter::predictCT(float dt)
{
    float w = xCT_[4];
    float absW = std::max(std::abs(w), TINY);
    float sinWT = std::sin(w * dt);
    float cosWT = std::cos(w * dt);
    float a = sinWT / absW;
    float b = (1 - cosWT) / absW;

    float F[CTX][CTX] = {
        {1, a, 0, -b, 0},
        {0, cosWT, 0, -sinWT, 0},
        {0, b, 1, a, 0},
        {0, sinWT, 0, cosWT, 0},
        {0, 0, 0, 0, 1}
    };

    float Qpos = cfg_.processNoisePos * dt;
    float Qtrn = cfg_.processNoiseTurn * dt;
    float Q[CTX][CTX] = {};
    Q[0][0] = Qpos; Q[1][1] = Qpos; Q[2][2] = Qpos; Q[3][3] = Qpos; Q[4][4] = Qtrn;

    float xPred[CTX] = {};
    float Pp[CTX][CTX] = {};
    for (size_t i = 0; i < CTX; i++)
        for (size_t j = 0; j < CTX; j++) xPred[i] += F[i][j] * xCT_[j];
    for (size_t i = 0; i < CTX; i++)
        for (size_t j = 0; j < CTX; j++) {
            float sum = 0;
            for (size_t k = 0; k < CTX; k++) sum += F[i][k] * PxCT_[k][j];
            sum += Q[i][j];
            float sum2 = 0;
            for (size_t k = 0; k < CTX; k++) sum2 += sum * F[j][k];
            Pp[i][j] = sum2;
        }
    for (size_t i = 0; i < CTX; i++) xCT_[i] = xPred[i];
    for (size_t i = 0; i < CTX; i++)
        for (size_t j = 0; j < CTX; j++)
            PxCT_[i][j] = Pp[i][j];
}

inline void IMMFilter::updateCT(float mx, float my)
{
    float meas[2] = {mx, my};
    float Sx = PxCT_[0][0] + cfg_.measurementNoiseX;
    float Kx = PxCT_[0][0] / std::max(Sx, TINY);
    float Sy = PxCT_[2][2] + cfg_.measurementNoiseY;
    float Ky = PxCT_[2][2] / std::max(Sy, TINY);
    float innovX = meas[0] - xCT_[0];
    float innovY = meas[1] - xCT_[2];
    xCT_[0] += Kx * innovX;
    xCT_[1] += (PxCT_[1][0] / std::max(Sx, TINY)) * innovX;
    xCT_[2] += Ky * innovY;
    xCT_[3] += (PxCT_[3][2] / std::max(Sy, TINY)) * innovY;
    PxCT_[0][0] = (1 - Kx) * PxCT_[0][0];
    PxCT_[2][2] = (1 - Ky) * PxCT_[2][2];
}

inline void IMMFilter::interact(float dt)
{
    float mixProb[3][3] = {};
    for (size_t j = 0; j < NUM_MODELS; j++) {
        float sum = 0;
        float probs[3] = {probCV_, probCA_, probCT_};
        for (size_t i = 0; i < NUM_MODELS; i++)
            sum += transMatrix_[i][j] * probs[i];
        for (size_t i = 0; i < NUM_MODELS; i++)
            mixProb[i][j] = (sum > TINY) ? transMatrix_[i][j] * probs[i] / sum : 0;
        if (j == 0) mixProb[0][j] = (sum > TINY) ? transMatrix_[0][0] * probCV_ / sum : 0;
    }

    float xMixed[3][4] = {};
    for (size_t j = 0; j < NUM_MODELS; j++) {
        for (size_t k = 0; k < CX; k++) {
            xMixed[j][k] = 0;
            if (j == MODEL_CV) xMixed[j][k] = mixProb[0][j] * xCV_[k];
        }
    }
}

inline void IMMFilter::updateModelProbabilities(float mx, float my)
{
    float meas[2] = {mx, my};
    float innovCV_X = meas[0] - xCV_[0];
    float innovCV_Y = meas[1] - xCV_[2];
    float SCV_X = PxCV_[0][0] + cfg_.measurementNoiseX;
    float SCV_Y = PxCV_[2][2] + cfg_.measurementNoiseY;
    float logLCV = -0.5f * (innovCV_X * innovCV_X / std::max(SCV_X, TINY) + innovCV_Y * innovCV_Y / std::max(SCV_Y, TINY));

    float innovCA_X = meas[0] - xCA_[0];
    float innovCA_Y = meas[1] - xCA_[3];
    float SCA_X = PxCA_[0][0] + cfg_.measurementNoiseX;
    float SCA_Y = PxCA_[3][3] + cfg_.measurementNoiseY;
    float logLCA = -0.5f * (innovCA_X * innovCA_X / std::max(SCA_X, TINY) + innovCA_Y * innovCA_Y / std::max(SCA_Y, TINY));

    float innovCT_X = meas[0] - xCT_[0];
    float innovCT_Y = meas[1] - xCT_[2];
    float SCT_X = PxCT_[0][0] + cfg_.measurementNoiseX;
    float SCT_Y = PxCT_[2][2] + cfg_.measurementNoiseY;
    float logLCT = -0.5f * (innovCT_X * innovCT_X / std::max(SCT_X, TINY) + innovCT_Y * innovCT_Y / std::max(SCT_Y, TINY));

    float sumL = std::exp(logLCV) * probCV_ + std::exp(logLCA) * probCA_ + std::exp(logLCT) * probCT_;
    probCV_ = (sumL > TINY) ? std::exp(logLCV) * probCV_ / sumL : 0;
    probCA_ = (sumL > TINY) ? std::exp(logLCA) * probCA_ / sumL : 0;
    probCT_ = (sumL > TINY) ? std::exp(logLCT) * probCT_ / sumL : 0;
}

inline void IMMFilter::combine()
{
    float xCombined[4] = {};
    xCombined[0] = probCV_ * xCV_[0] + probCA_ * xCA_[0] + probCT_ * xCT_[0];
    xCombined[1] = probCV_ * xCV_[1] + probCA_ * xCA_[1] + probCT_ * xCT_[1];
    xCombined[2] = probCV_ * xCV_[2] + probCA_ * xCA_[3] + probCT_ * xCT_[2];
    xCombined[3] = probCV_ * xCV_[3] + probCA_ * xCA_[4] + probCT_ * xCT_[3];
    for (size_t i = 0; i < CX; i++) {
        xCV_[i] = xCombined[i];
        xCA_[i < 3 ? i : i + 1] = xCombined[i];
        if (i < CTX-1) xCT_[i] = xCombined[i];
    }
}

inline void IMMFilter::predict(float dt)
{
    if (!cfg_.enabled) { initialized_ = false; return; }
    if (!initialized_) {
        initialized_ = true;
        return;
    }
    interact(dt);
    predictCV(dt);
    predictCA(dt);
    predictCT(dt);
}

inline void IMMFilter::update(float measuredX, float measuredY)
{
    if (!cfg_.enabled) return;
    if (!initialized_) {
        xCV_[0] = measuredX; xCV_[2] = measuredY;
        xCA_[0] = measuredX; xCA_[3] = measuredY;
        xCT_[0] = measuredX; xCT_[2] = measuredY;
        initialized_ = true;
        return;
    }
    updateCV(measuredX, measuredY);
    updateCA(measuredX, measuredY);
    updateCT(measuredX, measuredY);
    updateModelProbabilities(measuredX, measuredY);
    combine();
}

inline void IMMFilter::getState(float& estX, float& estY, float& velX, float& velY)
{
    estX = xCV_[0]; estY = xCV_[2];
    velX = xCV_[1]; velY = xCV_[3];
}

inline void IMMFilter::getPrediction(float predictDt, float& predX, float& predY)
{
    predX = xCV_[0] + xCV_[1] * predictDt + 0.5f * xCA_[2] * predictDt * predictDt;
    predY = xCV_[2] + xCV_[3] * predictDt + 0.5f * xCA_[5] * predictDt * predictDt;
}

#endif