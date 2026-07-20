#ifndef IMM_FILTER_HPP
#define IMM_FILTER_HPP

#include <cmath>
#include <algorithm>
#include <array>

// IMM 交互多模型滤波器（陈金广《目标跟踪系统中的滤波方法》）
// 跟踪量是屏幕误差 error = target - fovCenter，不是绝对坐标。
// 自己鼠标移动会直接改变 error，必须作为控制输入注入，否则速度估计反向。
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

	// uMoveX/Y：上一帧实际鼠标输出（像素）。自己向右移 → errorX 变小，故位置状态减 move。
	void predict(float dt, float uMoveX = 0.0f, float uMoveY = 0.0f);
	void update(float measuredErrorX, float measuredErrorY);

	// 当前滤波后误差 + 目标运动外推（不含自己再移动）
	void getState(float& estX, float& estY, float& velX, float& velY) const;
	void getPrediction(float predictDt, float& predErrorX, float& predErrorY) const;

private:
	Config cfg_;

	static constexpr size_t MODEL_CV = 0;
	static constexpr size_t MODEL_CA = 1;
	static constexpr size_t MODEL_CT = 2;
	static constexpr size_t NUM_MODELS = 3;
	static constexpr float TINY = 1e-6f;
	static constexpr size_t CX = 4;
	static constexpr size_t CAX = 6;
	static constexpr size_t CTX = 5;

	void applyControl(float uMoveX, float uMoveY);
	void predictCV(float dt);
	void updateCV(float mx, float my);
	void predictCA(float dt);
	void updateCA(float mx, float my);
	void predictCT(float dt);
	void updateCT(float mx, float my);
	void interact();
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
			PxCV_[i][j] = (i == j) ? 100.0f : 0.0f;

	for (size_t i = 0; i < CAX; i++) xCA_[i] = 0.0f;
	for (size_t i = 0; i < CAX; i++)
		for (size_t j = 0; j < CAX; j++)
			PxCA_[i][j] = (i == j) ? 100.0f : 0.0f;

	for (size_t i = 0; i < CTX; i++) xCT_[i] = 0.0f;
	for (size_t i = 0; i < CTX; i++)
		for (size_t j = 0; j < CTX; j++)
			PxCT_[i][j] = (i == j) ? 100.0f : 0.0f;

	probCV_ = 1.0f / 3.0f;
	probCA_ = 1.0f / 3.0f;
	probCT_ = 1.0f / 3.0f;
	initialized_ = false;
}

inline void IMMFilter::setConfig(const Config& cfg)
{
	cfg_ = cfg;
}

// 鼠标输出直接改变误差：error_new ≈ error_old - mouseMove
inline void IMMFilter::applyControl(float uMoveX, float uMoveY)
{
	xCV_[0] -= uMoveX;
	xCV_[2] -= uMoveY;
	xCA_[0] -= uMoveX;
	xCA_[3] -= uMoveY;
	xCT_[0] -= uMoveX;
	xCT_[2] -= uMoveY;
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
	float dt4 = dt2 * dt2;
	Q[0][0] = Qpos * dt3 / 3.0f + Qvel * dt4 / 4.0f;
	Q[0][1] = Qvel * dt3 / 2.0f;
	Q[1][0] = Qvel * dt3 / 2.0f;
	Q[1][1] = Qpos * dt + Qvel * dt2;
	Q[2][2] = Qpos * dt3 / 3.0f + Qvel * dt4 / 4.0f;
	Q[2][3] = Qvel * dt3 / 2.0f;
	Q[3][2] = Qvel * dt3 / 2.0f;
	Q[3][3] = Qpos * dt + Qvel * dt2;

	float xPred[CX] = {};
	float Pp[CX][CX] = {};
	for (size_t i = 0; i < CX; i++)
		for (size_t j = 0; j < CX; j++)
			xPred[i] += F[i][j] * xCV_[j];

	for (size_t i = 0; i < CX; i++) {
		for (size_t j = 0; j < CX; j++) {
			float row[CX] = {};
			for (size_t k = 0; k < CX; k++)
				row[j] += F[i][k] * PxCV_[k][j];
			(void)row;
		}
	}
	for (size_t i = 0; i < CX; i++) {
		for (size_t j = 0; j < CX; j++) {
			float sum = 0.0f;
			for (size_t k = 0; k < CX; k++)
				sum += F[i][k] * PxCV_[k][j];
			float sum2 = 0.0f;
			for (size_t k = 0; k < CX; k++)
				sum2 += sum * F[j][k];
			Pp[i][j] = sum2 + Q[i][j];
		}
	}

	for (size_t i = 0; i < CX; i++) xCV_[i] = xPred[i];
	for (size_t i = 0; i < CX; i++)
		for (size_t j = 0; j < CX; j++)
			PxCV_[i][j] = Pp[i][j];
}

inline void IMMFilter::updateCV(float mx, float my)
{
	float innovX = mx - xCV_[0];
	float innovY = my - xCV_[2];
	float Sx = PxCV_[0][0] + cfg_.measurementNoiseX;
	float Sy = PxCV_[2][2] + cfg_.measurementNoiseY;
	float Kx[4] = {
		PxCV_[0][0] / std::max(Sx, TINY),
		PxCV_[1][0] / std::max(Sx, TINY),
		PxCV_[2][0] / std::max(Sx, TINY),
		PxCV_[3][0] / std::max(Sx, TINY)
	};
	float Ky[4] = {
		PxCV_[0][2] / std::max(Sy, TINY),
		PxCV_[1][2] / std::max(Sy, TINY),
		PxCV_[2][2] / std::max(Sy, TINY),
		PxCV_[3][2] / std::max(Sy, TINY)
	};
	for (size_t i = 0; i < CX; i++)
		xCV_[i] += Kx[i] * innovX + Ky[i] * innovY;

	float Pold[CX][CX];
	for (size_t i = 0; i < CX; i++)
		for (size_t j = 0; j < CX; j++)
			Pold[i][j] = PxCV_[i][j];
	for (size_t i = 0; i < CX; i++)
		for (size_t j = 0; j < CX; j++)
			PxCV_[i][j] = Pold[i][j] - Kx[i] * Pold[0][j] - Ky[i] * Pold[2][j];
}

inline void IMMFilter::predictCA(float dt)
{
	float halfDt2 = 0.5f * dt * dt;
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
	float dt2 = dt * dt;
	float dt3 = dt2 * dt;
	float dt4 = dt2 * dt2;
	float dt5 = dt3 * dt2;
	for (size_t d = 0; d < 2; d++) {
		size_t b = d * 3;
		Q[b][b] = Qacc * dt5 / 20.0f;
		Q[b][b + 1] = Qacc * dt4 / 8.0f; Q[b + 1][b] = Qacc * dt4 / 8.0f;
		Q[b][b + 2] = Qacc * dt3 / 6.0f; Q[b + 2][b] = Qacc * dt3 / 6.0f;
		Q[b + 1][b + 1] = Qacc * dt3 / 3.0f;
		Q[b + 1][b + 2] = Qacc * dt2 / 2.0f; Q[b + 2][b + 1] = Qacc * dt2 / 2.0f;
		Q[b + 2][b + 2] = Qacc * dt;
	}

	float xPred[CAX] = {};
	float Pp[CAX][CAX] = {};
	for (size_t i = 0; i < CAX; i++)
		for (size_t j = 0; j < CAX; j++)
			xPred[i] += F[i][j] * xCA_[j];
	for (size_t i = 0; i < CAX; i++) {
		for (size_t j = 0; j < CAX; j++) {
			float sum = 0.0f;
			for (size_t k = 0; k < CAX; k++)
				sum += F[i][k] * PxCA_[k][j];
			float sum2 = 0.0f;
			for (size_t k = 0; k < CAX; k++)
				sum2 += sum * F[j][k];
			Pp[i][j] = sum2 + Q[i][j];
		}
	}
	for (size_t i = 0; i < CAX; i++) xCA_[i] = xPred[i];
	for (size_t i = 0; i < CAX; i++)
		for (size_t j = 0; j < CAX; j++)
			PxCA_[i][j] = Pp[i][j];
}

inline void IMMFilter::updateCA(float mx, float my)
{
	float innovX = mx - xCA_[0];
	float innovY = my - xCA_[3];
	float Sx = PxCA_[0][0] + cfg_.measurementNoiseX;
	float Sy = PxCA_[3][3] + cfg_.measurementNoiseY;
	float Kx[6] = {
		PxCA_[0][0] / std::max(Sx, TINY),
		PxCA_[1][0] / std::max(Sx, TINY),
		PxCA_[2][0] / std::max(Sx, TINY),
		PxCA_[3][0] / std::max(Sx, TINY),
		PxCA_[4][0] / std::max(Sx, TINY),
		PxCA_[5][0] / std::max(Sx, TINY)
	};
	float Ky[6] = {
		PxCA_[0][3] / std::max(Sy, TINY),
		PxCA_[1][3] / std::max(Sy, TINY),
		PxCA_[2][3] / std::max(Sy, TINY),
		PxCA_[3][3] / std::max(Sy, TINY),
		PxCA_[4][3] / std::max(Sy, TINY),
		PxCA_[5][3] / std::max(Sy, TINY)
	};
	for (size_t i = 0; i < CAX; i++)
		xCA_[i] += Kx[i] * innovX + Ky[i] * innovY;

	float Pold[CAX][CAX];
	for (size_t i = 0; i < CAX; i++)
		for (size_t j = 0; j < CAX; j++)
			Pold[i][j] = PxCA_[i][j];
	for (size_t i = 0; i < CAX; i++)
		for (size_t j = 0; j < CAX; j++)
			PxCA_[i][j] = Pold[i][j] - Kx[i] * Pold[0][j] - Ky[i] * Pold[3][j];
}

inline void IMMFilter::predictCT(float dt)
{
	float w = xCT_[4];
	float sinWT = std::sin(w * dt);
	float cosWT = std::cos(w * dt);
	float a, b;
	if (std::abs(w) < 1e-4f) {
		a = dt;
		b = 0.5f * dt * dt * w;
	} else {
		a = sinWT / w;
		b = (1.0f - cosWT) / w;
	}

	// 状态 [x, vx, y, vy, ω]
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
	float temp[CTX][CTX] = {};
	float Pp[CTX][CTX] = {};
	for (size_t i = 0; i < CTX; i++)
		for (size_t j = 0; j < CTX; j++)
			xPred[i] += F[i][j] * xCT_[j];
	for (size_t i = 0; i < CTX; i++)
		for (size_t j = 0; j < CTX; j++)
			for (size_t k = 0; k < CTX; k++)
				temp[i][j] += F[i][k] * PxCT_[k][j];
	for (size_t i = 0; i < CTX; i++)
		for (size_t j = 0; j < CTX; j++) {
			float sum = 0.0f;
			for (size_t k = 0; k < CTX; k++)
				sum += temp[i][k] * F[j][k];
			Pp[i][j] = sum + Q[i][j];
		}
	for (size_t i = 0; i < CTX; i++) xCT_[i] = xPred[i];
	for (size_t i = 0; i < CTX; i++)
		for (size_t j = 0; j < CTX; j++)
			PxCT_[i][j] = Pp[i][j];
}

inline void IMMFilter::updateCT(float mx, float my)
{
	float innovX = mx - xCT_[0];
	float innovY = my - xCT_[2];
	float Sx = PxCT_[0][0] + cfg_.measurementNoiseX;
	float Sy = PxCT_[2][2] + cfg_.measurementNoiseY;
	float Kx[5] = {
		PxCT_[0][0] / std::max(Sx, TINY),
		PxCT_[1][0] / std::max(Sx, TINY),
		PxCT_[2][0] / std::max(Sx, TINY),
		PxCT_[3][0] / std::max(Sx, TINY),
		PxCT_[4][0] / std::max(Sx, TINY)
	};
	float Ky[5] = {
		PxCT_[0][2] / std::max(Sy, TINY),
		PxCT_[1][2] / std::max(Sy, TINY),
		PxCT_[2][2] / std::max(Sy, TINY),
		PxCT_[3][2] / std::max(Sy, TINY),
		PxCT_[4][2] / std::max(Sy, TINY)
	};
	for (size_t i = 0; i < CTX; i++)
		xCT_[i] += Kx[i] * innovX + Ky[i] * innovY;

	float Pold[CTX][CTX];
	for (size_t i = 0; i < CTX; i++)
		for (size_t j = 0; j < CTX; j++)
			Pold[i][j] = PxCT_[i][j];
	for (size_t i = 0; i < CTX; i++)
		for (size_t j = 0; j < CTX; j++)
			PxCT_[i][j] = Pold[i][j] - Kx[i] * Pold[0][j] - Ky[i] * Pold[2][j];
}

inline void IMMFilter::interact()
{
	float probs[3] = {probCV_, probCA_, probCT_};
	float c[3] = {};
	for (size_t j = 0; j < NUM_MODELS; j++)
		c[j] = transMatrix_[0][j] * probs[0] + transMatrix_[1][j] * probs[1] + transMatrix_[2][j] * probs[2];

	float mixProb[3][3] = {};
	for (size_t i = 0; i < NUM_MODELS; i++)
		for (size_t j = 0; j < NUM_MODELS; j++)
			mixProb[i][j] = (c[j] > TINY) ? transMatrix_[i][j] * probs[i] / c[j] : (i == j ? 1.0f : 0.0f);

	float xCV0 = xCV_[0], xCV1 = xCV_[1], xCV2 = xCV_[2], xCV3 = xCV_[3];
	float xCA0 = xCA_[0], xCA1 = xCA_[1], xCA3 = xCA_[3], xCA4 = xCA_[4];
	float xCT0 = xCT_[0], xCT1 = xCT_[1], xCT2 = xCT_[2], xCT3 = xCT_[3];

	xCV_[0] = mixProb[0][0] * xCV0 + mixProb[1][0] * xCA0 + mixProb[2][0] * xCT0;
	xCV_[1] = mixProb[0][0] * xCV1 + mixProb[1][0] * xCA1 + mixProb[2][0] * xCT1;
	xCV_[2] = mixProb[0][0] * xCV2 + mixProb[1][0] * xCA3 + mixProb[2][0] * xCT2;
	xCV_[3] = mixProb[0][0] * xCV3 + mixProb[1][0] * xCA4 + mixProb[2][0] * xCT3;

	xCA_[0] = mixProb[0][1] * xCV0 + mixProb[1][1] * xCA0 + mixProb[2][1] * xCT0;
	xCA_[1] = mixProb[0][1] * xCV1 + mixProb[1][1] * xCA1 + mixProb[2][1] * xCT1;
	xCA_[3] = mixProb[0][1] * xCV2 + mixProb[1][1] * xCA3 + mixProb[2][1] * xCT2;
	xCA_[4] = mixProb[0][1] * xCV3 + mixProb[1][1] * xCA4 + mixProb[2][1] * xCT3;

	xCT_[0] = mixProb[0][2] * xCV0 + mixProb[1][2] * xCA0 + mixProb[2][2] * xCT0;
	xCT_[1] = mixProb[0][2] * xCV1 + mixProb[1][2] * xCA1 + mixProb[2][2] * xCT1;
	xCT_[2] = mixProb[0][2] * xCV2 + mixProb[1][2] * xCA3 + mixProb[2][2] * xCT2;
	xCT_[3] = mixProb[0][2] * xCV3 + mixProb[1][2] * xCA4 + mixProb[2][2] * xCT3;
}

inline void IMMFilter::updateModelProbabilities(float mx, float my)
{
	float innovCV_X = mx - xCV_[0];
	float innovCV_Y = my - xCV_[2];
	float SCV_X = std::max(PxCV_[0][0] + cfg_.measurementNoiseX, TINY);
	float SCV_Y = std::max(PxCV_[2][2] + cfg_.measurementNoiseY, TINY);
	float logLCV = -0.5f * (innovCV_X * innovCV_X / SCV_X + innovCV_Y * innovCV_Y / SCV_Y);

	float innovCA_X = mx - xCA_[0];
	float innovCA_Y = my - xCA_[3];
	float SCA_X = std::max(PxCA_[0][0] + cfg_.measurementNoiseX, TINY);
	float SCA_Y = std::max(PxCA_[3][3] + cfg_.measurementNoiseY, TINY);
	float logLCA = -0.5f * (innovCA_X * innovCA_X / SCA_X + innovCA_Y * innovCA_Y / SCA_Y);

	float innovCT_X = mx - xCT_[0];
	float innovCT_Y = my - xCT_[2];
	float SCT_X = std::max(PxCT_[0][0] + cfg_.measurementNoiseX, TINY);
	float SCT_Y = std::max(PxCT_[2][2] + cfg_.measurementNoiseY, TINY);
	float logLCT = -0.5f * (innovCT_X * innovCT_X / SCT_X + innovCT_Y * innovCT_Y / SCT_Y);

	// 数值稳定：相对最大似然
	float maxLog = std::max(logLCV, std::max(logLCA, logLCT));
	float lCV = std::exp(logLCV - maxLog) * probCV_;
	float lCA = std::exp(logLCA - maxLog) * probCA_;
	float lCT = std::exp(logLCT - maxLog) * probCT_;
	float sumL = lCV + lCA + lCT;
	if (sumL > TINY) {
		probCV_ = lCV / sumL;
		probCA_ = lCA / sumL;
		probCT_ = lCT / sumL;
	} else {
		probCV_ = probCA_ = probCT_ = 1.0f / 3.0f;
	}
}

inline void IMMFilter::combine()
{
	// 公共状态 [ex, vx, ey, vy]
	float ex = probCV_ * xCV_[0] + probCA_ * xCA_[0] + probCT_ * xCT_[0];
	float vx = probCV_ * xCV_[1] + probCA_ * xCA_[1] + probCT_ * xCT_[1];
	float ey = probCV_ * xCV_[2] + probCA_ * xCA_[3] + probCT_ * xCT_[2];
	float vy = probCV_ * xCV_[3] + probCA_ * xCA_[4] + probCT_ * xCT_[3];

	xCV_[0] = ex; xCV_[1] = vx; xCV_[2] = ey; xCV_[3] = vy;
	// CA: [ex,vx,ax,ey,vy,ay] — 不覆盖 ax/ay
	xCA_[0] = ex; xCA_[1] = vx; xCA_[3] = ey; xCA_[4] = vy;
	// CT: [ex,vx,ey,vy,ω] — 不覆盖 ω
	xCT_[0] = ex; xCT_[1] = vx; xCT_[2] = ey; xCT_[3] = vy;
}

inline void IMMFilter::predict(float dt, float uMoveX, float uMoveY)
{
	if (!cfg_.enabled) {
		initialized_ = false;
		return;
	}
	if (!initialized_ || dt <= TINY)
		return;

	// 1) 先扣掉自己鼠标位移（控制输入）
	applyControl(uMoveX, uMoveY);
	// 2) 模型交互
	interact();
	// 3) 各模型自由动力学预测（目标运动）
	predictCV(dt);
	predictCA(dt);
	predictCT(dt);
}

inline void IMMFilter::update(float measuredErrorX, float measuredErrorY)
{
	if (!cfg_.enabled)
		return;

	if (!initialized_) {
		xCV_[0] = measuredErrorX; xCV_[1] = 0.0f; xCV_[2] = measuredErrorY; xCV_[3] = 0.0f;
		xCA_[0] = measuredErrorX; xCA_[1] = 0.0f; xCA_[2] = 0.0f;
		xCA_[3] = measuredErrorY; xCA_[4] = 0.0f; xCA_[5] = 0.0f;
		xCT_[0] = measuredErrorX; xCT_[1] = 0.0f; xCT_[2] = measuredErrorY; xCT_[3] = 0.0f; xCT_[4] = 0.0f;
		for (size_t i = 0; i < CX; i++)
			for (size_t j = 0; j < CX; j++)
				PxCV_[i][j] = (i == j) ? 50.0f : 0.0f;
		for (size_t i = 0; i < CAX; i++)
			for (size_t j = 0; j < CAX; j++)
				PxCA_[i][j] = (i == j) ? 50.0f : 0.0f;
		for (size_t i = 0; i < CTX; i++)
			for (size_t j = 0; j < CTX; j++)
				PxCT_[i][j] = (i == j) ? 50.0f : 0.0f;
		probCV_ = probCA_ = probCT_ = 1.0f / 3.0f;
		initialized_ = true;
		return;
	}

	updateCV(measuredErrorX, measuredErrorY);
	updateCA(measuredErrorX, measuredErrorY);
	updateCT(measuredErrorX, measuredErrorY);
	updateModelProbabilities(measuredErrorX, measuredErrorY);
	combine();
}

inline void IMMFilter::getState(float& estX, float& estY, float& velX, float& velY) const
{
	estX = xCV_[0];
	estY = xCV_[2];
	velX = xCV_[1];
	velY = xCV_[3];
}

inline void IMMFilter::getPrediction(float predictDt, float& predErrorX, float& predErrorY) const
{
	// 滤波后的当前误差 + 目标速度/加速度外推（不含鼠标再移动）
	float h = std::max(0.0f, predictDt);
	float h2 = h * h;
	float ax = xCA_[2];
	float ay = xCA_[5];
	predErrorX = xCV_[0] + xCV_[1] * h + 0.5f * ax * h2;
	predErrorY = xCV_[2] + xCV_[3] * h + 0.5f * ay * h2;
}

#endif
