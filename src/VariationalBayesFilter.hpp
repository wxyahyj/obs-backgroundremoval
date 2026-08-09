#ifndef VARIATIONAL_BAYES_FILTER_HPP
#define VARIATIONAL_BAYES_FILTER_HPP

#include <cmath>
#include <algorithm>

// 变分贝叶斯鲁棒滤波器（VB-AKF，Särkkä & Nummenmaa 2009）
// 跟踪量是屏幕误差 error = target - fovCenter，与 IMMFilter 同约定。
// 自己鼠标移动会直接改变 error，必须作为控制输入注入，否则速度估计反向。
//
// 与普通 Kalman 的区别：
//  1) 测量噪声 R 未知 → 逆Gamma先验 + 变分定点迭代在线估计，检测抖动/帧率波动时
//     自动调 R，不用手工调"测量噪声"参数。
//  2) 野值抑制 → 新息 z-score 超过 outlierGate 时收缩（Huber式截断），
//     检测框瞬间跳几十像素的离群点不会把状态打飞。
//  3) 遗忘因子 rho 控制 R 估计的时变跟踪速度。
class VariationalBayesFilter {
public:
	struct Config {
		float processNoisePos = 0.1f;   // 位置过程噪声（同 IMM CV）
		float processNoiseVel = 0.5f;   // 速度过程噪声（同 IMM CV）
		float measurementNoiseX = 1.0f; // X轴测量噪声先验 R0（初值）
		float measurementNoiseY = 1.0f; // Y轴测量噪声先验 R0（初值）
		float nu0 = 5.0f;               // 逆Gamma先验自由度，越大→R估计越稳（收敛慢）
		float rho = 0.97f;              // 遗忘因子(0.5,1)，越大→R估计越平滑，越小→越跟手
		int iterations = 5;             // 变分定点迭代次数(1..20)，越大→越接近精确贝叶斯解
		float outlierGate = 4.0f;       // 野值门限(标准差倍数)，0=关闭野值抑制
		bool enabled = true;
	};

	VariationalBayesFilter();
	void reset();
	void setConfig(const Config& cfg);

	// uMoveX/Y：上一帧实际鼠标输出（像素）。自己向右移 → errorX 变小，故位置状态减 move。
	void predict(float dt, float uMoveX = 0.0f, float uMoveY = 0.0f);
	void update(float measuredErrorX, float measuredErrorY);

	// getPrediction 只返回速度外推增量 delta，不含滤波位置（同 IMM 语义）
	void getState(float& estX, float& estY, float& velX, float& velY) const;
	void getPrediction(float predictDt, float& deltaX, float& deltaY) const;

	// 诊断用：当前 R 估计
	void getNoiseEstimates(float& rX, float& rY) const { rX = estRX_; rY = estRY_; }

	// 机动检测：目标急转弯时速度估计指向旧方向不可信，调用方应关闭提前量外推。
	// 判定 = 绝对位移(像素) + 统计σ 双阈值——纯 z-score 会把检测框噪声跳误判为机动
	// （320x320 画面框跳几十px是常态，z 巨大但并非目标真动）。
	bool maneuverDetected() const {
		return lastInnovPx_ > kManeuverAbsPx && lastInnovZ_ > kManeuverZ;
	}

private:
	static constexpr float kManeuverAbsPx = 32.0f;  // 绝对位移门限（像素）
	static constexpr float kManeuverZ = 3.0f;       // 统计门限（σ）
	float maneuverZ_ = 4.0f;    // 野值截断门限（随 outlierGate），与机动判定独立
	float lastInnovZ_ = 0.0f;   // 最近一次 update 的最大新息 z-score
	float lastInnovPx_ = 0.0f;  // 最近一次 update 的最大新息绝对值（像素）
	Config cfg_;

	// 状态 [posX, velX, posY, velY]，每轴独立 2 状态 CV（与 IMM CV 同布局）
	float x_[4];
	float P_[4][4]; // 块对角：X 轴 2x2 + Y 轴 2x2

	// 逆Gamma 超参数：p(R) = IG(alpha, beta)，R 估计 = beta/(alpha-1)
	float alphaX_, betaX_;
	float alphaY_, betaY_;
	float estRX_, estRY_;

	bool initialized_;

	void applyControl(float dt, float uMoveX, float uMoveY);
	void predictAxis(float dt, float& pos, float& vel,
	                 float& p00, float& p01, float& p11);
	void updateAxis(float meas, float& pos, float& vel,
	                float& p00, float& p01, float& p11,
	                float& alpha, float& beta, float& estR,
	                float& maxZ, float& maxPx);
};

inline VariationalBayesFilter::VariationalBayesFilter()
	: estRX_(1.0f)
	, estRY_(1.0f)
	, initialized_(false)
{
	reset();
}

inline void VariationalBayesFilter::reset()
{
	for (size_t i = 0; i < 4; i++) x_[i] = 0.0f;
	for (size_t i = 0; i < 4; i++)
		for (size_t j = 0; j < 4; j++)
			P_[i][j] = (i == j) ? 100.0f : 0.0f;

	// 逆Gamma 先验：alpha = (nu0+1)/2, beta = (nu0-2)/2 * R0 → 初值 R ≈ 0.75*R0
	alphaX_ = 0.5f * (cfg_.nu0 + 1.0f);
	betaX_ = 0.5f * std::max(cfg_.nu0 - 2.0f, 0.5f) * std::max(cfg_.measurementNoiseX, 1e-3f);
	alphaY_ = 0.5f * (cfg_.nu0 + 1.0f);
	betaY_ = 0.5f * std::max(cfg_.nu0 - 2.0f, 0.5f) * std::max(cfg_.measurementNoiseY, 1e-3f);
	estRX_ = cfg_.measurementNoiseX;
	estRY_ = cfg_.measurementNoiseY;
	initialized_ = false;
}

inline void VariationalBayesFilter::setConfig(const Config& cfg)
{
	cfg_ = cfg;
	cfg_.rho = std::max(0.5f, std::min(1.0f, cfg_.rho));
	cfg_.iterations = std::max(1, std::min(20, cfg_.iterations));
	cfg_.outlierGate = std::max(0.0f, cfg_.outlierGate);
	cfg_.nu0 = std::max(1.1f, cfg_.nu0);
}

// 鼠标输出直接改变误差：error_new ≈ error_old - mouseMove（同 IMM）。
// 位置和速度都扣自己移动——只扣位置会把"自己移动"误判成目标运动。
inline void VariationalBayesFilter::applyControl(float dt, float uMoveX, float uMoveY)
{
	float invDt = (dt > 1e-6f) ? (1.0f / dt) : 0.0f;
	x_[0] -= uMoveX;          x_[1] -= uMoveX * invDt;
	x_[2] -= uMoveY;          x_[3] -= uMoveY * invDt;
}

inline void VariationalBayesFilter::predictAxis(float dt, float& pos, float& vel,
                                                float& p00, float& p01, float& p11)
{
	pos += vel * dt;

	float dt2 = dt * dt;
	float dt3 = dt2 * dt;
	float dt4 = dt2 * dt2;
	float q00 = cfg_.processNoisePos * dt3 / 3.0f + cfg_.processNoiseVel * dt4 / 4.0f;
	float q01 = cfg_.processNoiseVel * dt3 / 2.0f;
	float q11 = cfg_.processNoisePos * dt + cfg_.processNoiseVel * dt2;

	// F*P*F' + Q（2x2 手写）
	p00 = p00 + 2.0f * dt * p01 + dt2 * p11 + q00;
	p01 = p01 + dt * p11 + q01;
	p11 = p11 + q11;
}

inline void VariationalBayesFilter::updateAxis(float meas, float& pos, float& vel,
                                               float& p00, float& p01, float& p11,
                                               float& alpha, float& beta, float& estR,
                                               float& maxZ, float& maxPx)
{
	// 1) 遗忘：α^- = ρ·α, β^- = ρ·β
	float alphaPred = cfg_.rho * alpha;
	float betaPred = cfg_.rho * beta;

	// 2) 变分定点迭代：x/P 从预测值出发，R 每轮重估
	for (int it = 0; it < cfg_.iterations; it++) {
		// R = β^- / (α^- - 1)（IG 均值，要求 α>1），夹紧防退化
		float R = betaPred / std::max(alphaPred - 1.0f, 1e-3f);
		R = std::max(1e-3f, std::min(1e6f, R));

		float innov = meas - pos;
		float S = p00 + R;

		// 机动检测：原始（未截断）新息，z-score + 像素双尺度，供 maneuverDetected 用
		if (S > 0.0f) {
			float z = std::abs(innov) / std::sqrt(S);
			if (z > maxZ) maxZ = z;
		}
		float apx = std::abs(innov);
		if (apx > maxPx) maxPx = apx;

		// 野值抑制：Huber 式截断新息（以 sqrt(S) 为标准差尺度）
		if (cfg_.outlierGate > 0.0f && S > 0.0f) {
			float zscore = std::abs(innov) / std::sqrt(S);
			if (zscore > cfg_.outlierGate)
				innov *= cfg_.outlierGate / zscore;
		}

		float K0 = p00 / S;
		float K1 = p01 / S;
		pos += K0 * innov;
		vel += K1 * innov;
		// P -= K*S*K'（利用 K0*S=p00, K1*S=p01 化简）
		p11 -= K1 * p01;
		p01 -= K0 * p01;
		p00 -= K0 * p00;

		// 3) 变分更新超参数：α += 1/2, β += 1/2·(z - Hx)²
		alpha = alphaPred + 0.5f;
		beta = betaPred + 0.5f * innov * innov;
	}

	estR = beta / std::max(alpha - 1.0f, 1e-3f);
	estR = std::max(1e-3f, std::min(1e6f, estR));
}

inline void VariationalBayesFilter::predict(float dt, float uMoveX, float uMoveY)
{
	if (!cfg_.enabled) {
		initialized_ = false;
		return;
	}
	if (!initialized_ || dt <= 1e-6f)
		return;

	// 1) 先扣掉自己鼠标位移（控制输入，位置+速度）
	applyControl(dt, uMoveX, uMoveY);
	// 2) 自由动力学预测（目标运动），两轴独立块对角
	predictAxis(dt, x_[0], x_[1], P_[0][0], P_[0][1], P_[1][1]);
	predictAxis(dt, x_[2], x_[3], P_[2][2], P_[2][3], P_[3][3]);
}

inline void VariationalBayesFilter::update(float measuredErrorX, float measuredErrorY)
{
	if (!cfg_.enabled)
		return;

	if (!initialized_) {
		x_[0] = measuredErrorX; x_[1] = 0.0f;
		x_[2] = measuredErrorY; x_[3] = 0.0f;
		for (size_t i = 0; i < 4; i++)
			for (size_t j = 0; j < 4; j++)
				P_[i][j] = (i == j) ? 50.0f : 0.0f;
		lastInnovZ_ = 0.0f;
		lastInnovPx_ = 0.0f;
		initialized_ = true;
		return;
	}

	// 机动检测门限：outlierGate 关闭(0)时用固定 3.5σ
	maneuverZ_ = (cfg_.outlierGate > 0.0f) ? cfg_.outlierGate : 3.5f;

	float maxZ = 0.0f, maxPx = 0.0f;
	updateAxis(measuredErrorX, x_[0], x_[1], P_[0][0], P_[0][1], P_[1][1],
	           alphaX_, betaX_, estRX_, maxZ, maxPx);
	updateAxis(measuredErrorY, x_[2], x_[3], P_[2][2], P_[2][3], P_[3][3],
	           alphaY_, betaY_, estRY_, maxZ, maxPx);
	lastInnovZ_ = maxZ;
	lastInnovPx_ = maxPx;
}

inline void VariationalBayesFilter::getState(float& estX, float& estY, float& velX, float& velY) const
{
	estX = x_[0];
	estY = x_[2];
	velX = x_[1];
	velY = x_[3];
}

inline void VariationalBayesFilter::getPrediction(float predictDt, float& deltaX, float& deltaY) const
{
	float h = std::max(0.0f, predictDt);
	const float maxVel = 8000.0f;
	float vx = std::max(-maxVel, std::min(maxVel, x_[1]));
	float vy = std::max(-maxVel, std::min(maxVel, x_[3]));
	deltaX = vx * h;
	deltaY = vy * h;
}

#endif
