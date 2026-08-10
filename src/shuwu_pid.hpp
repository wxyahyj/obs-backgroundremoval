#ifndef SHUWU_PID_HPP
#define SHUWU_PID_HPP

#include <string>
#include <cmath>

// ============================================================================
// 书屋控制器 (ShuWuPid) — 由 pid_x64.lib (LTCG/C2 IL, MSVC 14.44.35207)
// 全函数反汇编重构 (2026-08-10)，bit 级验证 114/114 一致。
//
// 设计要点:
//   1. 双调制积分: kiIntegral(thr=50, rate) 调 I 项强度,
//      kpIntegral(thr=1920, rate) 调 P 项强度(启动渐入防猛拉)
//   2. 双标量卡尔曼: kf1 滤 Δe+prevMove; kf2 滤 kf1.x,
//      小误差路径(|e|<1 && |Δe|<0.1)观测 prevMove*0.5+Δe
//   3. 突变重置: |Δe|>30 → 全状态清零 (目标转向/换目标瞬间旧状态作废)
//   4. atan2 软限幅: v = atan2(v, 10000)*(10000-limit) — 渐近饱和不突变
//   5. 全链路 round1 两位小数量化; 死区 0.3(输入)/0.5(输出)
//   6. predict = 输出速度倍率
// ============================================================================

namespace shuwu {

static inline double sw_round1(double v) {
    return std::round(v * 100.0) / 100.0;
}
static inline double sw_clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

// ---------------------------------------------------------------- 标量卡尔曼
class SwKalman {
public:
    double q = 1.0, r = 1.0, x = 0.0, p = 1.0;

    SwKalman() = default;
    SwKalman(double q_, double r_, double x0_) : q(q_), r(r_), x(x0_), p(1.0) {}

    void init(double q_) { q = q_; }
    double update(double z) {
        double pp = p + q;
        double k = pp / (pp + r);
        x += (z - x) * k;
        p = pp * (1.0 - k);
        return x;
    }
    void reset() { x = 0.0; p = 0.0; }
};

// ---------------------------------------------------------------- 书屋 PID
class ShuWuPid {
public:
    // 参数（对应 pid.h C 接口 init/setBase，完全一致）
    double kp = 0.0, ki = 0.0, kd = 0.0;
    double maxLimit = 10000.0;   // atan2 分母
    double kpLimit = 9900.0;     // P 限幅 (setBase 参数2)
    double kiLimit = 9900.0;     // I 限幅 (setBase 参数3)
    double kdLimit = 9900.0;     // D 限幅 (setBase 参数4)
    double limit = 0.0;          // 总输出限幅 (setBase 参数5, 0=关)
    double predict = 1.0;        // 输出速度倍率 (init 参数4)
    double kiDeadband = 0.5;     // I 输出死区 (setBase 参数7)
    int    kiMode = 1;           // (setBase 参数1) 0=I累加 1=I开 其它=关
    double lastOut = 0.0;
    double iacc = 0.0, iaccMirror = 0.0;
    double prevError = 0.0, prevMove = 0.0;

    // 积分调制
    double kpIntegral = 0.0, kpThreshold = 1920.0, rate = 0.03;   // init 参数5
    double kiIntegral = 0.0, kiThreshold = 50.0, kiRate = 0.025;  // setBase 参数6 → kf1.Q

    // 双卡尔曼
    SwKalman kf1{1.0, 1.0, 0.0};
    SwKalman kf2{1.0, 1.0, 0.0};

    // 名称守卫（pid.lib: 仅 name==fixedId 才写参数）
    std::string name;
    std::string fixedId = "1458679219";

    ShuWuPid() = default;

    void setName(const char* deviceCode) {
        name = deviceCode;
        fixedId = "1458679219";
    }

    // init: 守卫内写入全部默认+主参数
    bool init(double kp_, double ki_, double kd_, double predict_, double rate_) {
        if (name == fixedId) {
            maxLimit = 10000.0;
            kpLimit = 9000.0;
            kiLimit = 1000.0;
            kdLimit = 9000.0;
            limit = 0.0;
            kiDeadband = 0.3;
            kf1.init(0.1);
            kf2.init(0.1);
            kp = kp_; ki = ki_; kd = kd_;
            predict = predict_;
            rate = rate_;
            kiMode = 1;
        }
        return true;
    }

    // setBase: 与 pid.lib 完全一致 (kiMode + 6 double)
    void setBase(int kiMode_, double kpLimit_, double kiLimit_, double kdLimit_,
                 double limit_, double kf1Q_, double kiDeadband_) {
        kpLimit = kpLimit_;
        kiLimit = kiLimit_;
        kdLimit = kdLimit_;
        limit = limit_;
        kf1.q = kf1Q_;
        kiDeadband = kiDeadband_;
        kiMode = kiMode_;
    }

    void updateParams(double kp_, double ki_, double kd_, double predict_, double rate_) {
        if (name == fixedId) {
            kp = kp_; ki = ki_; kd = kd_;
            predict = predict_;
            rate = rate_;
        }
    }

    // |v|<thr : i += (1-|v|/thr - i)*rate;  |v|>=thr : i += (thr/|v|*i - i)*0.1
    static double adjustIntegral(double v, double thr, double rate, double integral) {
        double a = std::fabs(v);
        if (thr > a)
            integral += (1.0 - a / thr - integral) * rate;
        else
            integral += (thr / a * integral - integral) * 0.1;
        return sw_clamp01(integral);
    }

    // atan2 软限幅（渐近饱和，不硬截断）
    static double softLimit(double v, double lim) {
        if (lim == 0.0) return v;
        return sw_round1(std::atan2(v, 10000.0) * (10000.0 - lim));
    }

    double update(double error) {
        // 输入死区 0.3
        if (std::fabs(error) < 0.3) error = 0.0;

        double dError = error - prevError;

        // 突变重置: 目标转向/瞬跳 >30 → 全状态清零
        if (std::fabs(dError) > 30.0) {
            kpIntegral = 0.0; kiIntegral = 0.0;
            lastOut = 0.0; prevError = 0.0;
            iacc = 0.0; prevMove = 0.0;
            kf1.reset(); kf2.reset();
        }

        // 双调制积分
        kiIntegral = adjustIntegral(error, kiThreshold, kiRate, kiIntegral);
        kpIntegral = adjustIntegral(error, kpThreshold, rate, kpIntegral);

        // 双卡尔曼: kf1 滤 Δe+prevMove; kf2 观测 kf1.x 或小误差路径
        double move = sw_round1(dError + prevMove);
        kf1.update(move);
        double z2;
        if (std::fabs(error) < 1.0 && std::fabs(dError) < 0.1)
            z2 = sw_round1(prevMove * 0.5 + dError);
        else
            z2 = sw_round1(kf1.x);
        kf2.update(z2);

        // 输出链: kf2.x → 死区0.5 → ×predict → ×kiIntegral → I限幅
        double out = sw_round1(kf2.x);
        if (std::fabs(out) <= 0.5) out = 0.0;
        out *= predict;
        out *= kiIntegral;
        if (kiLimit != 0.0)
            out = sw_round1(std::atan2(out, maxLimit) * (maxLimit - kiLimit * 0.1));

        // I 项
        double iterm = 0.0;
        if (kiMode == 0) iacc += out;
        if (kiMode == 0 || kiMode == 1) {
            double i = error * ki * kiIntegral + iacc;
            iacc = i;
            if (kiLimit != 0.0)
                i = sw_round1(std::atan2(i, maxLimit) * (maxLimit - kiLimit));
            iterm = i + out;
            if (std::fabs(iterm) <= kiDeadband) iterm = 0.0;
        }

        // P / D
        double pterm = softLimit(sw_round1(error * kp), kpLimit);
        double dterm = softLimit(sw_round1((error - prevError) * kd), kdLimit);

        // 合计 (P+I)+D → 总限幅 → ×kpIntegral
        double total = sw_round1((pterm + iterm) + dterm);
        if (limit != 0.0)
            total = sw_round1(std::atan2(total, maxLimit) * (maxLimit - limit));

        double out2 = sw_round1(total * kpIntegral);

        iaccMirror = iacc;
        prevError = error;
        lastOut = out2;
        prevMove = out2;
        return out2;
    }

    void reset() {
        kpIntegral = 0.0; kiIntegral = 0.0;
        lastOut = 0.0; prevError = 0.0;
        iacc = 0.0; prevMove = 0.0;
        kf1.reset(); kf2.reset();
    }
};

} // namespace shuwu

#endif
