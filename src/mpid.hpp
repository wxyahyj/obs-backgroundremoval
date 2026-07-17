#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace mist::reconstructed {

struct Detection {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int class_id = 0;
    int reserved = 0; // Original records advance by 24 bytes.
};

struct MoveCommand {
    int x = 0;
    int y = 0;
};

struct AimProfile {
    bool enabled = false;
    bool hotkey_down = false;
    int class_id = 0;

    // Matches the original formula:
    // target = box_origin + box_size - box_size * ratio.
    // A ratio of 0.5 means center; larger vertical ratios aim higher.
    float horizontal_ratio = 0.5f;
    float vertical_ratio = 0.5f;
};

struct MixedAimMode {
    bool enabled = false;
    bool hotkey_down = false;
};

struct RandomClassAimMode {
    bool enabled = false;
    bool hotkey_down = false;
};

struct RecoilAccumulator {
    bool enabled = false;
    bool left_button_down = false;
    int current = 0;
    int max = 0;
    int step = 0;
};

struct SideCompensation {
    bool enabled = false;
    bool hotkey_down = false;
    float cap = 0.0f;
    float denominator = 1.0f;
};

struct MotionFilters {
    bool line_enabled = false; // sub_18007C570, left as a hook.
    bool adrc_enabled = false; // sub_18010C750, left as a hook.
};

struct ChainConfig {
    int center = 0;        // dword_18167CD90
    int center_x = 0;      // App integration override; 0 falls back to center.
    int center_y = 0;      // App integration override; 0 falls back to center.
    int aim_radius = 0;    // dword_18167CDC0
    float speed_x = 1.0f;  // dword_18167CDD4 / ZYSD
    float speed_y = 1.0f;  // dword_18167CDD8 / SXSD

    bool jitter_enabled = false;                  // byte_18167CDE7
    bool idle_randomize_vertical_ratio = false;   // byte_18167CDE6
    bool pid_enabled = false;                     // byte_18167CE4C

    AimProfile profile_a;
    AimProfile profile_b;
    MixedAimMode mixed_mode;
    RandomClassAimMode random_class_mode;
    RecoilAccumulator recoil;
    SideCompensation side_compensation;
    MotionFilters filters;
};

struct ProcessResult {
    bool has_target = false;
    bool emitted_move = false;
    MoveCommand selected_error{};
    MoveCommand output_move{};
    int selected_distance = 99999;
    int target_x = 0;
    int target_y = 0;
    int target_width = 0;
    int target_height = 0;
};

// ============================================================
// 1 维标准卡尔曼滤波器(对应 pid.obj 中的 KF3 / KF4)
// 用于 D 项的两次平滑滤波
// ============================================================
class KalmanFilter1D {
public:
    // 初始化参数
    void init(double q, double r, double p, double x) {
        q_ = q; r_ = r; p_ = p; x_ = x; k_ = 0.0;
    }

    // 更新状态:z 为测量值,返回滤波后的估计值
    double update(double z) {
        // 除零保护:denom <= 0 时直接返回当前估计,避免 NaN 污染状态
        double denom = p_ + q_ + r_;
        if (denom <= 0.0) {
            k_ = 0.0;
            return x_;
        }
        k_ = (p_ + q_) / denom;
        x_ = x_ + k_ * (z - x_);
        p_ = (1.0 - k_) * (p_ + q_);
        return x_;
    }

    void reset() { x_ = 0.0; p_ = 1.0; k_ = 0.0; }

    double q() const { return q_; }
    double r() const { return r_; }
    double p() const { return p_; }
    double x() const { return x_; }
    double k() const { return k_; }

    void set_q(double v) { q_ = v; }
    void set_r(double v) { r_ = v; }
    void set_p(double v) { p_ = v; }
    void set_x(double v) { x_ = v; }

private:
    double q_ = 1.0;  // 过程噪声
    double r_ = 1.0;  // 测量噪声
    double p_ = 1.0;  // 估计误差协方差
    double x_ = 0.0;  // 状态估计
    double k_ = 0.0;  // 卡尔曼增益
};

// ============================================================
// 1 维自适应滤波器(对应 pid.obj 中的 KF1 / KF2)
// 根据误差大小自适应调整滤波强度,输出限制在 [0, 1]
// 用于积分权重(KF1)和输出权重(KF2)
// ============================================================
class AdaptiveFilter1D {
public:
    // 初始化参数
    void init(double threshold, double param) {
        threshold_ = threshold;
        param_ = param;
        x_ = 0.0;
    }

    // 更新状态:error_abs 为当前误差绝对值,返回滤波后的权重 [0, 1]
    double update(double error_abs) {
        // 除零保护:误差或阈值非正时直接返回当前权重,避免 0/0 产生 NaN
        if (error_abs <= 0.0 || threshold_ <= 0.0) {
            return x_;
        }
        double delta;
        if (threshold_ <= error_abs) {
            // 大误差:逐步衰减权重
            delta = (threshold_ / error_abs * x_ - x_) * 0.1;
        } else {
            // 小误差:逐步增加权重
            delta = (1.0 - error_abs / threshold_ - x_) * param_;
        }
        x_ = x_ + delta;
        // 限制在 [0, 1]
        if (x_ < 0.0) x_ = 0.0;
        if (x_ > 1.0) x_ = 1.0;
        return x_;
    }

    void reset() { x_ = 0.0; }

    double threshold() const { return threshold_; }
    double param() const { return param_; }
    double x() const { return x_; }

    void set_threshold(double v) { threshold_ = v; }
    void set_param(double v) { param_ = v; }
    void set_x(double v) { x_ = v; }

private:
    double threshold_ = 50.0;  // 误差阈值
    double param_ = 0.1;       // 小误差时的调整参数(对应旧库 integral_gain_rate_ = 0.1)
    double x_ = 0.0;           // 当前权重 [0, 1]
};

// ============================================================
// PID 控制器(完整逆向自 pid.obj / pid_x64.lib)
//
// 实现特性:
//   - 4 路滤波:KF1(积分权重) + KF2(输出权重) + KF3(D一次) + KF4(D二次)
//   - atan2 软限幅:对 P/I/D/输出 分别应用反正切限幅,避免硬截断抖动
//   - 变积分模式:kiMode 0=串级累加+I+D, 1=I+D合并, 2=不积分
//   - 跳变检测:|Δerror| > 30 时重置所有状态
//   - 多级死区:|error|<0.3→0, |D|≤0.5→0, |I|≤kiLimit1→0
//   - 小误差平滑:|error|<1 且 |diff|>0.5 时用 lastDiff*0.5+diff 累加到 D
//   - 精度处理:全程 round(x*100)/100 保留两位小数
// ============================================================
class PidController {
public:
    PidController() {
        init_defaults();
    }

    // ---------- 基础配置(兼容旧接口,仅设置 kp/ki/kd,不启用 D1) ----------
    void configure(double kp, double ki, double kd) {
        kp_ = kp;
        ki_ = ki;
        kd_ = kd;
    }

    // ---------- 完整初始化(对应 pid_init) ----------
    // 设置完整参数,启用 D1 项(targetGain_)和 4 路滤波
    // 注意:predict 对应 KF2 param(旧库 kp_gain_rate_),rate 对应 D 增益(旧库 rate_)
    void init(double kp, double ki, double kd, double predict, double rate) {
        kpLimit1_ = 9000.0;     // P 限幅 hardLimit
        kpLimit2_ = 1000.0;     // D2 限幅 hardLimit
        kdLimit_ = 10000.0;     // atan2 softParam(所有限幅共用)
        kiLimit1_ = 0.3;        // I 死区
        predict_ = 12000.0;     // atan2 参测量(保留)
        kf3_.set_q(0.1);        // KF3 过程噪声(对应旧库 kf2Q_)
        kp_ = kp;
        ki_ = ki;
        kd_ = kd;
        kf2_.set_param(predict);    // KF2 参数 = predict(对应旧库 kp_gain_rate_ = predict)
        targetGain_ = rate;         // D 增益 = rate(对应旧库 rate_ = rate)
        outputLimit_ = 0.0;         // 不限幅
        kiMode_ = 1;                // 默认 I+D 合并模式
    }

    // ---------- 高级配置(对应 pid_set_base) ----------
    // 第4参数 kdLimit 存到 kdLimit2_(对应旧库,实际未使用),不覆盖 kdLimit_(softParam)
    void set_base(int kiMode, double kpLimit1, double kpLimit2,
                  double kdLimit, double outputLimit,
                  double kf3_q, double kiLimit1) {
        kiMode_ = kiMode;
        kpLimit1_ = kpLimit1;       // P 限幅 hardLimit
        kpLimit2_ = kpLimit2;       // D2 限幅 hardLimit
        kdLimit2_ = kdLimit;        // 存储但未使用(对应旧库 kdLimit2_)
        outputLimit_ = outputLimit; // 输出限幅 hardLimit
        kf3_.set_q(kf3_q);          // KF3 过程噪声(对应旧库 kf2Q_)
        kiLimit1_ = kiLimit1;       // I 死区
    }

    // ---------- 动态更新参数(对应 pid_update_params) ----------
    // 注意:predict 对应 KF2 param(旧库 kp_gain_rate_),rate 对应 D 增益(旧库 rate_)
    void update_params(double kp, double ki, double kd, double predict, double rate) {
        targetGain_ = rate;         // D 增益 = rate
        kf2_.set_param(predict);    // KF2 参数 = predict
        kp_ = kp;
        ki_ = ki;
        kd_ = kd;
    }

    // ---------- 核心 PID 计算(对应 pid_update) ----------
    double update(double error);

    // ---------- 重置内部状态(对应 pid_reset) ----------
    // 只重置运行时状态(output/integral/error/diff/4 路滤波器状态),
    // 不重置配置参数(kp/ki/kd/limit/kf_q 等)
    void reset() {
        lastOutput_ = 0.0;
        integralTerm_ = 0.0;
        integralCopy_ = 0.0;
        lastError_ = 0.0;
        lastDiff_ = 0.0;
        kf1_.reset();
        kf2_.reset();
        kf3_.reset();
        kf4_.reset();
    }

    // ---------- 状态查询 ----------
    double output() const { return lastOutput_; }
    double last_error() const { return lastError_; }
    double last_diff() const { return lastDiff_; }
    double integral() const { return integralTerm_; }
    double kp() const { return kp_; }
    double ki() const { return ki_; }
    double kd() const { return kd_; }

private:
    void init_defaults() {
        // KF1(积分权重)默认参数:threshold=50, param=0.1(对应旧库 integral_gain_rate_)
        kf1_.init(50.0, 0.1);
        // KF2(输出权重)默认参数
        kf2_.init(1920.0, 0.03);
        // KF3(D 项一次滤波)默认参数
        kf3_.init(1.0, 1.0, 1.0, 0.0);
        // KF4(D 项二次滤波)默认参数
        kf4_.init(1.0, 1.0, 1.0, 0.0);
    }

    // ---------- PID 基础参数 ----------
    double kp_ = 0.0;            // +0   比例系数
    double ki_ = 0.0;            // +8   积分系数
    double kd_ = 0.0;            // +16  微分系数
    // +24 未使用
    double predict_ = 12000.0;   // +32  atan2 参考量
    double kpLimit1_ = 9000.0;   // +40  P 限幅 hardLimit(对应旧库 kpLimit_)
    double kpLimit2_ = 1000.0;   // +48  D2 限幅 hardLimit(对应旧库 kiLimit_)
    double kdLimit_ = 10000.0;   // +56  atan2 softParam(对应旧库 kdLimit_, 所有限幅共用)
    double kdLimit2_ = 9000.0;   // set_base 第4参数存储(对应旧库 kdLimit2_, 实际未使用)
    double outputLimit_ = 0.0;   // +64  输出限幅 hardLimit(0=不限幅)
    double targetGain_ = 0.0;    // +72  D 增益
    double kiLimit1_ = 0.3;      // +80  I 死区
    int    kiMode_ = 1;          // +88  积分模式(0=串级累加, 1=I+D合并, 2=不积分)
    int    pad92_ = 0;           // +92

    // ---------- 运行时状态 ----------
    // +96, +104 未使用
    // +112 reset 清零(未使用)
    double lastOutput_ = 0.0;    // +120 上次输出
    // +128, +136 未使用
    // +144 reset 清零(未使用)
    double integralTerm_ = 0.0;  // +152 积分累积项
    double integralCopy_ = 0.0;  // +160 积分副本(原反编译影子字段,保留以匹配内存布局)
    double lastError_ = 0.0;     // +168 上次误差
    double lastDiff_ = 0.0;      // +176 上次总输出(原反编译作为下次微分基线使用,见 update 步骤6 小误差平滑分支)
    // +184 未使用

    // ---------- 4 路滤波器 ----------
    AdaptiveFilter1D kf1_;       // +256~ KF1 积分权重(自适应,0~1)
    AdaptiveFilter1D kf2_;       // +256~ KF2 输出权重(自适应,0~1)
    KalmanFilter1D   kf3_;       // +304~ KF3 D 项一次滤波(标准卡尔曼)
    KalmanFilter1D   kf4_;       // +336~ KF4 D 项二次滤波(标准卡尔曼)
};

// ============================================================
// PID 控制链(上层逻辑:目标选择 + 后处理)
// 保持原 PidControlChain 接口,内部 pid_ 改用 PidController
// ============================================================
class PidControlChain {
public:
    using FilterHook = std::function<float(float)>;

    // 基础 PID 配置(仅 kp/ki/kd,不启用 D1 项)
    void configure_pid(float kp, float ki, float kd);

    // 完整 PID 初始化(启用 D1 项 + 4 路滤波)
    void init_pid(float kp, float ki, float kd, float predict, float rate);

    // 高级配置转发(对应 pid_set_base)
    void set_pid_base(int kiMode, double kpLimit1, double kpLimit2,
                      double kdLimit, double outputLimit,
                      double kf3_q, double kiLimit1);

    // 动态参数更新转发(对应 pid_update_params)
    void update_pid_params(float kp, float ki, float kd, float predict, float rate);

    void reset_runtime();

    ProcessResult process(ChainConfig& config,
                          const std::vector<Detection>& detections);

    void set_line_filter(FilterHook hook) { line_filter_ = std::move(hook); }
    void set_adrc_filter(FilterHook hook) { adrc_filter_ = std::move(hook); }
    void set_rng_seed(std::uint32_t seed) { rng_.seed(seed); }

    // 暴露 PidController 配置接口(用于需要直接配置的场景)
    PidController& pid() { return pid_; }
    const PidController& pid() const { return pid_; }
    int previous_output_x() const { return previous_output_x_; }
    int last_distance() const { return last_distance_; }

private:
    void consider_detection(const Detection& detection,
                            const AimProfile& profile,
                            const ChainConfig& config,
                            ProcessResult& result);
    MoveCommand post_process_motion(ChainConfig& config, MoveCommand move, int selected_distance);
    int jitter();
    static int distance_from_center(int dx, int dy);

    PidController pid_;
    FilterHook line_filter_;
    FilterHook adrc_filter_;
    std::mt19937 rng_{0x4D495354u};

    int random_class_selector_ = 0; // dword_18167CDE0 behavior: rand() % 2
    int previous_output_x_ = 0;     // dword_18167CFDC
    int last_distance_ = 99999;     // dword_18167CFB0

    int negative_counter_ = 0;      // dword_18167CFD8
    int positive_counter_ = 0;      // dword_18167CFD4
    bool compensate_negative_ = false; // byte_18167CE56
    bool compensate_positive_ = false; // byte_18167CE57
    float positive_bias_ = 0.0f;    // dword_18167CFC0
    float negative_bias_ = 0.0f;    // dword_18167CFC4
    float side_probe_ = 0.0f;       // dword_18167CFB8
};

} // namespace mist::reconstructed
