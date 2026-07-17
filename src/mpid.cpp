#include "mpid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace mist::reconstructed {

namespace {

// 四舍五入到两位小数(对应 pid.obj 中的 round(x * 100.0) / 100.0)
inline double round2(double x) {
    return std::round(x * 100.0) / 100.0;
}

// atan2 软限幅(对应 pid.obj 中的 atan2 + (predict - limit*0.1) 逻辑)
// value: 待限幅的值
// predict: atan2 参考量
// limit: 限幅参数(0 表示不限幅)
inline double atan2_limit(double value, double predict, double limit) {
    if (limit == 0.0) return value;
    double angle = std::atan2(value, predict);
    return round2(angle * (predict - limit * 0.1));
}

// 向 0 截断(对应原反编译中的 (int)float 强转,非四舍五入)
// 加范围保护:float 转 int 超出 INT 范围时是 UB,此处截断到 INT_MAX/INT_MIN
int trunc_to_int(float value) {
    if (value >= static_cast<float>(std::numeric_limits<int>::max()))
        return std::numeric_limits<int>::max();
    if (value <= static_cast<float>(std::numeric_limits<int>::min()))
        return std::numeric_limits<int>::min();
    return static_cast<int>(value);
}

AimProfile profile_for_class(const ChainConfig& config, int class_id) {
    return class_id == config.profile_a.class_id ? config.profile_a : config.profile_b;
}

} // namespace

// ============================================================
// PidController::update - 核心 PID 算法
// 精确逆向自 pid.obj sub_1800011C0 (1533 字节,带地址对照)
// ============================================================
double PidController::update(double error) {
    // === 0. NaN/Inf 守卫:异常输入返回上次有效输出,避免污染滤波器状态 ===
    // 若放行 NaN:fabs(NaN)<0.3 为 false → 不归零;后续 NaN 会污染 kf1_.x_ 永久残留
    if (std::isnan(error) || std::isinf(error)) {
        return lastOutput_;
    }

    // === 1. 死区过滤:|error| < 0.3 → error = 0 (0x180001204) ===
    double v3 = error;
    if (std::fabs(v3) < 0.3) {
        v3 = 0.0;
    }

    // === 2. 误差跳变检测:|v3 - lastError| > 30 → 重置所有状态 (0x180001225) ===
    double v5 = lastError_;  // +168
    if (std::fabs(v3 - v5) > 30.0) {
        // 重置所有运行时状态(对应 0x18000124B ~ 0x1800012A9)
        v5 = 0.0;
        kf2_.set_x(0.0);         // +256 KF2 输出权重
        kf1_.set_x(0.0);         // +280 KF1 积分权重
        lastOutput_ = 0.0;       // +120
        lastError_ = 0.0;        // +168
        integralTerm_ = 0.0;     // +152
        lastDiff_ = 0.0;         // +176
        kf3_.set_p(0.0);         // +328 KF3 p
        kf3_.set_x(0.0);         // +320 KF3 x
        kf4_.set_p(0.0);         // +360 KF4 p
        kf4_.set_x(0.0);         // +352 KF4 x
        // +112, +144 也清零(未使用字段)
        integralCopy_ = 0.0;     // +160 (对应 +144 的重复清零)
    }

    // 当前误差绝对值(用于自适应滤波器)
    double v11 = std::fabs(v3);

    // === 3. KF1:积分权重自适应滤波 (0x1800012C2 ~ 0x180001360) ===
    // KF1 阈值=50, 参数=0.025
    // if (50 <= |error|) 衰减; else 增加
    // 输出 kf1_x 限制在 [0, 1],用作积分权重
    kf1_.update(v11);

    // === 4. KF2:输出权重自适应滤波 (0x180001363 ~ 0x1800013EB) ===
    // KF2 阈值=1920, 参数=0.03 (init 时 = rate)
    // if (1920 <= |error|) 衰减; else 增加
    // 输出 kf2_x 限制在 [0, 1],用作输出权重
    kf2_.update(v11);

    // === 5. KF3:D 项一次滤波(标准卡尔曼) (0x1800013EE ~ 0x18000144D) ===
    // 原始 D = (error - lastError + lastDiff) * 100,四舍五入后 /100
    double v20 = v3 - v5;  // error - lastError
    double v7 = lastDiff_; // lastDiff
    double v23 = std::round((v3 - v5 + v7) * 100.0);  // 原始 D * 100
    double kf3_input = v23 / 100.0;
    double v26 = kf3_.update(kf3_input);  // KF3 滤波
    double v28 = round2(v26);             // 四舍五入

    // === 6. 小误差平滑修正 (0x18000148C ~ 0x1800014AB) ===
    // |error| < 1.0 且 |error - lastError| > 0.5 时(对应旧库 Step 6)
    // D += round1(0.5 * lastOutput + delta_error)(累加,不是替换)
    if (v11 < 1.0 && std::fabs(v20) > 0.5) {
        v28 += round2(0.5 * lastDiff_ + v20);
    }

    // === 7. KF4:D 项二次滤波(标准卡尔曼) (0x1800014B5 ~ 0x18000150E) ===
    // 对 KF3 输出(经过小误差修正)再次卡尔曼滤波
    // 注意:KF4 的 Q 来自 KF3 的输出(对应旧库 Step 7: P_pred = kf3P_ + kf2_.x_)
    kf4_.set_q(kf3_.x());
    kf4_.set_r(kf3_.r());
    double v29 = std::round(v28 * 100.0);
    double kf4_input = v29 / 100.0;
    double v32 = kf4_.update(kf4_input);
    double v33 = round2(v32);

    // === 8. D 项死区:|filteredD| ≤ 0.5 → 0 (0x180001535) ===
    if (std::fabs(v33) <= 0.5) {
        v33 = 0.0;
    }

    // === 9. D 项计算 + atan2 软限幅 (0x180001541 ~ 0x180001585) ===
    // D = filteredD * targetGain * kf1_x(积分权重)
    // atan2 限幅:softParam=kdLimit_, hardLimit=outputLimit_(对应旧库 Step 9)
    double Y = v33 * targetGain_ * kf1_.x();  // D 项
    if (outputLimit_ != 0.0) {
        Y = atan2_limit(Y, kdLimit_, outputLimit_);
    }

    // === 10. 积分计算 (0x18000158D ~ 0x1800015B6) ===
    // kiMode <= 1 时计算积分
    // I = error * ki * kf1_x + integralTerm
    double v38 = 0.0;   // I 项
    double Y_2 = 0.0;   // 积分项副本
    if (kiMode_ <= 1) {
        Y_2 = v3 * ki_ * kf1_.x() + integralTerm_;
        integralTerm_ = Y_2;  // +152 更新积分累积
    }

    // === 11. P 项和 D2 项计算 (0x1800015E1 ~ 0x180001604) ===
    // P = round(error * kp * 100) / 100
    double Y_1 = round2(v3 * kp_);
    // D2 = round((error - lastError) * kd * 100) / 100
    double Y_3 = round2((v3 - lastError_) * kd_);

    // === 12. P/I/D2 的 atan2 软限幅 (0x18000160B ~ 0x1800016AC) ===
    // 所有限幅的 softParam 都是 kdLimit_(对应旧库 atan2Clamp 的统一 softParam)
    // P 限幅:hardLimit = kpLimit1_(对应旧库 kpLimit_)
    if (kpLimit1_ != 0.0) {
        Y_1 = atan2_limit(Y_1, kdLimit_, kpLimit1_);
    }
    // I 限幅:hardLimit = outputLimit_(对应旧库 Step 10: i_out 限幅用 outputLimit_)
    if (outputLimit_ != 0.0) {
        Y_2 = atan2_limit(Y_2, kdLimit_, outputLimit_);
    }
    // D2 限幅:hardLimit = kpLimit2_(对应旧库 Step 11b: d2_out 限幅用 kiLimit_)
    if (kpLimit2_ != 0.0) {
        Y_3 = atan2_limit(Y_3, kdLimit_, kpLimit2_);
    }

    // === 13. 积分模式分支 (0x1800016B1 ~ 0x1800016D7) ===
    // 验证自原反编译:
    //   kiMode==0: integralTerm_ += Y(串级累加 D),然后 fall through 到 0x6D3
    //              执行 v38 = Y_2 + Y(I 与 D 合并输出)
    //   kiMode==1: 直接跳到 0x6D3,执行 v38 = Y_2 + Y
    //   kiMode==2: 跳过 0x6D3,v38 保持初始值 0(不积分)
    if (kiMode_ == 0) {
        integralTerm_ = Y + integralTerm_;  // 串级累加 D 到积分项
    }
    if (kiMode_ <= 1) {
        v38 = Y_2 + Y;  // kiMode 0 或 1:I + D 合并参与输出
    }
    // kiMode == 2 时 v38 保持 0

    // === 14. I 项死区:|I| ≤ kiLimit1 → 0 (0x1800016E9) ===
    if (std::fabs(v38) <= kiLimit1_) {
        v38 = 0.0;
    }

    // === 15. 总输出 = P + I + D2 (0x180001702) ===
    double v49 = std::round((Y_1 + v38 + Y_3) * 100.0);
    double Y_4 = v49 / 100.0;

    // === 16. 输出 atan2 软限幅 (0x180001719 ~ 0x18000173B) ===
    // softParam = kdLimit_, hardLimit = outputLimit_(对应旧库 Step 13)
    if (outputLimit_ != 0.0) {
        Y_4 = atan2_limit(Y_4, kdLimit_, outputLimit_);
    }

    // === 17. 应用 KF2 输出权重 + 精度处理 (0x18000174D) ===
    // output = round(Y_4 * kf2_x * 100) / 100
    double v54 = std::round(Y_4 * kf2_.x() * 100.0);
    double output = v54 / 100.0;

    // === 18. 保存状态 (0x180001753 ~ 0x1800017B0) ===
    integralCopy_ = integralTerm_;  // +160 = integralTerm
    lastError_ = v3;                // +168 = error
    lastOutput_ = output;           // +120 = output
    lastDiff_ = output;             // +176 = output

    return output;
}

// ============================================================
// PidControlChain - 上层逻辑(保持原实现,仅 pid_ 类型变更)
// ============================================================
void PidControlChain::configure_pid(float kp, float ki, float kd) {
    pid_.configure(static_cast<double>(kp),
                   static_cast<double>(ki),
                   static_cast<double>(kd));
}

void PidControlChain::init_pid(float kp, float ki, float kd,
                                float predict, float rate) {
    pid_.init(static_cast<double>(kp),
              static_cast<double>(ki),
              static_cast<double>(kd),
              static_cast<double>(predict),
              static_cast<double>(rate));
}

void PidControlChain::set_pid_base(int kiMode, double kpLimit1, double kpLimit2,
                                    double kdLimit, double outputLimit,
                                    double kf3_q, double kiLimit1) {
    pid_.set_base(kiMode, kpLimit1, kpLimit2, kdLimit, outputLimit, kf3_q, kiLimit1);
}

void PidControlChain::update_pid_params(float kp, float ki, float kd,
                                         float predict, float rate) {
    pid_.update_params(static_cast<double>(kp),
                       static_cast<double>(ki),
                       static_cast<double>(kd),
                       static_cast<double>(predict),
                       static_cast<double>(rate));
}

void PidControlChain::reset_runtime() {
    pid_.reset();
    random_class_selector_ = 0;
    previous_output_x_ = 0;
    last_distance_ = 99999;
    negative_counter_ = 0;
    positive_counter_ = 0;
    compensate_negative_ = false;
    compensate_positive_ = false;
    positive_bias_ = 0.0f;
    negative_bias_ = 0.0f;
    side_probe_ = 0.0f;
}

ProcessResult PidControlChain::process(ChainConfig& config,
                                       const std::vector<Detection>& detections) {
    ProcessResult result;
    result.selected_distance = 99999;

    bool any_aim_hotkey_active = false;

    for (const Detection& detection : detections) {
        if (config.profile_a.enabled && config.profile_a.hotkey_down) {
            any_aim_hotkey_active = true;
            consider_detection(detection, config.profile_a, config, result);
        }

        if (config.profile_b.enabled && config.profile_b.hotkey_down) {
            any_aim_hotkey_active = true;
            consider_detection(detection, config.profile_b, config, result);
        }

        if (config.mixed_mode.enabled && config.mixed_mode.hotkey_down) {
            any_aim_hotkey_active = true;
            AimProfile selected = profile_for_class(config, detection.class_id);
            selected.enabled = true;
            selected.hotkey_down = true;
            selected.class_id = detection.class_id;
            consider_detection(detection, selected, config, result);
        }

        if (config.random_class_mode.enabled && config.random_class_mode.hotkey_down) {
            any_aim_hotkey_active = true;
            if (detection.class_id == random_class_selector_) {
                AimProfile selected = profile_for_class(config, random_class_selector_);
                selected.enabled = true;
                selected.hotkey_down = true;
                selected.class_id = detection.class_id;
                consider_detection(detection, selected, config, result);
            }
        } else {
            random_class_selector_ = std::uniform_int_distribution<int>(0, 1)(rng_);
        }
    }

    if (config.recoil.enabled && config.recoil.left_button_down) {
        result.selected_error.y += config.recoil.current;
        if (config.recoil.current < config.recoil.max) {
            config.recoil.current += config.recoil.step;
        }
    } else {
        config.recoil.current = 0;
    }

    if (any_aim_hotkey_active) {
        if (result.selected_distance < config.aim_radius) {
            result.has_target = true;
            result.output_move = post_process_motion(config,
                                                     result.selected_error,
                                                     result.selected_distance);
            result.emitted_move = true;
        } else {
            result.selected_error = {};
        }
    } else if (config.idle_randomize_vertical_ratio) {
        const int value = std::uniform_int_distribution<int>(2, 8)(rng_);
        const float ratio = static_cast<float>(value) * 0.1f;
        config.profile_a.vertical_ratio = ratio;
        config.profile_b.vertical_ratio = ratio;
    }

    return result;
}

void PidControlChain::consider_detection(const Detection& detection,
                                         const AimProfile& profile,
                                         const ChainConfig& config,
                                         ProcessResult& result) {
    if (!profile.enabled || !profile.hotkey_down) {
        return;
    }

    if (detection.class_id != profile.class_id) {
        return;
    }

    const int target_x = trunc_to_int(
        static_cast<float>(detection.x + detection.width) -
        static_cast<float>(detection.width) * profile.horizontal_ratio);

    const int target_y = trunc_to_int(
        static_cast<float>(detection.y + detection.height) -
        static_cast<float>(detection.height) * profile.vertical_ratio);

    const int center_x = config.center_x > 0 ? config.center_x : config.center;
    const int center_y = config.center_y > 0 ? config.center_y : config.center;
    const int dx = target_x - center_x;
    const int dy = target_y - center_y;
    const int distance = distance_from_center(dx, dy);

    if (distance >= result.selected_distance) {
        return;
    }

    if (distance >= config.aim_radius) {
        result.selected_error = {};
        return;
    }

    result.selected_distance = distance;
    result.target_x = target_x;
    result.target_y = target_y;
    result.target_width = detection.width;
    result.target_height = detection.height;

    result.selected_error.x = dx;
    result.selected_error.y = dy;
    if (config.jitter_enabled) {
        result.selected_error.x += jitter();
        result.selected_error.y += jitter();
    }
}

MoveCommand PidControlChain::post_process_motion(ChainConfig& config,
                                                 MoveCommand move,
                                                 int selected_distance) {
    if (config.filters.line_enabled && line_filter_) {
        move.x = trunc_to_int(line_filter_(static_cast<float>(move.x)));
        move.y = trunc_to_int(line_filter_(static_cast<float>(move.y)));
    }

    if (config.filters.adrc_enabled && adrc_filter_) {
        move.x = trunc_to_int(-adrc_filter_(static_cast<float>(move.x)));
        move.y = trunc_to_int(-adrc_filter_(static_cast<float>(move.y)));
    }

    // PID 计算:用 PidController::update(double) -> double
    // 内部完整实现 4 路滤波 + atan2 限幅 + 变积分
    if (config.pid_enabled) {
        move.x = trunc_to_int(static_cast<float>(
            pid_.update(static_cast<double>(move.x))));
    }

    if (config.side_compensation.enabled && config.side_compensation.hotkey_down) {
        if (previous_output_x_ >= 0) {
            negative_counter_ = 0;
        } else if (++negative_counter_ > 10) {
            compensate_negative_ = true;
            compensate_positive_ = false;
            negative_counter_ = 0;
        }

        if (previous_output_x_ <= 0) {
            positive_counter_ = 0;
        } else if (++positive_counter_ > 10) {
            compensate_positive_ = true;
            compensate_negative_ = false;
            positive_counter_ = 0;
        }

        const float denom = config.side_compensation.denominator == 0.0f
                                ? 1.0f
                                : config.side_compensation.denominator;
        // 用 long long 避免 INT_MIN 的有符号溢出 UB
        side_probe_ = (static_cast<float>(std::abs(static_cast<long long>(move.x))) / denom) * 2.0f;

        if (previous_output_x_ == 0) {
            positive_bias_ = 0.0f;
            negative_bias_ = 0.0f;
        }

        if (compensate_positive_) {
            move.x = trunc_to_int(static_cast<float>(move.x) + positive_bias_);
            positive_bias_ = std::min(config.side_compensation.cap, positive_bias_ + 0.2f);
        }

        if (compensate_negative_) {
            move.x = trunc_to_int(static_cast<float>(move.x) - negative_bias_);
            negative_bias_ = std::min(config.side_compensation.cap, negative_bias_ + 0.2f);
        }
    }

    MoveCommand output;
    output.x = trunc_to_int(static_cast<float>(move.x) * config.speed_x);
    output.y = trunc_to_int(static_cast<float>(move.y) * config.speed_y);

    previous_output_x_ = output.x;
    last_distance_ = selected_distance;
    return output;
}

int PidControlChain::jitter() {
    return std::uniform_int_distribution<int>(0, 6)(rng_) - 3;
}

int PidControlChain::distance_from_center(int dx, int dy) {
    const double x = static_cast<double>(dx);
    const double y = static_cast<double>(dy);
    const double mag = std::sqrt(x * x + y * y);
    // 溢出保护:超过 INT_MAX 时截断到 INT_MAX
    if (mag > static_cast<double>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(mag);
}

} // namespace mist::reconstructed
