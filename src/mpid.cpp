#include "mpid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace mist::reconstructed {

namespace {

int trunc_to_int(float value) {
    return static_cast<int>(value);
}

AimProfile profile_for_class(const ChainConfig& config, int class_id) {
    return class_id == config.profile_a.class_id ? config.profile_a : config.profile_b;
}

// 一欧元滤波器实现
class OneEuroFilter {
public:
    OneEuroFilter(float min_cutoff = 1.0f, float beta = 0.0f, float d_cutoff = 1.0f)
        : min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff) {}

    float filter(float value, float dt) {
        if (dt <= 0.0f) return value;

        // 首次初始化
        if (!initialized_) {
            x_prev_ = value;
            dx_prev_ = 0.0f;
            initialized_ = true;
            return value;
        }

        // 计算导数（速度）
        float dx = (value - x_prev_) / dt;

        // 平滑导数
        float edx = exponential_smoothing(dx, dx_prev_, alpha(dt, d_cutoff_));
        dx_prev_ = edx;

        // 动态截止频率
        float cutoff = min_cutoff_ + beta_ * std::abs(edx);

        // 平滑信号
        float result = exponential_smoothing(value, x_prev_, alpha(dt, cutoff));
        x_prev_ = result;

        return result;
    }

    void reset() {
        initialized_ = false;
        x_prev_ = 0.0f;
        dx_prev_ = 0.0f;
    }

private:
    float min_cutoff_;
    float beta_;
    float d_cutoff_;
    bool initialized_ = false;
    float x_prev_ = 0.0f;
    float dx_prev_ = 0.0f;

    static float alpha(float dt, float cutoff) {
        float tau = 1.0f / (2.0f * static_cast<float>(M_PI) * cutoff);
        return 1.0f / (1.0f + tau / dt);
    }

    static float exponential_smoothing(float value, float prev, float alpha) {
        return alpha * value + (1.0f - alpha) * prev;
    }
};

} // namespace

void IncrementalPid::configure(float kp, float ki, float kd) {
    kp_ = kp;
    ki_ = ki;
    kd_ = kd;
}

void IncrementalPid::reset(float output) {
    output_ = output;
    previous_output_ = output;
    previous_error_ = 0.0f;
    previous_previous_error_ = 0.0f;
}

float IncrementalPid::update(float error) {
    // 计算增量
    float p_term = kp_ * (error - previous_error_);
    float i_term = ki_ * error;
    float d_term = kd_ * (error - 2.0f * previous_error_ + previous_previous_error_);

    // 输出变化量
    float delta = p_term + i_term + d_term;

    // 限制最大变化量（防止突变）
    const float max_delta = 50.0f;
    delta = std::clamp(delta, -max_delta, max_delta);

    // 更新输出
    output_ += delta;

    // 输出衰减：防止累积过冲
    // 当误差减小时，加速衰减
    float error_trend = error - previous_error_;
    float decay_factor = 0.85f; // 基础衰减

    // 如果误差在减小（接近目标），增加衰减
    if (std::abs(error) < std::abs(previous_error_) && std::signbit(error) == std::signbit(previous_error_)) {
        decay_factor = 0.7f; // 更强的衰减
    }

    output_ *= decay_factor;

    // 限制输出范围
    const float max_output = 100.0f;
    output_ = std::clamp(output_, -max_output, max_output);

    previous_previous_error_ = previous_error_;
    previous_error_ = error;
    previous_output_ = output_;
    return output_;
}

void PidControlChain::configure_pid(float kp, float ki, float kd) {
    pid_.configure(kp, ki, kd);
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
                                         ChainConfig& config,
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

    if (config.pid_enabled) {
        move.x = trunc_to_int(pid_.update(static_cast<float>(move.x)));
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
        side_probe_ = (static_cast<float>(std::abs(move.x)) / denom) * 2.0f;

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
    return static_cast<int>(std::sqrt(x * x + y * y));
}

} // namespace mist::reconstructed