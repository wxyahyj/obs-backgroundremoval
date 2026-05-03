#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace mist::reconstructed {

struct Detection {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int class_id = 0;
    int reserved = 0;
};

struct MoveCommand {
    int x = 0;
    int y = 0;
};

struct AimProfile {
    bool enabled = false;
    bool hotkey_down = false;
    int class_id = 0;
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
    bool line_enabled = false;
    bool adrc_enabled = false;
};

struct ChainConfig {
    int center = 0;
    int center_x = 0;
    int center_y = 0;
    int aim_radius = 0;
    float speed_x = 1.0f;
    float speed_y = 1.0f;

    bool jitter_enabled = false;
    bool idle_randomize_vertical_ratio = false;
    bool pid_enabled = false;

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

class IncrementalPid {
public:
    void configure(float kp, float ki, float kd, float d_alpha = 0.2f);
    void reset(float output = 0.0f);
    float update(float error);

    float output() const { return output_; }
    float previous_output() const { return previous_output_; }
    float previous_error() const { return previous_error_; }
    float previous_previous_error() const { return previous_previous_error_; }
    float kp() const { return kp_; }
    float ki() const { return ki_; }
    float kd() const { return kd_; }

private:
    float output_ = 0.0f;
    float previous_output_ = 0.0f;
    float previous_error_ = 0.0f;
    float previous_previous_error_ = 0.0f;
    float previous_d_term_ = 0.0f;
    float kp_ = 0.0f;
    float ki_ = 0.0f;
    float kd_ = 0.0f;
    float d_alpha_ = 0.2f;  // D项滤波系数
};

class PidControlChain {
public:
    using FilterHook = std::function<float(float)>;

    void configure_pid(float kp, float ki, float kd, float d_alpha = 0.2f);
    void reset_runtime();

    ProcessResult process(ChainConfig& config,
                          const std::vector<Detection>& detections);

    void set_line_filter(FilterHook hook) { line_filter_ = std::move(hook); }
    void set_adrc_filter(FilterHook hook) { adrc_filter_ = std::move(hook); }
    void set_rng_seed(std::uint32_t seed) { rng_.seed(seed); }

    const IncrementalPid& pid() const { return pid_; }
    int previous_output_x() const { return previous_output_x_; }
    int last_distance() const { return last_distance_; }

private:
    void consider_detection(const Detection& detection,
                            const AimProfile& profile,
                            ChainConfig& config,
                            ProcessResult& result);
    MoveCommand post_process_motion(ChainConfig& config, MoveCommand move, int selected_distance);
    int jitter();
    static int distance_from_center(int dx, int dy);

    IncrementalPid pid_;
    FilterHook line_filter_;
    FilterHook adrc_filter_;
    std::mt19937 rng_{0x4D495354u};

    int random_class_selector_ = 0;
    int previous_output_x_ = 0;
    int last_distance_ = 99999;

    int negative_counter_ = 0;
    int positive_counter_ = 0;
    bool compensate_negative_ = false;
    bool compensate_positive_ = false;
    float positive_bias_ = 0.0f;
    float negative_bias_ = 0.0f;
    float side_probe_ = 0.0f;
};

}
