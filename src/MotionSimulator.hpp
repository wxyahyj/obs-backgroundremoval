#ifndef MOTION_SIMULATOR_HPP
#define MOTION_SIMULATOR_HPP

#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

class MotionSimulator {
private:
    enum MotionPhase {
        PHASE_INIT,
        PHASE_DIRECT,
        PHASE_LB_CHECK_WAIT,
        PHASE_DIRECT_PAUSE_1,
        PHASE_DIRECT_PAUSE_2,
        PHASE_OVERSHOOT,
        PHASE_OVERSHOOT_PAUSE_1,
        PHASE_OVERSHOOT_PAUSE_2,
        PHASE_OVERSHOOT_PAUSE_3,
        PHASE_OVERSHOOT_LB_WAIT,
        PHASE_MICRO_OVERSHOOT,
        PHASE_MICRO_OVERSHOOT_PAUSE,
        PHASE_INERTIA
    };

    enum LeftBtnState { LB_IDLE, LB_TAP, LB_SPRAY };

    static constexpr double DIST_MIN_RATIO = 0.05;   // 5% of diagonal
    static constexpr double DIST_MID_RATIO = 0.15;   // 15% of diagonal
    static constexpr double DIST_MAX_RATIO = 0.30;   // 30% of diagonal
    static constexpr double SIZE_SMALL = 0.02;
    static constexpr double SIZE_LARGE = 0.3;
    static constexpr double OVS_MIN_RATIO = 0.0;
    static constexpr double OVS_MAX_RATIO = 1.0;
    static constexpr double OVS_RETRY_OVS_PROB = 0.5;
    static constexpr double OVS_RETRY_DIRECT_PROB = 0.7;
    static constexpr double PRECISE_HEAD_MIN = 0.47;
    static constexpr double PRECISE_HEAD_MAX = 0.53;
    static constexpr double PRECISE_BODY_MIN = 0.1;
    static constexpr double PRECISE_BODY_MAX = 0.9;
    static constexpr double SECOND_HEAD_MIN = 0.47;
    static constexpr double SECOND_HEAD_MAX = 0.53;
    static constexpr double SECOND_BODY_MIN = 0.3;
    static constexpr double SECOND_BODY_MAX = 0.7;
    static constexpr double DIRECT_D20_MIN = 0.04;
    static constexpr double DIRECT_D20_MAX = 0.05;
    static constexpr double DIRECT_D40_MIN = 0.04;
    static constexpr double DIRECT_D40_MAX = 0.06;
    static constexpr double DIRECT_D60_MIN = 0.06;
    static constexpr double DIRECT_D60_MAX = 0.10;
    static constexpr double DIRECT_PAUSE1_MIN = 0.03;
    static constexpr double DIRECT_PAUSE1_MAX = 0.06;
    static constexpr double DIRECT_PAUSE2_MIN = 0.06;
    static constexpr double DIRECT_PAUSE2_MAX = 0.08;
    static constexpr double OVSDUR_D20_MIN = 0.02;
    static constexpr double OVSDUR_D20_MAX = 0.04;
    static constexpr double OVSDUR_D40_MIN = 0.03;
    static constexpr double OVSDUR_D40_MAX = 0.06;
    static constexpr double OVSDUR_D60_MIN = 0.03;
    static constexpr double OVSDUR_D60_MAX = 0.08;
    static constexpr double OVSDUR_PAUSE1_MIN = 0.01;
    static constexpr double OVSDUR_PAUSE1_MAX = 0.06;
    static constexpr double OVSDUR_PAUSE2_MIN = 0.06;
    static constexpr double OVSDUR_PAUSE2_MAX = 0.08;
    static constexpr double OVSDUR_PAUSE3_MIN = 0.06;
    static constexpr double OVSDUR_PAUSE3_MAX = 0.08;
    static constexpr double MOVS_DUR_MIN = 0.015;
    static constexpr double MOVS_DUR_MAX = 0.025;
    static constexpr double MOVS_PAUSE_MIN = 0.02;
    static constexpr double MOVS_PAUSE_MAX = 0.04;
    static constexpr double MOVS_HEAD_MIN = 0.55;
    static constexpr double MOVS_HEAD_MAX = 0.65;
    static constexpr double MOVS_BODY_MIN = 0.55;
    static constexpr double MOVS_BODY_MAX = 0.70;
    static constexpr double LB_TAP_THRESHOLD = 0.03;
    static constexpr double LB_TAP_PAUSE_MIN = 0.01;
    static constexpr double LB_TAP_PAUSE_MAX = 0.06;
    static constexpr double LB_CHECK_WAIT_MIN = 0.10;
    static constexpr double LB_CHECK_WAIT_MAX = 0.20;
    static constexpr double LB_SPRAY_MIN = 0.0;
    static constexpr double LB_SPRAY_MAX = 1.0;
    static constexpr double LB_OVS_WAIT_MIN = 0.10;
    static constexpr double LB_OVS_WAIT_MAX = 0.20;
    static constexpr double HEADOVS_LEFT_MIN = 0.5;
    static constexpr double HEADOVS_LEFT_MAX = 0.6;
    static constexpr double HEADOVS_RIGHT_MIN = 0.4;
    static constexpr double HEADOVS_RIGHT_MAX = 0.5;
    static constexpr double BODYOVS_LEFT_MIN = 0.55;
    static constexpr double BODYOVS_LEFT_MAX = 0.75;
    static constexpr double BODYOVS_RIGHT_MIN = 0.25;
    static constexpr double BODYOVS_RIGHT_MAX = 0.45;
    static constexpr double BREGION_HEAD_TOP = 1.0;
    static constexpr double BREGION_HEAD_BOTTOM = 0.85;
    static constexpr double BREGION_BODY_BOTTOM = 0.4;
    static constexpr double BREGION_LEG_BOTTOM = 0.0;
    static constexpr double INERTIA_DUR_MIN = 0.015;
    static constexpr double INERTIA_DUR_MAX = 0.030;
    static constexpr double INERTIA_DECAY_RATE = 0.85;

    bool   cfg_enableRandomPosition_;
    bool   cfg_enableOvershoot_;
    bool   cfg_enableMicroOvershoot_;
    bool   cfg_enableInertiaStop_;
    bool   cfg_enableLeftClickAdaptive_;
    bool   cfg_enableSprayMode_;
    bool   cfg_enableTapPause_;
    bool   cfg_enableRetry_;
    int    cfg_maxRetryCount_;
    int    cfg_targetDelayMs_;
    double cfg_directProb_;
    double cfg_overshootProb_;
    double cfg_microOvshootProb_;
    double cfg_dyMinRatio_;
    double cfg_dyDefaultRatio_;
    double cfg_dyUpperLimit_;

    std::mt19937 gen_;
    std::uniform_real_distribution<double> dis_;

    MotionPhase current_phase_;
    LeftBtnState left_btn_state_;

    double rect_x_, rect_y_, rect_w_, rect_h_;
    double calculated_distance_;
    double clamped_distance_;
    double motion_duration_;
    double pause_duration_;
    double start_ratio_;
    double target_ratio_;
    double current_ratio_;
    bool is_fast_then_slow_;
    double fast_phase_ratio_;
    int retry_count_;
    int overshoot_retry_count_;
    bool is_second_direct_phase_;

    std::chrono::steady_clock::time_point phase_start_time_;
    std::chrono::steady_clock::time_point left_btn_press_time_;

    double left_btn_hold_duration_;
    bool left_btn_pressed_;
    bool left_btn_prev_state_;
    bool is_in_tap_pause_;
    std::chrono::steady_clock::time_point tap_pause_start_time_;
    double tap_pause_duration_;

    double last_rect_x_, last_rect_y_, last_rect_w_, last_rect_h_;
    bool last_rect_valid_;

    bool is_initialized_;
    double center_x_, center_y_;
    bool has_target_timing_;
    std::chrono::steady_clock::time_point target_start_time_;

    bool enable_debug_print_;
    int frame_counter_;

    double image_width_, image_height_, image_area_;
    double current_overshoot_ratio_;

    bool is_left_btn_locked_;
    bool has_entered_correction_;

    std::chrono::steady_clock::time_point overshoot_wait_start_time_;
    double overshoot_wait_max_duration_;
    double saved_ratio_before_wait_;

    MotionPhase next_phase_after_lb_check_;
    double lb_check_wait_duration_;

    double overshoot_magnitude_;

    double inertia_duration_;
    double inertia_initial_velocity_;
    double inertia_start_ratio_;
    MotionPhase next_phase_after_inertia_;

    double rectCx()     const { return rect_x_ + rect_w_ * 0.5; }
    double rectCy()     const { return rect_y_ + rect_h_ * 0.5; }
    double rectRight()  const { return rect_x_ + rect_w_; }
    double rectBottom() const { return rect_y_ + rect_h_; }
    double rectArea()   const { return rect_w_ * rect_h_; }

    double rng() { return dis_(gen_); }
    double rngRange(double lo, double hi) { return lo + rng() * (hi - lo); }
    double randomDuration(double lo, double hi, double jitter = 0.1) {
        double base = rngRange(lo, hi);
        double vary = base * (rng() * jitter * 2.0 - jitter);
        return std::clamp(base + vary, lo * (1.0 - jitter), hi * (1.0 + jitter));
    }

    static double easeInQuad(double t) { return t * t; }
    static double easeOutQuad(double t) { return t * (2.0 - t); }
    static double easeInOutQuad(double t) {
        return t < 0.5 ? 2.0 * t * t : 1.0 - 2.0 * (1.0 - t) * (1.0 - t);
    }
    static double easeInCubic(double t) { return t * t * t; }

    int getBodyRegion() const;
    double getRelativeX() const;
    bool isInPreciseRange(double relative_x, int body_region) const;
    double calculateDistance() const;
    double clampDistance(double distance) const;

    void updateOvershootRatio();
    int computeDy() const;
    void updateLeftBtnState(bool current_pressed);
    double calculateSprayAdaptive() const;

    double calculateDirectDuration(double distance);
    double calculateOvershootDuration(double distance);
    bool shouldUseFastThenSlow(double distance);
    double calculatePositionRatio(double progress);

    double generateTargetRatio();
    double generateSecondTargetRatio();
    double generateOvershootRatio();

    void initializeMotion();
    void enterDirectPhase(bool is_second_phase);
    void enterOvershootPhase();
    void enterMicroOvershootPhase();
    void enterLeftBtnCheckWait(bool left_btn_pressed, MotionPhase next_phase);
    void enterInertiaPhase(MotionPhase next_phase);
    void enterOvershootRetryFlow();
    void enterFirstStageFlow();

    double processMotionFlow(bool left_btn_input);
    double handleDirect(double elapsed, bool left_btn_input);
    double handleLbCheckWait(double elapsed, bool left_btn_input);
    double handleDirectPause1(double elapsed);
    double handleDirectPause2(double elapsed);
    double handleOvershoot(double elapsed, bool left_btn_input);
    double handleOvershootLbWait(double elapsed, bool left_btn_input);
    double handleOvershootPause1(double elapsed);
    double handleOvershootPause2(double elapsed);
    double handleOvershootPause3(double elapsed);
    double handleMicroOvershoot(double elapsed, bool left_btn_input);
    double handleMicroOvershootPause(double elapsed);
    double handleInertia(double elapsed);

public:
    MotionSimulator();

    void configSwitches(bool randomPos, bool overshoot, bool microOvs, bool inertia,
                        bool lbAdaptive, bool spray, bool tapPause, bool retry);
    void configParams(int maxRetry, int delayMs,
                      double directProb, double overshootProb, double microOvsProb);
    void configDy(double minRatio, double defaultRatio, double upperLimit);

    void initializeImage(int frame_width, int frame_height);
    void setImageCenter(double cx, double cy);

    void setDebugPrint(bool enable);
    bool checkTargetDelay(size_t target_size);
    void reset();
    void onTargetLost();

    void tick(double box_x, double box_y, double box_w, double box_h,
             bool left_btn_input);
    int lastDx() const { return cached_dx_; }
    int lastDy() const { return cached_dy_; }

private:
    int cached_dx_ = 0;
    int cached_dy_ = 0;
};

#endif // MOTION_SIMULATOR_HPP
