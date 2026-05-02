#include "MotionSimulator.hpp"

MotionSimulator::MotionSimulator()
    : gen_(std::random_device{}()), dis_(0.0, 1.0),
    cfg_enableRandomPosition_(true),
    cfg_enableOvershoot_(true),
    cfg_enableMicroOvershoot_(true),
    cfg_enableInertiaStop_(true),
    cfg_enableLeftClickAdaptive_(true),
    cfg_enableSprayMode_(true),
    cfg_enableTapPause_(true),
    cfg_enableRetry_(true),
    cfg_maxRetryCount_(2),
    cfg_targetDelayMs_(80),
    cfg_directProb_(0.85),
    cfg_overshootProb_(0.10),
    cfg_microOvshootProb_(0.05),
    cfg_dyMinRatio_(0.2),
    cfg_dyDefaultRatio_(0.65),
    cfg_dyUpperLimit_(0.9),
    current_phase_(PHASE_INIT),
    left_btn_state_(LB_IDLE),
    rect_x_(0), rect_y_(0), rect_w_(0), rect_h_(0),
    calculated_distance_(0), clamped_distance_(0),
    motion_duration_(0), pause_duration_(0),
    start_ratio_(0.5), target_ratio_(0.5), current_ratio_(0.5),
    is_fast_then_slow_(false), fast_phase_ratio_(0.7),
    retry_count_(0), overshoot_retry_count_(0), is_second_direct_phase_(false),
    left_btn_hold_duration_(0),
    left_btn_pressed_(false), left_btn_prev_state_(false),
    is_in_tap_pause_(false), tap_pause_duration_(0),
    last_rect_x_(0), last_rect_y_(0), last_rect_w_(0), last_rect_h_(0),
    last_rect_valid_(false),
    is_initialized_(false),
    center_x_(160.0), center_y_(160.0),
    has_target_timing_(false),
    enable_debug_print_(false), frame_counter_(0),
    image_width_(320), image_height_(320), image_area_(320 * 320),
    current_overshoot_ratio_(OVS_MAX_RATIO),
    is_left_btn_locked_(false), has_entered_correction_(false),
    overshoot_wait_max_duration_(0), saved_ratio_before_wait_(0.5),
    next_phase_after_lb_check_(PHASE_DIRECT_PAUSE_1),
    lb_check_wait_duration_(0.0),
    overshoot_magnitude_(0.0),
    inertia_duration_(0.02), inertia_initial_velocity_(0.0),
    inertia_start_ratio_(0.5),
    next_phase_after_inertia_(PHASE_INIT)
{
}

void MotionSimulator::configSwitches(bool randomPos, bool overshoot, bool microOvs, bool inertia,
                                     bool lbAdaptive, bool spray, bool tapPause, bool retry) {
    cfg_enableRandomPosition_   = randomPos;
    cfg_enableOvershoot_        = overshoot;
    cfg_enableMicroOvershoot_   = microOvs;
    cfg_enableInertiaStop_      = inertia;
    cfg_enableLeftClickAdaptive_= lbAdaptive;
    cfg_enableSprayMode_        = spray;
    cfg_enableTapPause_         = tapPause;
    cfg_enableRetry_            = retry;
}

void MotionSimulator::configParams(int maxRetry, int delayMs,
                                   double directProb, double overshootProb, double microOvsProb) {
    cfg_maxRetryCount_  = maxRetry;
    cfg_targetDelayMs_  = delayMs;
    cfg_directProb_     = directProb;
    cfg_overshootProb_  = overshootProb;
    cfg_microOvshootProb_ = microOvsProb;
}

void MotionSimulator::configDy(double minRatio, double defaultRatio, double upperLimit) {
    cfg_dyMinRatio_     = minRatio;
    cfg_dyDefaultRatio_ = defaultRatio;
    cfg_dyUpperLimit_   = upperLimit;
}

void MotionSimulator::setDebugPrint(bool enable) {
    enable_debug_print_ = enable;
}

void MotionSimulator::initializeImage(int frame_width, int frame_height) {
    image_width_ = static_cast<double>(frame_width);
    image_height_ = static_cast<double>(frame_height);
    image_area_ = image_width_ * image_height_;
    center_x_ = image_width_ / 2.0;
    center_y_ = image_height_ / 2.0;
}

void MotionSimulator::setImageCenter(double cx, double cy) {
    center_x_ = cx;
    center_y_ = cy;
}

bool MotionSimulator::checkTargetDelay(size_t target_size) {
    if (target_size > 0) {
        if (!has_target_timing_) {
            target_start_time_ = std::chrono::steady_clock::now();
            has_target_timing_ = true;
        }
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - target_start_time_).count();
        return elapsed_ms >= cfg_targetDelayMs_;
    }
    has_target_timing_ = false;
    return false;
}

void MotionSimulator::reset() {
    current_phase_ = PHASE_INIT;
    left_btn_state_ = LB_IDLE;
    calculated_distance_ = 0; clamped_distance_ = 0;
    motion_duration_ = 0; pause_duration_ = 0;
    start_ratio_ = 0.5; target_ratio_ = 0.5; current_ratio_ = 0.5;
    is_fast_then_slow_ = false;
    retry_count_ = 0; overshoot_retry_count_ = 0; is_second_direct_phase_ = false;
    left_btn_hold_duration_ = 0;
    left_btn_pressed_ = false; left_btn_prev_state_ = false;
    is_in_tap_pause_ = false;
    is_initialized_ = false;
    has_target_timing_ = false;
    last_rect_valid_ = false;
    frame_counter_ = 0;
    current_overshoot_ratio_ = OVS_MAX_RATIO;
    is_left_btn_locked_ = false; has_entered_correction_ = false;
    overshoot_wait_max_duration_ = 0; saved_ratio_before_wait_ = 0.5;
    next_phase_after_lb_check_ = PHASE_DIRECT_PAUSE_1;
    lb_check_wait_duration_ = 0.0;
    overshoot_magnitude_ = 0.0;
    inertia_duration_ = 0.02; inertia_initial_velocity_ = 0.0;
    inertia_start_ratio_ = 0.5;
    next_phase_after_inertia_ = PHASE_INIT;
}

void MotionSimulator::onTargetLost() { reset(); }

void MotionSimulator::tick(
    double box_x, double box_y, double box_w, double box_h,
    bool left_btn_input)
{
    frame_counter_++;
    rect_x_ = box_x; rect_y_ = box_y; rect_w_ = box_w; rect_h_ = box_h;

    updateOvershootRatio();

    bool effective_left_btn = is_left_btn_locked_ ? false : left_btn_input;
    if (cfg_enableLeftClickAdaptive_) {
        updateLeftBtnState(effective_left_btn);
    }

    if (cfg_enableTapPause_ && is_in_tap_pause_) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - tap_pause_start_time_).count();
        if (elapsed < tap_pause_duration_) {
            cached_dx_ = 0;
            cached_dy_ = computeDy();
            return;
        }
        else {
            is_in_tap_pause_ = false;
        }
    }

    if (cfg_enableSprayMode_ && left_btn_state_ == LB_SPRAY) {
        double spray_ratio = calculateSprayAdaptive();
        double dx_final_d = rectCx() - center_x_ + rect_w_ * 0.5 - rect_w_ * spray_ratio;
        cached_dx_ = static_cast<int>(std::round(dx_final_d));
        cached_dy_ = computeDy();
        return;
    }

    last_rect_x_ = rect_x_; last_rect_y_ = rect_y_;
    last_rect_w_ = rect_w_; last_rect_h_ = rect_h_;
    last_rect_valid_ = true;

    double dx_ratio = processMotionFlow(left_btn_input);
    double dx_final_d = rectCx() - center_x_ + rect_w_ * 0.5 - rect_w_ * dx_ratio;
    cached_dx_ = static_cast<int>(std::round(dx_final_d));
    cached_dy_ = computeDy();
}

int MotionSimulator::getBodyRegion() const {
    if (center_y_ < rect_y_) return 0;
    if (center_y_ > rectBottom()) return 1;
    double relative_y = (rectBottom() - center_y_) / rect_h_;
    if (relative_y >= BREGION_HEAD_BOTTOM && relative_y <= BREGION_HEAD_TOP) return 0;
    if (relative_y >= BREGION_BODY_BOTTOM && relative_y < BREGION_HEAD_BOTTOM) return 1;
    if (relative_y >= BREGION_LEG_BOTTOM  && relative_y < BREGION_BODY_BOTTOM) return 2;
    return 1;
}

double MotionSimulator::getRelativeX() const {
    if (center_x_ <= rect_x_) return 1.0;
    if (center_x_ >= rectRight()) return 0.0;
    return (rectRight() - center_x_) / rect_w_;
}

bool MotionSimulator::isInPreciseRange(double relative_x, int body_region) const {
    return (body_region == 0) ?
        (relative_x >= PRECISE_HEAD_MIN && relative_x <= PRECISE_HEAD_MAX) :
        (relative_x >= PRECISE_BODY_MIN && relative_x <= PRECISE_BODY_MAX);
}

double MotionSimulator::calculateDistance() const {
    double dx = rectCx() - center_x_;
    double dy = rectCy() - center_y_;
    return std::sqrt(dx * dx + dy * dy);
}

double MotionSimulator::clampDistance(double distance) const {
    return std::clamp(distance, DIST_MIN, DIST_MAX);
}

void MotionSimulator::updateOvershootRatio() {
    if (!cfg_enableOvershoot_) {
        current_overshoot_ratio_ = 0.0;
        return;
    }
    double ratio = rectArea() / image_area_;
    if (ratio <= SIZE_SMALL) {
        current_overshoot_ratio_ = OVS_MAX_RATIO;
    }
    else if (ratio >= SIZE_LARGE) {
        current_overshoot_ratio_ = OVS_MIN_RATIO;
    }
    else {
        double norm = (ratio - SIZE_SMALL) / (SIZE_LARGE - SIZE_SMALL);
        current_overshoot_ratio_ = OVS_MAX_RATIO - norm * (OVS_MAX_RATIO - OVS_MIN_RATIO);
    }
}

int MotionSimulator::computeDy() const {
    double dy = rectCy() - center_y_;
    double targetTop = rect_y_;
    double targetBottom = rectBottom();
    if (center_y_ >= targetTop && center_y_ <= targetBottom) {
        double relPos = (targetBottom - center_y_) / rect_h_;
        relPos = std::clamp(relPos, cfg_dyMinRatio_, cfg_dyUpperLimit_);
        dy = dy + rect_h_ / 2.0 - rect_h_ * relPos;
    }
    else {
        dy = dy + rect_h_ / 2.0 - rect_h_ * cfg_dyDefaultRatio_;
    }
    return static_cast<int>(std::round(dy));
}

void MotionSimulator::updateLeftBtnState(bool current_pressed) {
    auto now = std::chrono::steady_clock::now();
    if (current_pressed && !left_btn_prev_state_) {
        left_btn_pressed_ = true;
        left_btn_press_time_ = now;
        left_btn_hold_duration_ = 0;
    }
    else if (!current_pressed && left_btn_prev_state_) {
        left_btn_pressed_ = false;
        if (cfg_enableTapPause_ && left_btn_hold_duration_ < LB_TAP_THRESHOLD) {
            left_btn_state_ = LB_TAP;
            is_in_tap_pause_ = true;
            tap_pause_duration_ = rngRange(LB_TAP_PAUSE_MIN, LB_TAP_PAUSE_MAX);
            tap_pause_start_time_ = now;
        }
        else {
            left_btn_state_ = LB_IDLE;
        }
        left_btn_hold_duration_ = 0;
    }
    else if (current_pressed && left_btn_prev_state_) {
        left_btn_hold_duration_ = std::chrono::duration<double>(now - left_btn_press_time_).count();
        if (cfg_enableSprayMode_ && left_btn_hold_duration_ >= LB_TAP_THRESHOLD) {
            double relative_x = getRelativeX();
            if (relative_x >= 0.0 && relative_x <= 1.0) {
                if (left_btn_state_ != LB_SPRAY) {
                    left_btn_state_ = LB_SPRAY;
                }
            }
        }
    }
    left_btn_prev_state_ = current_pressed;
}

double MotionSimulator::calculateSprayAdaptive() const {
    double extended_left = rectRight() - LB_SPRAY_MAX * rect_w_;
    double extended_right = rectRight() - LB_SPRAY_MIN * rect_w_;
    if (center_x_ <= extended_left) return LB_SPRAY_MAX;
    if (center_x_ >= extended_right) return LB_SPRAY_MIN;
    double relative = LB_SPRAY_MAX - (center_x_ - extended_left) / (extended_right - extended_left)
        * (LB_SPRAY_MAX - LB_SPRAY_MIN);
    return std::clamp(relative, LB_SPRAY_MIN, LB_SPRAY_MAX);
}

double MotionSimulator::calculateDirectDuration(double distance) {
    double lo_min, lo_max;
    if (distance <= DIST_MIN) {
        lo_min = DIRECT_D20_MIN; lo_max = DIRECT_D20_MAX;
    }
    else if (distance <= DIST_MID) {
        double r = (distance - DIST_MIN) / (DIST_MID - DIST_MIN);
        lo_min = DIRECT_D20_MIN + r * (DIRECT_D40_MIN - DIRECT_D20_MIN);
        lo_max = DIRECT_D20_MAX + r * (DIRECT_D40_MAX - DIRECT_D20_MAX);
    }
    else if (distance <= DIST_MAX) {
        double r = (distance - DIST_MID) / (DIST_MAX - DIST_MID);
        lo_min = DIRECT_D40_MIN + r * (DIRECT_D60_MIN - DIRECT_D40_MIN);
        lo_max = DIRECT_D40_MAX + r * (DIRECT_D60_MAX - DIRECT_D40_MAX);
    }
    else {
        lo_min = DIRECT_D60_MIN; lo_max = DIRECT_D60_MAX;
    }
    return randomDuration(lo_min, lo_max, 0.05);
}

double MotionSimulator::calculateOvershootDuration(double distance) {
    double lo_min, lo_max;
    if (distance <= DIST_MIN) {
        lo_min = OVSDUR_D20_MIN; lo_max = OVSDUR_D20_MAX;
    }
    else if (distance <= DIST_MID) {
        double r = (distance - DIST_MIN) / (DIST_MID - DIST_MIN);
        lo_min = OVSDUR_D20_MIN + r * (OVSDUR_D40_MIN - OVSDUR_D20_MIN);
        lo_max = OVSDUR_D20_MAX + r * (OVSDUR_D40_MAX - OVSDUR_D20_MAX);
    }
    else if (distance <= DIST_MAX) {
        double r = (distance - DIST_MID) / (DIST_MAX - DIST_MID);
        lo_min = OVSDUR_D40_MIN + r * (OVSDUR_D60_MIN - OVSDUR_D40_MIN);
        lo_max = OVSDUR_D40_MAX + r * (OVSDUR_D60_MAX - OVSDUR_D40_MAX);
    }
    else {
        lo_min = OVSDUR_D60_MIN; lo_max = OVSDUR_D60_MAX;
    }
    return randomDuration(lo_min, lo_max, 0.05);
}

bool MotionSimulator::shouldUseFastThenSlow(double distance) {
    if (distance <= DIST_MIN - 2) return false;
    if (distance >= DIST_MIN + 2) return true;
    return rng() < 0.5;
}

double MotionSimulator::calculatePositionRatio(double progress) {
    if (!is_fast_then_slow_) {
        return start_ratio_ + (target_ratio_ - start_ratio_) * progress;
    }
    if (progress < fast_phase_ratio_) {
        double t = progress / fast_phase_ratio_;
        return start_ratio_ + (target_ratio_ - start_ratio_) * fast_phase_ratio_ * easeOutQuad(t);
    }
    double t = (progress - fast_phase_ratio_) / (1.0 - fast_phase_ratio_);
    double start_val = start_ratio_ + (target_ratio_ - start_ratio_) * fast_phase_ratio_;
    return start_val + (target_ratio_ - start_val) * easeInCubic(t);
}

double MotionSimulator::generateTargetRatio() {
    if (!cfg_enableRandomPosition_) return 0.5;
    return (getBodyRegion() == 0) ? (0.45 + rng() * 0.1) : (0.3 + rng() * 0.4);
}

double MotionSimulator::generateSecondTargetRatio() {
    if (!cfg_enableRandomPosition_) return 0.5;
    return (getBodyRegion() == 0) ?
        rngRange(SECOND_HEAD_MIN, SECOND_HEAD_MAX) :
        rngRange(SECOND_BODY_MIN, SECOND_BODY_MAX);
}

double MotionSimulator::generateOvershootRatio() {
    int body_region = getBodyRegion();
    bool from_left = (center_x_ < rectCx());
    double base_min, base_max;
    if (body_region == 0) {
        base_min = from_left ? HEADOVS_RIGHT_MIN : HEADOVS_LEFT_MIN;
        base_max = from_left ? HEADOVS_RIGHT_MAX : HEADOVS_LEFT_MAX;
    }
    else {
        base_min = from_left ? BODYOVS_RIGHT_MIN : BODYOVS_LEFT_MIN;
        base_max = from_left ? BODYOVS_RIGHT_MAX : BODYOVS_LEFT_MAX;
    }
    return rngRange(base_min, base_max) * current_overshoot_ratio_;
}

void MotionSimulator::enterDirectPhase(bool is_second_phase) {
    current_phase_ = PHASE_DIRECT;
    motion_duration_ = calculateDirectDuration(clamped_distance_);
    is_fast_then_slow_ = shouldUseFastThenSlow(clamped_distance_);
    is_second_direct_phase_ = is_second_phase;

    if (is_second_phase && is_left_btn_locked_) {
        has_entered_correction_ = true;
    }

    if (clamped_distance_ <= DIST_MIN) fast_phase_ratio_ = 1.0;
    else if (clamped_distance_ <= DIST_MID) fast_phase_ratio_ = 0.7 + rng() * 0.1;
    else fast_phase_ratio_ = 0.65 + rng() * 0.1;

    start_ratio_ = current_ratio_;
    target_ratio_ = is_second_phase ? generateSecondTargetRatio() : generateTargetRatio();
}

void MotionSimulator::enterOvershootPhase() {
    current_phase_ = PHASE_OVERSHOOT;

    if (!is_left_btn_locked_) {
        is_left_btn_locked_ = true;
        has_entered_correction_ = false;
    }

    motion_duration_ = calculateOvershootDuration(clamped_distance_);
    is_fast_then_slow_ = shouldUseFastThenSlow(clamped_distance_);

    if (clamped_distance_ <= DIST_MIN) fast_phase_ratio_ = 1.0;
    else if (clamped_distance_ <= DIST_MID) fast_phase_ratio_ = 0.7 + rng() * 0.1;
    else fast_phase_ratio_ = 0.65 + rng() * 0.1;

    start_ratio_ = current_ratio_;
    target_ratio_ = generateOvershootRatio();
}

void MotionSimulator::enterMicroOvershootPhase() {
    current_phase_ = PHASE_MICRO_OVERSHOOT;

    if (!is_left_btn_locked_) {
        is_left_btn_locked_ = true;
        has_entered_correction_ = false;
    }

    motion_duration_ = rngRange(MOVS_DUR_MIN, MOVS_DUR_MAX);

    int body_region = getBodyRegion();
    bool from_left = (center_x_ < rectCx());
    start_ratio_ = current_ratio_;

    double base_min, base_max;
    if (body_region == 0) {
        base_min = from_left ? MOVS_HEAD_MIN : (1.0 - MOVS_HEAD_MAX);
        base_max = from_left ? MOVS_HEAD_MAX : (1.0 - MOVS_HEAD_MIN);
    }
    else {
        base_min = from_left ? MOVS_BODY_MIN : (1.0 - MOVS_BODY_MAX);
        base_max = from_left ? MOVS_BODY_MAX : (1.0 - MOVS_BODY_MIN);
    }
    target_ratio_ = rngRange(base_min, base_max) * current_overshoot_ratio_;
}

void MotionSimulator::enterLeftBtnCheckWait(bool left_btn_pressed, MotionPhase next_phase) {
    if (cfg_enableLeftClickAdaptive_ && left_btn_pressed) {
        lb_check_wait_duration_ = rngRange(LB_CHECK_WAIT_MIN, LB_CHECK_WAIT_MAX);
        next_phase_after_lb_check_ = next_phase;
        current_phase_ = PHASE_LB_CHECK_WAIT;
        phase_start_time_ = std::chrono::steady_clock::now();
    }
    else {
        current_phase_ = next_phase;
        phase_start_time_ = std::chrono::steady_clock::now();
    }
}

void MotionSimulator::enterInertiaPhase(MotionPhase next_phase) {
    if (!cfg_enableInertiaStop_) {
        current_phase_ = next_phase;
        phase_start_time_ = std::chrono::steady_clock::now();
        return;
    }
    double velocity = target_ratio_ - start_ratio_;
    inertia_initial_velocity_ = velocity * 0.15;
    inertia_start_ratio_ = current_ratio_;
    inertia_duration_ = rngRange(INERTIA_DUR_MIN, INERTIA_DUR_MAX);
    next_phase_after_inertia_ = next_phase;
    current_phase_ = PHASE_INERTIA;
    phase_start_time_ = std::chrono::steady_clock::now();
}

void MotionSimulator::enterOvershootRetryFlow() {
    if (cfg_enableRetry_ && overshoot_retry_count_ < cfg_maxRetryCount_) {
        overshoot_retry_count_++;
        double r = rng();
        if (r < OVS_RETRY_OVS_PROB) {
            enterDirectPhase(true);
        }
        else if (cfg_enableOvershoot_) {
            enterOvershootPhase();
        }
        else {
            enterDirectPhase(true);
        }
        phase_start_time_ = std::chrono::steady_clock::now();
    }
    else {
        enterDirectPhase(true);
        phase_start_time_ = std::chrono::steady_clock::now();
    }
}

void MotionSimulator::enterFirstStageFlow() {
    pause_duration_ = randomDuration(DIRECT_PAUSE1_MIN, DIRECT_PAUSE1_MAX);
    current_phase_ = PHASE_DIRECT_PAUSE_1;
    phase_start_time_ = std::chrono::steady_clock::now();
}

void MotionSimulator::initializeMotion() {
    calculated_distance_ = calculateDistance();
    clamped_distance_ = clampDistance(calculated_distance_);
    double target_size_ratio = rectArea() / image_area_;

    if (!cfg_enableRandomPosition_) {
        current_ratio_ = 0.5;
        current_phase_ = PHASE_DIRECT;
        motion_duration_ = 0.01;
        start_ratio_ = 0.5;
        target_ratio_ = 0.5;
        retry_count_ = 0; overshoot_retry_count_ = 0; is_second_direct_phase_ = false;
        return;
    }

    if (target_size_ratio >= SIZE_LARGE || calculated_distance_ <= DIST_IMMEDIATE) {
        enterDirectPhase(false);
        retry_count_ = 0; overshoot_retry_count_ = 0; is_second_direct_phase_ = false;
        return;
    }

    double final_direct_prob = cfg_directProb_;
    if (calculated_distance_ < DIST_LOW_OVERSHOOT) {
        double dist_factor = 1.0 - (calculated_distance_ / DIST_LOW_OVERSHOOT);
        dist_factor = std::clamp(dist_factor, 0.0, 1.0);
        dist_factor *= dist_factor;
        final_direct_prob += (1.0 - final_direct_prob) * dist_factor;
    }

    double r = rng();
    if (r < final_direct_prob) {
        enterDirectPhase(false);
    }
    else if (cfg_enableMicroOvershoot_ && r < final_direct_prob + cfg_microOvshootProb_) {
        enterMicroOvershootPhase();
    }
    else if (cfg_enableOvershoot_) {
        enterOvershootPhase();
    }
    else {
        enterDirectPhase(false);
    }

    retry_count_ = 0; overshoot_retry_count_ = 0; is_second_direct_phase_ = false;
}

double MotionSimulator::processMotionFlow(bool left_btn_input) {
    auto now = std::chrono::steady_clock::now();
    if (!is_initialized_) {
        initializeMotion();
        is_initialized_ = true;
        phase_start_time_ = now;
        return current_ratio_;
    }
    double elapsed = std::chrono::duration<double>(now - phase_start_time_).count();
    switch (current_phase_) {
    case PHASE_INIT:                    break;
    case PHASE_DIRECT:                  return handleDirect(elapsed, left_btn_input);
    case PHASE_LB_CHECK_WAIT:           return handleLbCheckWait(elapsed, left_btn_input);
    case PHASE_DIRECT_PAUSE_1:          return handleDirectPause1(elapsed);
    case PHASE_DIRECT_PAUSE_2:          return handleDirectPause2(elapsed);
    case PHASE_OVERSHOOT:               return handleOvershoot(elapsed, left_btn_input);
    case PHASE_OVERSHOOT_LB_WAIT:       return handleOvershootLbWait(elapsed, left_btn_input);
    case PHASE_OVERSHOOT_PAUSE_1:       return handleOvershootPause1(elapsed);
    case PHASE_OVERSHOOT_PAUSE_2:       return handleOvershootPause2(elapsed);
    case PHASE_OVERSHOOT_PAUSE_3:       return handleOvershootPause3(elapsed);
    case PHASE_MICRO_OVERSHOOT:         return handleMicroOvershoot(elapsed, left_btn_input);
    case PHASE_MICRO_OVERSHOOT_PAUSE:   return handleMicroOvershootPause(elapsed);
    case PHASE_INERTIA:                 return handleInertia(elapsed);
    default:                            return 0.5;
    }
    return current_ratio_;
}

double MotionSimulator::handleDirect(double elapsed, bool left_btn_input) {
    if (elapsed < motion_duration_) {
        current_ratio_ = calculatePositionRatio(elapsed / motion_duration_);
        return current_ratio_;
    }
    pause_duration_ = randomDuration(DIRECT_PAUSE1_MIN, DIRECT_PAUSE1_MAX);
    enterInertiaPhase(PHASE_LB_CHECK_WAIT);
    next_phase_after_lb_check_ = PHASE_DIRECT_PAUSE_1;
    lb_check_wait_duration_ = (cfg_enableLeftClickAdaptive_ && left_btn_input) ?
        rngRange(LB_CHECK_WAIT_MIN, LB_CHECK_WAIT_MAX) : 0.0;
    return current_ratio_;
}

double MotionSimulator::handleLbCheckWait(double elapsed, bool left_btn_input) {
    if (elapsed < lb_check_wait_duration_) return current_ratio_;
    current_phase_ = next_phase_after_lb_check_;
    phase_start_time_ = std::chrono::steady_clock::now();
    return current_ratio_;
}

double MotionSimulator::handleDirectPause1(double elapsed) {
    if (elapsed < pause_duration_) return current_ratio_;

    int body_region = getBodyRegion();
    double relative_x = getRelativeX();
    if (isInPreciseRange(relative_x, body_region)) {
        pause_duration_ = randomDuration(DIRECT_PAUSE2_MIN, DIRECT_PAUSE2_MAX);
        current_phase_ = PHASE_DIRECT_PAUSE_2;
        phase_start_time_ = std::chrono::steady_clock::now();
    }
    else {
        if (cfg_enableRetry_ && retry_count_ < cfg_maxRetryCount_) {
            retry_count_++;
            if (rng() < OVS_RETRY_DIRECT_PROB) {
                enterDirectPhase(false);
            }
            else if (cfg_enableOvershoot_) {
                enterOvershootPhase();
            }
            else {
                enterDirectPhase(false);
            }
            phase_start_time_ = std::chrono::steady_clock::now();
        }
        else {
            current_ratio_ = 0.5;
        }
    }
    return current_ratio_;
}

double MotionSimulator::handleDirectPause2(double elapsed) {
    if (elapsed < pause_duration_) return current_ratio_;

    if (is_left_btn_locked_ && has_entered_correction_) {
        is_left_btn_locked_ = false;
        has_entered_correction_ = false;
    }

    double relative_x = getRelativeX();
    if (relative_x >= 0.0 && relative_x <= 1.0) {
        current_ratio_ = 0.5;
    }
    else {
        reset();
        initializeMotion();
        phase_start_time_ = std::chrono::steady_clock::now();
    }
    return current_ratio_;
}

double MotionSimulator::handleOvershoot(double elapsed, bool left_btn_input) {
    if (cfg_enableLeftClickAdaptive_ && left_btn_input && elapsed > 0.01) {
        saved_ratio_before_wait_ = current_ratio_;
        overshoot_wait_max_duration_ = rngRange(LB_OVS_WAIT_MIN, LB_OVS_WAIT_MAX);
        current_phase_ = PHASE_OVERSHOOT_LB_WAIT;
        phase_start_time_ = std::chrono::steady_clock::now();
        overshoot_wait_start_time_ = phase_start_time_;
        return current_ratio_;
    }

    if (elapsed < motion_duration_) {
        current_ratio_ = calculatePositionRatio(elapsed / motion_duration_);
        return current_ratio_;
    }

    is_left_btn_locked_ = false;
    overshoot_magnitude_ = std::min(std::abs(target_ratio_ - 0.5), 1.0);

    double overshoot_mul = 1.0 + overshoot_magnitude_;
    pause_duration_ = std::clamp(
        rngRange(OVSDUR_PAUSE1_MIN, OVSDUR_PAUSE1_MAX) * overshoot_mul,
        OVSDUR_PAUSE1_MIN, OVSDUR_PAUSE1_MAX * 2.0);

    enterInertiaPhase(PHASE_LB_CHECK_WAIT);
    next_phase_after_lb_check_ = PHASE_OVERSHOOT_PAUSE_1;
    lb_check_wait_duration_ = (cfg_enableLeftClickAdaptive_ && left_btn_input) ?
        rngRange(LB_CHECK_WAIT_MIN, LB_CHECK_WAIT_MAX) : 0.0;
    return current_ratio_;
}

double MotionSimulator::handleOvershootLbWait(double elapsed, bool left_btn_input) {
    if (!left_btn_input || elapsed >= overshoot_wait_max_duration_) {
        is_left_btn_locked_ = false;
        double overshoot_mul = 1.0 + overshoot_magnitude_;
        pause_duration_ = std::clamp(
            rngRange(OVSDUR_PAUSE1_MIN, OVSDUR_PAUSE1_MAX) * overshoot_mul,
            OVSDUR_PAUSE1_MIN, OVSDUR_PAUSE1_MAX * 2.0);
        enterInertiaPhase(PHASE_OVERSHOOT_PAUSE_1);
        return current_ratio_;
    }
    return saved_ratio_before_wait_;
}

double MotionSimulator::handleOvershootPause1(double elapsed) {
    if (elapsed < pause_duration_) return current_ratio_;
    double relative_x = getRelativeX();
    if (relative_x >= 0.0 && relative_x <= 1.0) {
        double overshoot_mul = 1.0 + overshoot_magnitude_ * 0.5;
        pause_duration_ = std::clamp(
            rngRange(OVSDUR_PAUSE2_MIN, OVSDUR_PAUSE2_MAX) * overshoot_mul,
            OVSDUR_PAUSE2_MIN, OVSDUR_PAUSE2_MAX * 1.5);
        current_phase_ = PHASE_OVERSHOOT_PAUSE_2;
        phase_start_time_ = std::chrono::steady_clock::now();
    }
    else {
        enterOvershootRetryFlow();
    }
    return current_ratio_;
}

double MotionSimulator::handleOvershootPause2(double elapsed) {
    if (elapsed < pause_duration_) return current_ratio_;
    double relative_x = getRelativeX();
    if (relative_x >= 0.0 && relative_x <= 1.0) {
        double overshoot_mul = 1.0 + overshoot_magnitude_ * 0.3;
        pause_duration_ = std::clamp(
            rngRange(OVSDUR_PAUSE3_MIN, OVSDUR_PAUSE3_MAX) * overshoot_mul,
            OVSDUR_PAUSE3_MIN, OVSDUR_PAUSE3_MAX * 1.3);
        current_phase_ = PHASE_OVERSHOOT_PAUSE_3;
        phase_start_time_ = std::chrono::steady_clock::now();
    }
    else {
        enterOvershootRetryFlow();
    }
    return current_ratio_;
}

double MotionSimulator::handleOvershootPause3(double elapsed) {
    if (elapsed < pause_duration_) return current_ratio_;
    enterOvershootRetryFlow();
    return current_ratio_;
}

double MotionSimulator::handleMicroOvershoot(double elapsed, bool left_btn_input) {
    if (elapsed < motion_duration_) {
        current_ratio_ = start_ratio_ + (target_ratio_ - start_ratio_) * (elapsed / motion_duration_);
        return current_ratio_;
    }
    pause_duration_ = rngRange(MOVS_PAUSE_MIN, MOVS_PAUSE_MAX);
    enterInertiaPhase(PHASE_LB_CHECK_WAIT);
    next_phase_after_lb_check_ = PHASE_MICRO_OVERSHOOT_PAUSE;
    lb_check_wait_duration_ = (cfg_enableLeftClickAdaptive_ && left_btn_input) ?
        rngRange(LB_CHECK_WAIT_MIN, LB_CHECK_WAIT_MAX) : 0.0;
    return current_ratio_;
}

double MotionSimulator::handleMicroOvershootPause(double elapsed) {
    if (elapsed < pause_duration_) return current_ratio_;
    enterDirectPhase(true);
    phase_start_time_ = std::chrono::steady_clock::now();
    return current_ratio_;
}

double MotionSimulator::handleInertia(double elapsed) {
    if (!cfg_enableInertiaStop_) {
        current_phase_ = next_phase_after_inertia_;
        phase_start_time_ = std::chrono::steady_clock::now();
        return current_ratio_;
    }
    if (elapsed < inertia_duration_) {
        double progress = elapsed / inertia_duration_;
        double decay = std::pow(INERTIA_DECAY_RATE, progress * 10.0);
        current_ratio_ = inertia_start_ratio_ + inertia_initial_velocity_ * decay * (1.0 - progress);
        return current_ratio_;
    }
    current_phase_ = next_phase_after_inertia_;
    phase_start_time_ = std::chrono::steady_clock::now();
    return current_ratio_;
}
