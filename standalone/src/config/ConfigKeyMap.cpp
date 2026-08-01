// OBS 键映射实现 — 同构移植旧 ConfigStore(扁平键),目标改为嵌套 ConfigDocument。

#include "ConfigKeyMap.hpp"

#include <cstdlib>
#include <cstring>

namespace ya {
namespace config {

namespace {

using SlotFn = void (*)(AimSlot&, const nlohmann::json&);
using GlobalFn = void (*)(ConfigDocument&, const nlohmann::json&);

template <typename T>
void setv(T& out, const nlohmann::json& v)
{
    try {
        out = v.get<std::decay_t<T>>();
    } catch (...) {
    }
}

// 从 {base}_{slot} / {base}.{slot} 中找 "base" 的值
bool get_slot_value(const nlohmann::json& j, const char* base, int slot, const nlohmann::json*& out)
{
    std::string k = std::string(base) + "_" + std::to_string(slot);
    auto it = j.find(k);
    if (it != j.end()) {
        out = &*it;
        return true;
    }
    k = std::string(base) + "." + std::to_string(slot);
    it = j.find(k);
    if (it != j.end()) {
        out = &*it;
        return true;
    }
    return false;
}

struct SlotEntry {
    const char* name;
    SlotFn apply;
};

void apply_bool(AimSlot& s, const nlohmann::json& v) { setv(s.enabled, v); }
void apply_continuous(AimSlot& s, const nlohmann::json& v) { setv(s.continuous_aim, v); }

void apply_controller_type(AimSlot& s, const nlohmann::json& v)
{
    int t = 0;
    setv(t, v);
    s.mc.controllerType = static_cast<ControllerType>(t);
}

void apply_hotkey(AimSlot& s, const nlohmann::json& v) { setv(s.mc.hotkeyVirtualKey, v); }

// 槽字段条目
#define SLOT_ENTRY(name, member) \
    static_cast<SlotFn>([](AimSlot& s, const nlohmann::json& v) { setv(s.mc.member, v); })

const SlotEntry kSlotEntries[] = {
    {"enable_config", apply_bool},
    {"continuous_aim", apply_continuous},
    {"hotkey", apply_hotkey},
    {"controller_type", apply_controller_type},
    {"logi_driver_type", SLOT_ENTRY("logi_driver_type", logiDriverType)},
    {"makcu_port", SLOT_ENTRY("makcu_port", makcuPort)},
    {"makcu_baud_rate", SLOT_ENTRY("makcu_baud_rate", makcuBaudRate)},
    {"p_min", SLOT_ENTRY("p_min", pidPMin)},
    {"p_max", SLOT_ENTRY("p_max", pidPMax)},
    {"p_slope", SLOT_ENTRY("p_slope", pidPSlope)},
    {"d", SLOT_ENTRY("d", pidD)},
    {"i", SLOT_ENTRY("i", pidI)},
    {"derivative_filter_alpha", SLOT_ENTRY("derivative_filter_alpha", derivativeFilterAlpha)},
    {"adaptive_p_gain_rate", SLOT_ENTRY("adaptive_p_gain_rate", adaptivePGainRate)},
    {"d_term_scale", SLOT_ENTRY("d_term_scale", dTermScale)},
    {"target_y_offset", SLOT_ENTRY("target_y_offset", targetYOffset)},
    {"max_pixel_move", SLOT_ENTRY("max_pixel_move", maxPixelMove)},
    {"dead_zone_pixels", SLOT_ENTRY("dead_zone_pixels", deadZonePixels)},
    {"screen_offset_x", SLOT_ENTRY("screen_offset_x", screenOffsetX)},
    {"screen_offset_y", SLOT_ENTRY("screen_offset_y", screenOffsetY)},
    {"screen_width", SLOT_ENTRY("screen_width", screenWidth)},
    {"screen_height", SLOT_ENTRY("screen_height", screenHeight)},
    {"enable_y_axis_unlock", SLOT_ENTRY("enable_y_axis_unlock", yUnlockEnabled)},
    {"y_axis_unlock_delay", SLOT_ENTRY("y_axis_unlock_delay", yUnlockDelayMs)},
    {"trigger_radius", SLOT_ENTRY("trigger_radius", autoTriggerRadius)},
    {"trigger_cooldown", SLOT_ENTRY("trigger_cooldown", autoTriggerCooldownMs)},
    {"trigger_fire_delay", SLOT_ENTRY("trigger_fire_delay", autoTriggerFireDelay)},
    {"trigger_fire_duration", SLOT_ENTRY("trigger_fire_duration", autoTriggerFireDuration)},
    {"trigger_interval", SLOT_ENTRY("trigger_interval", autoTriggerInterval)},
    {"enable_trigger_delay_random", SLOT_ENTRY("enable_trigger_delay_random", autoTriggerDelayRandomEnabled)},
    {"trigger_delay_random_min", SLOT_ENTRY("trigger_delay_random_min", autoTriggerDelayRandomMin)},
    {"trigger_delay_random_max", SLOT_ENTRY("trigger_delay_random_max", autoTriggerDelayRandomMax)},
    {"enable_trigger_duration_random", SLOT_ENTRY("enable_trigger_duration_random", autoTriggerDurationRandomEnabled)},
    {"trigger_duration_random_min", SLOT_ENTRY("trigger_duration_random_min", autoTriggerDurationRandomMin)},
    {"trigger_duration_random_max", SLOT_ENTRY("trigger_duration_random_max", autoTriggerDurationRandomMax)},
    {"trigger_move_compensation", SLOT_ENTRY("trigger_move_compensation", autoTriggerMoveCompensation)},
    {"integral_limit", SLOT_ENTRY("integral_limit", integralLimit)},
    {"integral_rate", SLOT_ENTRY("integral_rate", integralRate)},
    {"p_gain_ramp_initial_scale", SLOT_ENTRY("p_gain_ramp_initial_scale", pGainRampInitialScale)},
    {"p_gain_ramp_duration", SLOT_ENTRY("p_gain_ramp_duration", pGainRampDuration)},
    {"prediction_weight_x", SLOT_ENTRY("prediction_weight_x", predictionWeightX)},
    {"prediction_weight_y", SLOT_ENTRY("prediction_weight_y", predictionWeightY)},
    {"recoil_strength", SLOT_ENTRY("recoil_strength", recoilStrength)},
    {"recoil_speed", SLOT_ENTRY("recoil_speed", recoilSpeed)},
    {"recoil_pid_gain_scale", SLOT_ENTRY("recoil_pid_gain_scale", recoilPidGainScale)},
    {"derivative_predictor_group", SLOT_ENTRY("derivative_predictor_group", useDerivativePredictor)},
    {"max_prediction_time", SLOT_ENTRY("max_prediction_time", maxPredictionTime)},
    {"smith_predictor_group", SLOT_ENTRY("smith_predictor_group", smithPredictorEnabled)},
    {"smith_enabled", SLOT_ENTRY("smith_enabled", smithPredictorEnabled)},
    {"smith_model_gain", SLOT_ENTRY("smith_model_gain", smithModelGain)},
    {"smith_model_tau", SLOT_ENTRY("smith_model_tau", smithModelTau)},
    {"smith_auto_tau", SLOT_ENTRY("smith_auto_tau", smithAutoTau)},
    {"slew_rate_group", SLOT_ENTRY("slew_rate_group", slewRateEnabled)},
    {"slew_rate_output_gain", SLOT_ENTRY("slew_rate_output_gain", slewRateOutputGain)},
    {"slew_rate_response_smoothing", SLOT_ENTRY("slew_rate_response_smoothing", slewRateResponseSmoothing)},
    {"slew_rate_approach_damping", SLOT_ENTRY("slew_rate_approach_damping", slewRateApproachDamping)},
    {"slew_rate_update_interval_ms", SLOT_ENTRY("slew_rate_update_interval_ms", slewRateUpdateIntervalMs)},
    {"slew_rate_normalization_scale", SLOT_ENTRY("slew_rate_normalization_scale", slewRateNormalizationScale)},
    {"adaptive_pid_kp", SLOT_ENTRY("adaptive_pid_kp", adaptivePidKp)},
    {"adaptive_pid_ki", SLOT_ENTRY("adaptive_pid_ki", adaptivePidKi)},
    {"adaptive_pid_kd", SLOT_ENTRY("adaptive_pid_kd", adaptivePidKd)},
    {"adaptive_pid_dead_zone", SLOT_ENTRY("adaptive_pid_dead_zone", adaptivePidDeadZone)},
    {"adaptive_pid_integral_limit", SLOT_ENTRY("adaptive_pid_integral_limit", adaptivePidIntegralLimit)},
    {"adaptive_pid_integral_deadzone", SLOT_ENTRY("adaptive_pid_integral_deadzone", adaptivePidIntegralDeadzone)},
    {"adaptive_pid_integral_gain_threshold", SLOT_ENTRY("adaptive_pid_integral_gain_threshold", adaptivePidIntegralGainThreshold)},
    {"adaptive_pid_integral_gain_rate", SLOT_ENTRY("adaptive_pid_integral_gain_rate", adaptivePidIntegralGainRate)},
    {"adaptive_pid_output_limit", SLOT_ENTRY("adaptive_pid_output_limit", adaptivePidOutputLimit)},
    {"bezier_movement_group", SLOT_ENTRY("bezier_movement_group", enableBezierMovement)},
    {"bezier_curvature", SLOT_ENTRY("bezier_curvature", bezierCurvature)},
    {"bezier_randomness", SLOT_ENTRY("bezier_randomness", bezierRandomness)},
    {"ghost_tracker_group", SLOT_ENTRY("ghost_tracker_group", enableGhostTracker)},
    {"ghost_curvature", SLOT_ENTRY("ghost_curvature", ghostCurvature)},
    {"ghost_noise_intensity", SLOT_ENTRY("ghost_noise_intensity", ghostNoiseIntensity)},
    {"ghost_vertical_snap", SLOT_ENTRY("ghost_vertical_snap", ghostVerticalSnapRatio)},
    {"ghost_noise_freq", SLOT_ENTRY("ghost_noise_freq", ghostNoiseFreq)},
    {"imm_filter_group", SLOT_ENTRY("imm_filter_group", immFilterEnabled)},
    {"imm_process_noise_pos", SLOT_ENTRY("imm_process_noise_pos", immProcessNoisePos)},
    {"imm_process_noise_vel", SLOT_ENTRY("imm_process_noise_vel", immProcessNoiseVel)},
    {"imm_process_noise_acc", SLOT_ENTRY("imm_process_noise_acc", immProcessNoiseAcc)},
    {"imm_process_noise_turn", SLOT_ENTRY("imm_process_noise_turn", immProcessNoiseTurn)},
    {"imm_measurement_noise_x", SLOT_ENTRY("imm_measurement_noise_x", immMeasurementNoiseX)},
    {"imm_measurement_noise_y", SLOT_ENTRY("imm_measurement_noise_y", immMeasurementNoiseY)},
    {"imm_active_models", SLOT_ENTRY("imm_active_models", immActiveModels)},
    {"use_one_euro_filter", SLOT_ENTRY("use_one_euro_filter", useOneEuroFilter)},
    {"one_euro_min_cutoff", SLOT_ENTRY("one_euro_min_cutoff", oneEuroMinCutoff)},
    {"one_euro_beta", SLOT_ENTRY("one_euro_beta", oneEuroBeta)},
    {"one_euro_d_cutoff", SLOT_ENTRY("one_euro_d_cutoff", oneEuroDCutoff)},
    {"auto_trigger_group", SLOT_ENTRY("auto_trigger_group", autoTriggerEnabled)},
    {"recoil_group", SLOT_ENTRY("recoil_group", autoRecoilControlEnabled)},
};

#undef SLOT_ENTRY

void apply_algorithm(AimSection& a, const nlohmann::json& v)
{
    int t = 0;
    setv(t, v);
    a.algorithm = static_cast<AlgorithmType>(t);
}

// 全局键条目
#define G_ENTRY(name, section, member) \
    static_cast<GlobalFn>([](ConfigDocument& d, const nlohmann::json& v) { setv(d.section.member, v); })

struct GlobalEntry {
    const char* name;
    GlobalFn apply;
};

const GlobalEntry kGlobalEntries[] = {
    // capture
    {"backend", [](ConfigDocument& d, const nlohmann::json& v) {
         std::string s;
         setv(s, v);
         if (!s.empty())
             d.capture.backend = s;
     }},
    {"mode", G_ENTRY("mode", capture, mode)},
    {"use_region", G_ENTRY("use_region", capture, use_region)},
    {"region_x", G_ENTRY("region_x", capture, region_x)},
    {"region_y", G_ENTRY("region_y", capture, region_y)},
    {"region_width", G_ENTRY("region_width", capture, region_width)},
    {"region_height", G_ENTRY("region_height", capture, region_height)},
    // infer
    {"use_gpu", [](ConfigDocument& d, const nlohmann::json& v) {
         std::string s;
         setv(s, v);
         if (!s.empty())
             d.infer.device = s;
     }},
    {"device", G_ENTRY("device", infer, device)},
    {"model_path", G_ENTRY("model_path", infer, model_path)},
    {"model_version", G_ENTRY("model_version", infer, model_version)},
    {"confidence_threshold", G_ENTRY("confidence_threshold", infer, confidence)},
    {"nms_threshold", G_ENTRY("nms_threshold", infer, nms)},
    {"input_resolution", G_ENTRY("input_resolution", infer, input_resolution)},
    {"num_threads", G_ENTRY("num_threads", infer, num_threads)},
    {"inference_interval_frames", G_ENTRY("inference_interval_frames", infer, interval_frames)},
    {"target_classes_text", [](ConfigDocument& d, const nlohmann::json& v) {
         std::string s;
         setv(s, v);
         d.infer.target_classes.clear();
         const char* p = s.c_str();
         while (*p) {
             while (*p && (*p == ',' || *p == ' ' || *p == ';'))
                 ++p;
             if (*p)
                 d.infer.target_classes.push_back(std::atoi(p));
             while (*p && *p != ',' && *p != ';')
                 ++p;
         }
     }},
    // tracker
    {"iou_threshold", G_ENTRY("iou_threshold", tracker, iou_threshold)},
    {"max_lost_frames", G_ENTRY("max_lost_frames", tracker, max_lost_frames)},
    {"max_reidentify_frames", G_ENTRY("max_reidentify_frames", tracker, max_reidentify_frames)},
    {"reidentify_center_threshold", G_ENTRY("reidentify_center_threshold", tracker, reidentify_center_threshold)},
    {"use_kalman_tracker", G_ENTRY("use_kalman_tracker", tracker, use_kalman)},
    {"kalman_generate_threshold", G_ENTRY("kalman_generate_threshold", tracker, kalman_generate_threshold)},
    {"kalman_terminate_count", G_ENTRY("kalman_terminate_count", tracker, kalman_terminate_count)},
    {"kalman_prediction_frames", G_ENTRY("kalman_prediction_frames", tracker, kalman_prediction_frames)},
    // vision
    {"bbox_line_width", G_ENTRY("bbox_line_width", vision, bbox_line_width)},
    {"label_font_scale", G_ENTRY("label_font_scale", vision, label_font_scale)},
    {"export_coordinates", G_ENTRY("export_coordinates", vision, export_coordinates)},
    {"coordinate_output_path", G_ENTRY("coordinate_output_path", vision, coordinate_output_path)},
    {"show_floating_window", G_ENTRY("show_floating_window", vision, show_floating_window)},
    {"floating_window_width", G_ENTRY("floating_window_width", vision, floating_window_width)},
    {"floating_window_height", G_ENTRY("floating_window_height", vision, floating_window_height)},
    {"show_track_id_in_floating_window", G_ENTRY("show_track_id_in_floating_window", vision, show_track_id_in_floating_window)},
    // aim 全局
    {"aim_enabled", G_ENTRY("aim_enabled", aim, enabled)},
    {"mouse_config_select", G_ENTRY("mouse_config_select", aim, config_select)},
    {"algorithm_type_global", [](ConfigDocument& d, const nlohmann::json& v) {
         apply_algorithm(d.aim, v);
     }},
    {"algorithm", [](ConfigDocument& d, const nlohmann::json& v) {
         apply_algorithm(d.aim, v);
     }},
    {"fov_radius", G_ENTRY("fov_radius", aim, fov_radius)},
    {"show_fov", G_ENTRY("show_fov", aim, show_fov)},
    {"show_fov_circle", G_ENTRY("show_fov_circle", aim, show_fov_circle)},
    {"show_fov_cross", G_ENTRY("show_fov_cross", aim, show_fov_cross)},
    {"fov_cross_line_scale", G_ENTRY("fov_cross_line_scale", aim, fov_cross_line_scale)},
    {"fov_cross_line_thickness", G_ENTRY("fov_cross_line_thickness", aim, fov_cross_line_thickness)},
    {"fov_circle_thickness", G_ENTRY("fov_circle_thickness", aim, fov_circle_thickness)},
    {"fov_color", G_ENTRY("fov_color", aim, fov_color)},
    {"use_dynamic_fov", G_ENTRY("use_dynamic_fov", aim, use_dynamic_fov)},
    {"show_fov2", G_ENTRY("show_fov2", aim, show_fov2)},
    {"fov_radius2", G_ENTRY("fov_radius2", aim, fov_radius2)},
    {"fov_color2", G_ENTRY("fov_color2", aim, fov_color2)},
    {"dynamic_fov_shrink_percent", [](ConfigDocument& d, const nlohmann::json& v) {
         // OBS 存 10–100,模型存 0.1–1.0
         float f = 0.f;
         setv(f, v);
         if (f > 1.0f)
             f /= 100.f;
         d.aim.dynamic_fov_shrink_percent = f;
     }},
    {"dynamic_fov_transition_time", [](ConfigDocument& d, const nlohmann::json& v) {
         float f = 0.f;
         setv(f, v);
         d.aim.dynamic_fov_transition_ms = f;
     }},
    {"target_switch_delay_ms", G_ENTRY("target_switch_delay_ms", aim, target_switch_delay_ms)},
    {"target_switch_tolerance", G_ENTRY("target_switch_tolerance", aim, target_switch_tolerance)},
    // external PID
    {"external_kp_x", G_ENTRY("external_kp_x", aim, external_kp_x)},
    {"external_ki_x", G_ENTRY("external_ki_x", aim, external_ki_x)},
    {"external_kd_x", G_ENTRY("external_kd_x", aim, external_kd_x)},
    {"external_kp_y", G_ENTRY("external_kp_y", aim, external_kp_y)},
    {"external_ki_y", G_ENTRY("external_ki_y", aim, external_ki_y)},
    {"external_kd_y", G_ENTRY("external_kd_y", aim, external_kd_y)},
    {"external_predict_x", G_ENTRY("external_predict_x", aim, external_predict_x)},
    {"external_predict_y", G_ENTRY("external_predict_y", aim, external_predict_y)},
    {"external_rate_x", G_ENTRY("external_rate_x", aim, external_rate_x)},
    {"external_rate_y", G_ENTRY("external_rate_y", aim, external_rate_y)},
    {"external_ki_mode", G_ENTRY("external_ki_mode", aim, external_ki_mode)},
    {"external_kp_limit", G_ENTRY("external_kp_limit", aim, external_kp_limit)},
    {"external_ki_limit", G_ENTRY("external_ki_limit", aim, external_ki_limit)},
    {"external_kd_limit", G_ENTRY("external_kd_limit", aim, external_kd_limit)},
    {"external_output_limit", G_ENTRY("external_output_limit", aim, external_output_limit)},
    {"external_ki_rate", G_ENTRY("external_ki_rate", aim, external_ki_rate)},
    {"external_ki_deadband", G_ENTRY("external_ki_deadband", aim, external_ki_deadband)},
    // aim 控制器
    {"aim_kp", G_ENTRY("aim_kp", aim, aim_kp)},
    {"aim_ki", G_ENTRY("aim_ki", aim, aim_ki)},
    {"aim_kd", G_ENTRY("aim_kd", aim, aim_kd)},
    {"aim_noise_enabled", G_ENTRY("aim_noise_enabled", aim, aim_noise_enabled)},
    {"aim_noise_amplitude", G_ENTRY("aim_noise_amplitude", aim, aim_noise_amplitude)},
    {"aim_prediction_weight_x", G_ENTRY("aim_prediction_weight_x", aim, aim_prediction_weight_x)},
    {"aim_prediction_weight_y", G_ENTRY("aim_prediction_weight_y", aim, aim_prediction_weight_y)},
    {"aim_ramp_time", G_ENTRY("aim_ramp_time", aim, aim_ramp_time)},
    {"aim_init_scale", G_ENTRY("aim_init_scale", aim, aim_init_scale)},
    {"aim_output_max", G_ENTRY("aim_output_max", aim, aim_output_max)},
    // neural
    {"enable_neural_path", G_ENTRY("enable_neural_path", aim, enable_neural_path)},
    {"neural_path_points", G_ENTRY("neural_path_points", aim, neural_path_points)},
    {"neural_mouse_step_size", G_ENTRY("neural_mouse_step_size", aim, neural_mouse_step_size)},
    {"neural_target_radius", G_ENTRY("neural_target_radius", aim, neural_target_radius)},
    {"neural_consume_per_frame", G_ENTRY("neural_consume_per_frame", aim, neural_consume_per_frame)},
    {"enable_neural_path_debug", G_ENTRY("enable_neural_path_debug", aim, enable_neural_path_debug)},
    // crosshair
    {"crosshair_enabled", G_ENTRY("crosshair_enabled", aim, crosshair_enabled)},
    {"crosshair_h_min", G_ENTRY("crosshair_h_min", aim, crosshair_h_min)},
    {"crosshair_h_max", G_ENTRY("crosshair_h_max", aim, crosshair_h_max)},
    {"crosshair_s_min", G_ENTRY("crosshair_s_min", aim, crosshair_s_min)},
    {"crosshair_s_max", G_ENTRY("crosshair_s_max", aim, crosshair_s_max)},
    {"crosshair_v_min", G_ENTRY("crosshair_v_min", aim, crosshair_v_min)},
    {"crosshair_v_max", G_ENTRY("crosshair_v_max", aim, crosshair_v_max)},
    {"crosshair_manual_r", G_ENTRY("crosshair_manual_r", aim, crosshair_manual_r)},
    {"crosshair_manual_g", G_ENTRY("crosshair_manual_g", aim, crosshair_manual_g)},
    {"crosshair_manual_b", G_ENTRY("crosshair_manual_b", aim, crosshair_manual_b)},
    {"crosshair_h_tolerance", G_ENTRY("crosshair_h_tolerance", aim, crosshair_h_tolerance)},
    {"crosshair_s_tolerance", G_ENTRY("crosshair_s_tolerance", aim, crosshair_s_tolerance)},
    {"crosshair_v_tolerance", G_ENTRY("crosshair_v_tolerance", aim, crosshair_v_tolerance)},
    {"crosshair_morph_kernel", G_ENTRY("crosshair_morph_kernel", aim, crosshair_morph_kernel)},
    {"crosshair_erode_iter", G_ENTRY("crosshair_erode_iter", aim, crosshair_erode_iter)},
    {"crosshair_dilate_iter", G_ENTRY("crosshair_dilate_iter", aim, crosshair_dilate_iter)},
    {"crosshair_grid_rows", G_ENTRY("crosshair_grid_rows", aim, crosshair_grid_rows)},
    {"crosshair_grid_cols", G_ENTRY("crosshair_grid_cols", aim, crosshair_grid_cols)},
    {"crosshair_quantile_threshold", G_ENTRY("crosshair_quantile_threshold", aim, crosshair_quantile_threshold)},
    {"crosshair_template_path", G_ENTRY("crosshair_template_path", aim, crosshair_template_path)},
    {"crosshair_match_threshold", G_ENTRY("crosshair_match_threshold", aim, crosshair_match_threshold)},
    {"crosshair_min_area", G_ENTRY("crosshair_min_area", aim, crosshair_min_area)},
    {"crosshair_max_area", G_ENTRY("crosshair_max_area", aim, crosshair_max_area)},
    {"crosshair_shape_filter_enabled", G_ENTRY("crosshair_shape_filter_enabled", aim, crosshair_shape_filter_enabled)},
    {"crosshair_shape_type", G_ENTRY("crosshair_shape_type", aim, crosshair_shape_type)},
    {"crosshair_min_fill_ratio", G_ENTRY("crosshair_min_fill_ratio", aim, crosshair_min_fill_ratio)},
    {"crosshair_max_fill_ratio", G_ENTRY("crosshair_max_fill_ratio", aim, crosshair_max_fill_ratio)},
    {"crosshair_min_aspect_ratio", G_ENTRY("crosshair_min_aspect_ratio", aim, crosshair_min_aspect_ratio)},
    {"crosshair_max_aspect_ratio", G_ENTRY("crosshair_max_aspect_ratio", aim, crosshair_max_aspect_ratio)},
    {"crosshair_detect_interval", G_ENTRY("crosshair_detect_interval", aim, crosshair_detect_interval)},
    {"crosshair_search_radius", G_ENTRY("crosshair_search_radius", aim, crosshair_search_radius)},
    {"crosshair_color_isolation", G_ENTRY("crosshair_color_isolation", aim, crosshair_color_isolation)},
    {"crosshair_debug_mask", G_ENTRY("crosshair_debug_mask", aim, crosshair_debug_mask)},
};

#undef G_ENTRY

} // namespace

int parse_slot_suffix(const std::string& key)
{
    const size_t p = key.rfind('_');
    if (p == std::string::npos || p + 1 >= key.size())
        return -1;
    const char* num = key.c_str() + p + 1;
    if (*num < '0' || *num > '9')
        return -1;
    int v = 0;
    for (; *num >= '0' && *num <= '9'; ++num)
        v = v * 10 + (*num - '0');
    if (*num != '\0')
        return -1;
    return v;
}

void apply_obs_slot_keys(ConfigDocument& d, const nlohmann::json& j, int slot)
{
    if (slot < 0 || slot >= AimSection::kSlots)
        return;
    AimSlot& s = d.aim.slots[slot];
    for (const auto& e : kSlotEntries) {
        const nlohmann::json* v = nullptr;
        if (get_slot_value(j, e.name, slot, v))
            e.apply(s, *v);
    }
    // 点键形式 mouse_config_<slot>.p_min
    const std::string prefix = "mouse_config_" + std::to_string(slot) + ".";
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& k = it.key();
        if (k.rfind(prefix, 0) != 0)
            continue;
        const std::string base = k.substr(prefix.size());
        for (const auto& e : kSlotEntries) {
            if (base == e.name) {
                e.apply(s, it.value());
                break;
            }
        }
    }
}

void apply_obs_global_keys(ConfigDocument& d, const nlohmann::json& j)
{
    for (const auto& e : kGlobalEntries) {
        auto it = j.find(e.name);
        if (it != j.end() && !it->is_null())
            e.apply(d, *it);
    }
}

bool apply_flat_obs_keys(ConfigDocument& d, const nlohmann::json& flat, std::string* err)
{
    if (!flat.is_object()) {
        if (err)
            *err = "patch must be an object";
        return false;
    }
    apply_obs_global_keys(d, flat);
    for (auto it = flat.begin(); it != flat.end(); ++it) {
        const std::string& k = it.key();
        const int slot = parse_slot_suffix(k);
        if (slot >= 0) {
            // 槽键:{base}_{slot};去掉 _slot 再查表
            const std::string base = k.substr(0, k.size() - std::to_string(slot).size() - 1);
            bool matched = false;
            for (const auto& e : kSlotEntries) {
                if (base == e.name) {
                    e.apply(d.aim.slots[slot], it.value());
                    matched = true;
                    break;
                }
            }
            (void)matched;
        }
        // 其余键由 apply_obs_global_keys 处理(已扫描);点键 mouse_config_N.x 在此处理
        const size_t dot = k.find('.');
        if (dot != std::string::npos && k.rfind("mouse_config_", 0) == 0) {
            const std::string idx = k.substr(13, dot - 13);
            const int s = std::atoi(idx.c_str());
            if (s >= 0 && s < AimSection::kSlots) {
                const std::string base = k.substr(dot + 1);
                for (const auto& e : kSlotEntries) {
                    if (base == e.name) {
                        e.apply(d.aim.slots[s], it.value());
                        break;
                    }
                }
            }
        }
    }
    return true;
}

} // namespace config
} // namespace ya
