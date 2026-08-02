// 配置数据模型序列化 — 嵌套 JSON + 合并。
// 结构: {capture, infer, tracker, vision, aim{slots[], globals…}}
// from_json 只覆盖存在的键,缺省保留默认值。

#include "ConfigModel.hpp"

namespace ya {
namespace config {

namespace {

template <typename T>
void set_if_present(const nlohmann::json& j, const char* key, T& v)
{
    auto it = j.find(key);
    if (it != j.end() && !it->is_null()) {
        try {
            v = it->get<T>();
        } catch (...) {
        }
    }
}

void set_if_present(const nlohmann::json& j, const char* key, std::string& v)
{
    auto it = j.find(key);
    if (it != j.end() && it->is_string())
        v = it->get<std::string>();
}

// 单字段写出
#define W(field) j[#field] = v.field
// 单字段读入
#define R(field) set_if_present(j, #field, v.field)

} // namespace

// ---- Capture ----
nlohmann::json capture_to_json(const CaptureSection& s)
{
    const auto& v = s;
    nlohmann::json j;
    W(backend); W(mode); W(use_region);
    W(region_x); W(region_y); W(region_width); W(region_height);
    W(width); W(height); W(max_fps);
    return j;
}

bool capture_from_json(CaptureSection& s, const nlohmann::json& j)
{
    auto& v = s;
    R(backend); R(mode); R(use_region);
    R(region_x); R(region_y); R(region_width); R(region_height);
    R(width); R(height); R(max_fps);
    return true;
}

// ---- Infer ----
nlohmann::json infer_to_json(const InferSection& s)
{
    const auto& v = s;
    nlohmann::json j;
    W(enabled); W(model_path); W(device); W(model_version);
    W(confidence); W(nms); W(input_resolution); W(num_threads);
    W(interval_frames); W(target_classes);
    return j;
}

bool infer_from_json(InferSection& s, const nlohmann::json& j)
{
    auto& v = s;
    R(enabled); R(model_path); R(device); R(model_version);
    R(confidence); R(nms); R(input_resolution); R(num_threads);
    R(interval_frames); R(target_classes);
    return true;
}

// ---- Tracker ----
nlohmann::json tracker_to_json(const TrackerSection& s)
{
    const auto& v = s;
    nlohmann::json j;
    W(iou_threshold); W(max_lost_frames); W(max_reidentify_frames);
    W(reidentify_center_threshold);
    W(weight_iou); W(weight_center); W(weight_aspect); W(weight_area);
    W(use_kalman); W(kalman_generate_threshold); W(kalman_terminate_count);
    W(kalman_prediction_frames); W(show_kalman_predictions); W(show_kalman_trajectories);
    return j;
}

bool tracker_from_json(TrackerSection& s, const nlohmann::json& j)
{
    auto& v = s;
    R(iou_threshold); R(max_lost_frames); R(max_reidentify_frames);
    R(reidentify_center_threshold);
    R(weight_iou); R(weight_center); R(weight_aspect); R(weight_area);
    R(use_kalman); R(kalman_generate_threshold); R(kalman_terminate_count);
    R(kalman_prediction_frames); R(show_kalman_predictions); R(show_kalman_trajectories);
    return true;
}

// ---- Vision ----
nlohmann::json vision_to_json(const VisionSection& s)
{
    const auto& v = s;
    nlohmann::json j;
    W(show_detection_results); W(bbox_line_width); W(label_font_scale);
    W(export_coordinates); W(coordinate_output_path);
    W(show_floating_window); W(floating_window_width); W(floating_window_height);
    W(show_track_id_in_floating_window); W(preview_enabled);
    return j;
}

bool vision_from_json(VisionSection& s, const nlohmann::json& j)
{
    auto& v = s;
    R(show_detection_results); R(bbox_line_width); R(label_font_scale);
    R(export_coordinates); R(coordinate_output_path);
    R(show_floating_window); R(floating_window_width); R(floating_window_height);
    R(show_track_id_in_floating_window); R(preview_enabled);
    return true;
}

// ---- AimSlot(MouseControllerConfig 全量) ----
nlohmann::json slot_to_json(const AimSlot& s, int index)
{
    nlohmann::json j;
    j["index"] = index;
    j["enabled"] = s.enabled;
    j["continuous_aim"] = s.continuous_aim;
    const auto& v = s.mc;
    j["mc"] = nlohmann::json{
        {"enableMouseControl", v.enableMouseControl},
        {"hotkeyVirtualKey", v.hotkeyVirtualKey},
        {"fovRadiusPixels", v.fovRadiusPixels},
        {"sourceCanvasPosX", v.sourceCanvasPosX},
        {"sourceCanvasPosY", v.sourceCanvasPosY},
        {"sourceCanvasScaleX", v.sourceCanvasScaleX},
        {"sourceCanvasScaleY", v.sourceCanvasScaleY},
        {"sourceWidth", v.sourceWidth},
        {"sourceHeight", v.sourceHeight},
        {"inferenceFrameWidth", v.inferenceFrameWidth},
        {"inferenceFrameHeight", v.inferenceFrameHeight},
        {"cropOffsetX", v.cropOffsetX},
        {"cropOffsetY", v.cropOffsetY},
        {"screenOffsetX", v.screenOffsetX},
        {"screenOffsetY", v.screenOffsetY},
        {"screenWidth", v.screenWidth},
        {"screenHeight", v.screenHeight},
        {"algorithmType", static_cast<int>(v.algorithmType)},
        {"pidPMin", v.pidPMin},
        {"pidPMax", v.pidPMax},
        {"pidPSlope", v.pidPSlope},
        {"pidD", v.pidD},
        {"pidI", v.pidI},
        {"maxPixelMove", v.maxPixelMove},
        {"deadZonePixels", v.deadZonePixels},
        {"targetYOffset", v.targetYOffset},
        {"derivativeFilterAlpha", v.derivativeFilterAlpha},
        {"adaptivePGainRate", v.adaptivePGainRate},
        {"dTermScale", v.dTermScale},
        {"controllerType", static_cast<int>(v.controllerType)},
        {"makcuPort", v.makcuPort},
        {"makcuBaudRate", v.makcuBaudRate},
        {"logiDriverType", v.logiDriverType},
        {"yUnlockDelayMs", v.yUnlockDelayMs},
        {"yUnlockEnabled", v.yUnlockEnabled},
        {"autoTriggerEnabled", v.autoTriggerEnabled},
        {"autoTriggerRadius", v.autoTriggerRadius},
        {"autoTriggerCooldownMs", v.autoTriggerCooldownMs},
        {"autoTriggerFireDelay", v.autoTriggerFireDelay},
        {"autoTriggerFireDuration", v.autoTriggerFireDuration},
        {"autoTriggerInterval", v.autoTriggerInterval},
        {"autoTriggerDelayRandomEnabled", v.autoTriggerDelayRandomEnabled},
        {"autoTriggerDelayRandomMin", v.autoTriggerDelayRandomMin},
        {"autoTriggerDelayRandomMax", v.autoTriggerDelayRandomMax},
        {"autoTriggerDurationRandomEnabled", v.autoTriggerDurationRandomEnabled},
        {"autoTriggerDurationRandomMin", v.autoTriggerDurationRandomMin},
        {"autoTriggerDurationRandomMax", v.autoTriggerDurationRandomMax},
        {"autoTriggerMoveCompensation", v.autoTriggerMoveCompensation},
        {"targetSwitchDelayMs", v.targetSwitchDelayMs},
        {"targetSwitchTolerance", v.targetSwitchTolerance},
        {"integralLimit", v.integralLimit},
        {"integralRate", v.integralRate},
        {"pGainRampInitialScale", v.pGainRampInitialScale},
        {"pGainRampDuration", v.pGainRampDuration},
        {"useDerivativePredictor", v.useDerivativePredictor},
        {"predictionWeightX", v.predictionWeightX},
        {"predictionWeightY", v.predictionWeightY},
        {"maxPredictionTime", v.maxPredictionTime},
        {"smithPredictorEnabled", v.smithPredictorEnabled},
        {"smithModelGain", v.smithModelGain},
        {"smithModelTau", v.smithModelTau},
        {"smithAutoTau", v.smithAutoTau},
        {"immFilterEnabled", v.immFilterEnabled},
        {"immProcessNoisePos", v.immProcessNoisePos},
        {"immProcessNoiseVel", v.immProcessNoiseVel},
        {"immProcessNoiseAcc", v.immProcessNoiseAcc},
        {"immProcessNoiseTurn", v.immProcessNoiseTurn},
        {"immMeasurementNoiseX", v.immMeasurementNoiseX},
        {"immMeasurementNoiseY", v.immMeasurementNoiseY},
        {"immActiveModels", v.immActiveModels},
        {"useOneEuroFilter", v.useOneEuroFilter},
        {"oneEuroMinCutoff", v.oneEuroMinCutoff},
        {"oneEuroBeta", v.oneEuroBeta},
        {"oneEuroDCutoff", v.oneEuroDCutoff},
        {"slewRateEnabled", v.slewRateEnabled},
        {"slewRateOutputGain", v.slewRateOutputGain},
        {"slewRateResponseSmoothing", v.slewRateResponseSmoothing},
        {"slewRateApproachDamping", v.slewRateApproachDamping},
        {"slewRateUpdateIntervalMs", v.slewRateUpdateIntervalMs},
        {"slewRateNormalizationScale", v.slewRateNormalizationScale},
        {"adaptivePidKp", v.adaptivePidKp},
        {"adaptivePidKi", v.adaptivePidKi},
        {"adaptivePidKd", v.adaptivePidKd},
        {"adaptivePidDeadZone", v.adaptivePidDeadZone},
        {"adaptivePidIntegralLimit", v.adaptivePidIntegralLimit},
        {"adaptivePidIntegralDeadzone", v.adaptivePidIntegralDeadzone},
        {"adaptivePidIntegralGainThreshold", v.adaptivePidIntegralGainThreshold},
        {"adaptivePidIntegralGainRate", v.adaptivePidIntegralGainRate},
        {"adaptivePidOutputLimit", v.adaptivePidOutputLimit},
        {"continuousAimEnabled", v.continuousAimEnabled},
        {"autoRecoilControlEnabled", v.autoRecoilControlEnabled},
        {"recoilStrength", v.recoilStrength},
        {"recoilSpeed", v.recoilSpeed},
        {"recoilPidGainScale", v.recoilPidGainScale},
        {"enableBezierMovement", v.enableBezierMovement},
        {"bezierCurvature", v.bezierCurvature},
        {"bezierRandomness", v.bezierRandomness},
        {"enableGhostTracker", v.enableGhostTracker},
        {"ghostCurvature", v.ghostCurvature},
        {"ghostNoiseIntensity", v.ghostNoiseIntensity},
        {"ghostVerticalSnapRatio", v.ghostVerticalSnapRatio},
        {"ghostNoiseFreq", v.ghostNoiseFreq},
        {"externalKpX", v.externalKpX},
        {"externalKiX", v.externalKiX},
        {"externalKdX", v.externalKdX},
        {"externalKpY", v.externalKpY},
        {"externalKiY", v.externalKiY},
        {"externalKdY", v.externalKdY},
        {"externalPredictX", v.externalPredictX},
        {"externalPredictY", v.externalPredictY},
        {"externalRateX", v.externalRateX},
        {"externalRateY", v.externalRateY},
        {"externalKiMode", v.externalKiMode},
        {"externalKpLimit", v.externalKpLimit},
        {"externalKiLimit", v.externalKiLimit},
        {"externalKdLimit", v.externalKdLimit},
        {"externalOutputLimit", v.externalOutputLimit},
        {"externalKiRate", v.externalKiRate},
        {"externalKiDeadband", v.externalKiDeadband},
        {"aimKp", v.aimKp},
        {"aimKi", v.aimKi},
        {"aimKd", v.aimKd},
    };
    return j;
}

bool slot_from_json(AimSlot& s, const nlohmann::json& j, int index)
{
    {
        auto it = j.find("enabled");
        if (it != j.end() && it->is_boolean())
            s.enabled = it->get<bool>();
    }
    set_if_present(j, "continuous_aim", s.continuous_aim);

    auto it = j.find("mc");
    if (it == j.end() || !it->is_object())
        return true; // 只有 enabled/continuous 也算合法 patch
    const nlohmann::json& m = *it;
    auto& v = s.mc;
#define MR(field) set_if_present(m, #field, v.field)
    MR(enableMouseControl); MR(hotkeyVirtualKey); MR(fovRadiusPixels);
    MR(sourceCanvasPosX); MR(sourceCanvasPosY); MR(sourceCanvasScaleX); MR(sourceCanvasScaleY);
    MR(sourceWidth); MR(sourceHeight); MR(inferenceFrameWidth); MR(inferenceFrameHeight);
    MR(cropOffsetX); MR(cropOffsetY); MR(screenOffsetX); MR(screenOffsetY);
    MR(screenWidth); MR(screenHeight);
    {
        int t = static_cast<int>(v.algorithmType);
        set_if_present(m, "algorithmType", t);
        v.algorithmType = static_cast<AlgorithmType>(t);
    }
    MR(pidPMin); MR(pidPMax); MR(pidPSlope); MR(pidD); MR(pidI);
    MR(maxPixelMove); MR(deadZonePixels); MR(targetYOffset); MR(derivativeFilterAlpha);
    MR(adaptivePGainRate); MR(dTermScale);
    {
        int t = static_cast<int>(v.controllerType);
        set_if_present(m, "controllerType", t);
        v.controllerType = static_cast<ControllerType>(t);
    }
    MR(makcuPort); MR(makcuBaudRate); MR(logiDriverType);
    MR(yUnlockDelayMs); MR(yUnlockEnabled);
    MR(autoTriggerEnabled); MR(autoTriggerRadius); MR(autoTriggerCooldownMs);
    MR(autoTriggerFireDelay); MR(autoTriggerFireDuration); MR(autoTriggerInterval);
    MR(autoTriggerDelayRandomEnabled); MR(autoTriggerDelayRandomMin); MR(autoTriggerDelayRandomMax);
    MR(autoTriggerDurationRandomEnabled); MR(autoTriggerDurationRandomMin);
    MR(autoTriggerDurationRandomMax); MR(autoTriggerMoveCompensation);
    MR(targetSwitchDelayMs); MR(targetSwitchTolerance);
    MR(integralLimit); MR(integralRate); MR(pGainRampInitialScale); MR(pGainRampDuration);
    MR(useDerivativePredictor); MR(predictionWeightX); MR(predictionWeightY); MR(maxPredictionTime);
    MR(smithPredictorEnabled); MR(smithModelGain); MR(smithModelTau); MR(smithAutoTau);
    MR(immFilterEnabled); MR(immProcessNoisePos); MR(immProcessNoiseVel); MR(immProcessNoiseAcc);
    MR(immProcessNoiseTurn); MR(immMeasurementNoiseX); MR(immMeasurementNoiseY); MR(immActiveModels);
    MR(useOneEuroFilter); MR(oneEuroMinCutoff); MR(oneEuroBeta); MR(oneEuroDCutoff);
    MR(slewRateEnabled); MR(slewRateOutputGain); MR(slewRateResponseSmoothing);
    MR(slewRateApproachDamping); MR(slewRateUpdateIntervalMs); MR(slewRateNormalizationScale);
    MR(adaptivePidKp); MR(adaptivePidKi); MR(adaptivePidKd); MR(adaptivePidDeadZone);
    MR(adaptivePidIntegralLimit); MR(adaptivePidIntegralDeadzone);
    MR(adaptivePidIntegralGainThreshold); MR(adaptivePidIntegralGainRate);
    MR(adaptivePidOutputLimit);
    MR(continuousAimEnabled); MR(autoRecoilControlEnabled); MR(recoilStrength);
    MR(recoilSpeed); MR(recoilPidGainScale);
    MR(enableBezierMovement); MR(bezierCurvature); MR(bezierRandomness);
    MR(enableGhostTracker); MR(ghostCurvature); MR(ghostNoiseIntensity);
    MR(ghostVerticalSnapRatio); MR(ghostNoiseFreq);
    MR(externalKpX); MR(externalKiX); MR(externalKdX);
    MR(externalKpY); MR(externalKiY); MR(externalKdY);
    MR(externalPredictX); MR(externalPredictY); MR(externalRateX); MR(externalRateY);
    MR(externalKiMode); MR(externalKpLimit); MR(externalKiLimit); MR(externalKdLimit);
    MR(externalOutputLimit); MR(externalKiRate); MR(externalKiDeadband);
    MR(aimKp); MR(aimKi); MR(aimKd);
#undef MR
    return true;
}

// ---- Aim(全局) ----
nlohmann::json aim_to_json(const AimSection& s)
{
    const auto& v = s;
    nlohmann::json j;
    W(enabled); W(config_select); W(algorithm);
    W(fov_radius); W(show_fov); W(show_fov_circle); W(show_fov_cross);
    W(fov_cross_line_scale); W(fov_cross_line_thickness); W(fov_circle_thickness);
    W(fov_color); W(use_dynamic_fov); W(show_fov2); W(fov_radius2); W(fov_color2);
    W(dynamic_fov_shrink_percent); W(dynamic_fov_transition_ms);
    W(target_switch_delay_ms); W(target_switch_tolerance);
    W(external_kp_x); W(external_ki_x); W(external_kd_x);
    W(external_kp_y); W(external_ki_y); W(external_kd_y);
    W(external_predict_x); W(external_predict_y);
    W(external_rate_x); W(external_rate_y);
    W(external_ki_mode); W(external_kp_limit); W(external_ki_limit); W(external_kd_limit);
    W(external_output_limit); W(external_ki_rate); W(external_ki_deadband);
    W(aim_kp); W(aim_ki); W(aim_kd);
    W(aim_noise_enabled); W(aim_noise_amplitude);
    W(aim_prediction_weight_x); W(aim_prediction_weight_y);
    W(aim_ramp_time); W(aim_init_scale); W(aim_output_max);
    W(enable_neural_path); W(neural_path_points); W(neural_mouse_step_size);
    W(neural_target_radius); W(neural_consume_per_frame); W(enable_neural_path_debug);
    W(crosshair_enabled); W(aim_origin_x); W(aim_origin_y);
    W(crosshair_h_min); W(crosshair_h_max);
    W(crosshair_s_min); W(crosshair_s_max);
    W(crosshair_v_min); W(crosshair_v_max);
    W(crosshair_manual_r); W(crosshair_manual_g); W(crosshair_manual_b);
    W(crosshair_h_tolerance); W(crosshair_s_tolerance); W(crosshair_v_tolerance);
    W(crosshair_morph_kernel); W(crosshair_erode_iter); W(crosshair_dilate_iter);
    W(crosshair_grid_rows); W(crosshair_grid_cols);
    W(crosshair_quantile_threshold); W(crosshair_template_path);
    W(crosshair_match_threshold); W(crosshair_min_area); W(crosshair_max_area);
    W(crosshair_shape_filter_enabled); W(crosshair_shape_type);
    W(crosshair_min_fill_ratio); W(crosshair_max_fill_ratio);
    W(crosshair_min_aspect_ratio); W(crosshair_max_aspect_ratio);
    W(crosshair_detect_interval); W(crosshair_search_radius);
    W(crosshair_color_isolation); W(crosshair_debug_mask);
    nlohmann::json slots = nlohmann::json::array();
    for (int i = 0; i < AimSection::kSlots; ++i)
        slots.push_back(slot_to_json(s.slots[i], i));
    j["slots"] = slots;
    return j;
}

bool aim_from_json(AimSection& s, const nlohmann::json& j)
{
    auto& v = s;
    R(enabled); R(config_select); R(algorithm);
    R(fov_radius); R(show_fov); R(show_fov_circle); R(show_fov_cross);
    R(fov_cross_line_scale); R(fov_cross_line_thickness); R(fov_circle_thickness);
    R(fov_color); R(use_dynamic_fov); R(show_fov2); R(fov_radius2); R(fov_color2);
    R(dynamic_fov_shrink_percent); R(dynamic_fov_transition_ms);
    R(target_switch_delay_ms); R(target_switch_tolerance);
    R(external_kp_x); R(external_ki_x); R(external_kd_x);
    R(external_kp_y); R(external_ki_y); R(external_kd_y);
    R(external_predict_x); R(external_predict_y);
    R(external_rate_x); R(external_rate_y);
    R(external_ki_mode); R(external_kp_limit); R(external_ki_limit); R(external_kd_limit);
    R(external_output_limit); R(external_ki_rate); R(external_ki_deadband);
    R(aim_kp); R(aim_ki); R(aim_kd);
    R(aim_noise_enabled); R(aim_noise_amplitude);
    R(aim_prediction_weight_x); R(aim_prediction_weight_y);
    R(aim_ramp_time); R(aim_init_scale); R(aim_output_max);
    R(enable_neural_path); R(neural_path_points); R(neural_mouse_step_size);
    R(neural_target_radius); R(neural_consume_per_frame); R(enable_neural_path_debug);
    R(crosshair_enabled); R(aim_origin_x); R(aim_origin_y);
    R(crosshair_h_min); R(crosshair_h_max);
    R(crosshair_s_min); R(crosshair_s_max);
    R(crosshair_v_min); R(crosshair_v_max);
    R(crosshair_manual_r); R(crosshair_manual_g); R(crosshair_manual_b);
    R(crosshair_h_tolerance); R(crosshair_s_tolerance); R(crosshair_v_tolerance);
    R(crosshair_morph_kernel); R(crosshair_erode_iter); R(crosshair_dilate_iter);
    R(crosshair_grid_rows); R(crosshair_grid_cols);
    R(crosshair_quantile_threshold); R(crosshair_template_path);
    R(crosshair_match_threshold); R(crosshair_min_area); R(crosshair_max_area);
    R(crosshair_shape_filter_enabled); R(crosshair_shape_type);
    R(crosshair_min_fill_ratio); R(crosshair_max_fill_ratio);
    R(crosshair_min_aspect_ratio); R(crosshair_max_aspect_ratio);
    R(crosshair_detect_interval); R(crosshair_search_radius);
    R(crosshair_color_isolation); R(crosshair_debug_mask);
    auto sit = j.find("slots");
    if (sit != j.end() && sit->is_array()) {
        for (size_t i = 0; i < sit->size() && i < AimSection::kSlots; ++i) {
            if ((*sit)[i].is_object())
                slot_from_json(s.slots[i], (*sit)[i], static_cast<int>(i));
        }
    }
    return true;
}

// ---- 顶层 ----
nlohmann::json document_to_json(const ConfigDocument& d)
{
    nlohmann::json j;
    j["capture"] = capture_to_json(d.capture);
    j["infer"] = infer_to_json(d.infer);
    j["tracker"] = tracker_to_json(d.tracker);
    j["vision"] = vision_to_json(d.vision);
    j["aim"] = aim_to_json(d.aim);
    return j;
}

bool document_from_json(ConfigDocument& d, const nlohmann::json& j)
{
    if (j.contains("capture") && j["capture"].is_object())
        capture_from_json(d.capture, j["capture"]);
    if (j.contains("infer") && j["infer"].is_object())
        infer_from_json(d.infer, j["infer"]);
    if (j.contains("tracker") && j["tracker"].is_object())
        tracker_from_json(d.tracker, j["tracker"]);
    if (j.contains("vision") && j["vision"].is_object())
        vision_from_json(d.vision, j["vision"]);
    if (j.contains("aim") && j["aim"].is_object())
        aim_from_json(d.aim, j["aim"]);
    return true;
}

#undef W
#undef R

} // namespace config
} // namespace ya
