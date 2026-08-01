#include "ConfigStore.hpp"
	
	#include <algorithm>
	#include <filesystem>
	#include <fstream>
	#include <iostream>
	#include <sstream>

namespace ya {
namespace {

using json = nlohmann::json;

// "0,1,2" / "0 1 2" → vector; empty or "all" clears filter (detect all classes)
static void parse_target_classes_text(const std::string &text, std::vector<int> &out)
{
	out.clear();
	std::string t = text;
	// trim
	auto l = t.find_first_not_of(" \t\r\n");
	if (l == std::string::npos) {
		return;
	}
	t = t.substr(l, t.find_last_not_of(" \t\r\n") - l + 1);
	if (t.empty() || t == "all" || t == "*" || t == "-1")
		return;
	for (char &ch : t) {
		if (ch == ';' || ch == '|' || ch == ' ')
			ch = ',';
	}
	std::stringstream ss(t);
	std::string part;
	while (std::getline(ss, part, ',')) {
		if (part.empty())
			continue;
		try {
			out.push_back(std::stoi(part));
		} catch (...) {
		}
	}
}

template <typename T>
void jget(const json &j, const char *key, T &dst)
{
	if (!j.contains(key) || j[key].is_null())
		return;
	try {
		dst = j[key].get<T>();
	} catch (...) {
	}
}

void jget_float(const json &j, const char *key, float &dst)
{
	if (!j.contains(key) || j[key].is_null())
		return;
	try {
		if (j[key].is_number())
			dst = static_cast<float>(j[key].get<double>());
	} catch (...) {
	}
}

void jget_int(const json &j, const char *key, int &dst)
{
	if (!j.contains(key) || j[key].is_null())
		return;
	try {
		if (j[key].is_number_integer())
			dst = j[key].get<int>();
		else if (j[key].is_number())
			dst = static_cast<int>(j[key].get<double>());
	} catch (...) {
	}
}

void jget_bool(const json &j, const char *key, bool &dst)
{
	if (!j.contains(key) || j[key].is_null())
		return;
	try {
		if (j[key].is_boolean())
			dst = j[key].get<bool>();
		else if (j[key].is_number())
			dst = j[key].get<int>() != 0;
	} catch (...) {
	}
}

void jget_str(const json &j, const char *key, std::string &dst)
{
	if (!j.contains(key) || j[key].is_null())
		return;
	try {
		if (j[key].is_string())
			dst = j[key].get<std::string>();
	} catch (...) {
	}
}

// Prefer OBS key; fall back to camelCase alias.
void jget_float2(const json &j, const char *obs, const char *alias, float &dst)
{
	if (j.contains(obs))
		jget_float(j, obs, dst);
	else if (alias)
		jget_float(j, alias, dst);
}
void jget_int2(const json &j, const char *obs, const char *alias, int &dst)
{
	if (j.contains(obs))
		jget_int(j, obs, dst);
	else if (alias)
		jget_int(j, alias, dst);
}
void jget_bool2(const json &j, const char *obs, const char *alias, bool &dst)
{
	if (j.contains(obs))
		jget_bool(j, obs, dst);
	else if (alias)
		jget_bool(j, alias, dst);
}
void jget_str2(const json &j, const char *obs, const char *alias, std::string &dst)
{
	if (j.contains(obs))
		jget_str(j, obs, dst);
	else if (alias)
		jget_str(j, alias, dst);
}

std::string slot_key(const char *base, int i)
{
	return std::string(base) + "_" + std::to_string(i);
}

json profile_to_json(const AimProfile &p, int i)
{
	const auto &m = p.mc;
	json o;
	// OBS keys (primary)
	o["enable_config"] = p.enabled;
	o["continuous_aim"] = p.continuous_aim;
	o["hotkey"] = m.hotkeyVirtualKey;
	o["controller_type"] = static_cast<int>(m.controllerType);
	o["logi_driver_type"] = m.logiDriverType;
	o["makcu_port"] = m.makcuPort;
	o["makcu_baud_rate"] = m.makcuBaudRate;
	o["p_min"] = m.pidPMin;
	o["p_max"] = m.pidPMax;
	o["p_slope"] = m.pidPSlope;
	o["d"] = m.pidD;
	o["i"] = m.pidI;
	o["derivative_filter_alpha"] = m.derivativeFilterAlpha;
	o["adaptive_p_gain_rate"] = m.adaptivePGainRate;
	o["d_term_scale"] = m.dTermScale;
	o["target_y_offset"] = m.targetYOffset;
	o["max_pixel_move"] = m.maxPixelMove;
	o["dead_zone_pixels"] = m.deadZonePixels;
	o["screen_offset_x"] = m.screenOffsetX;
	o["screen_offset_y"] = m.screenOffsetY;
	o["screen_width"] = m.screenWidth;
	o["screen_height"] = m.screenHeight;
	o["enable_y_axis_unlock"] = m.yUnlockEnabled;
	o["y_axis_unlock_delay"] = m.yUnlockDelayMs;
	o["auto_trigger_group"] = m.autoTriggerEnabled;
	o["trigger_radius"] = m.autoTriggerRadius;
	o["trigger_cooldown"] = m.autoTriggerCooldownMs;
	o["trigger_fire_delay"] = m.autoTriggerFireDelay;
	o["trigger_fire_duration"] = m.autoTriggerFireDuration;
	o["trigger_interval"] = m.autoTriggerInterval;
	o["enable_trigger_delay_random"] = m.autoTriggerDelayRandomEnabled;
	o["trigger_delay_random_min"] = m.autoTriggerDelayRandomMin;
	o["trigger_delay_random_max"] = m.autoTriggerDelayRandomMax;
	o["enable_trigger_duration_random"] = m.autoTriggerDurationRandomEnabled;
	o["trigger_duration_random_min"] = m.autoTriggerDurationRandomMin;
	o["trigger_duration_random_max"] = m.autoTriggerDurationRandomMax;
	o["trigger_move_compensation"] = m.autoTriggerMoveCompensation;
	o["integral_limit"] = m.integralLimit;
	o["integral_rate"] = m.integralRate;
	o["p_gain_ramp_initial_scale"] = m.pGainRampInitialScale;
	o["p_gain_ramp_duration"] = m.pGainRampDuration;
	o["prediction_weight_x"] = m.predictionWeightX;
	o["prediction_weight_y"] = m.predictionWeightY;
	o["recoil_group"] = m.autoRecoilControlEnabled;
	o["recoil_strength"] = m.recoilStrength;
	o["recoil_speed"] = m.recoilSpeed;
	o["recoil_pid_gain_scale"] = m.recoilPidGainScale;
	o["derivative_predictor_group"] = m.useDerivativePredictor;
	o["max_prediction_time"] = m.maxPredictionTime;
	o["smith_predictor_group"] = m.smithPredictorEnabled;
	o["smith_model_gain"] = m.smithModelGain;
	o["smith_model_tau"] = m.smithModelTau;
	o["smith_auto_tau"] = m.smithAutoTau;
	o["slew_rate_group"] = m.slewRateEnabled;
	o["slew_rate_output_gain"] = m.slewRateOutputGain;
	o["slew_rate_response_smoothing"] = m.slewRateResponseSmoothing;
	o["slew_rate_approach_damping"] = m.slewRateApproachDamping;
	o["slew_rate_update_interval_ms"] = m.slewRateUpdateIntervalMs;
	o["slew_rate_normalization_scale"] = m.slewRateNormalizationScale;
	o["adaptive_pid_kp"] = m.adaptivePidKp;
	o["adaptive_pid_ki"] = m.adaptivePidKi;
	o["adaptive_pid_kd"] = m.adaptivePidKd;
	o["adaptive_pid_dead_zone"] = m.adaptivePidDeadZone;
	o["adaptive_pid_integral_limit"] = m.adaptivePidIntegralLimit;
	o["adaptive_pid_integral_deadzone"] = m.adaptivePidIntegralDeadzone;
	o["adaptive_pid_integral_gain_threshold"] = m.adaptivePidIntegralGainThreshold;
	o["adaptive_pid_integral_gain_rate"] = m.adaptivePidIntegralGainRate;
	o["adaptive_pid_output_limit"] = m.adaptivePidOutputLimit;
	o["bezier_movement_group"] = m.enableBezierMovement;
	o["bezier_curvature"] = m.bezierCurvature;
	o["bezier_randomness"] = m.bezierRandomness;
	o["ghost_tracker_group"] = m.enableGhostTracker;
	o["ghost_curvature"] = m.ghostCurvature;
	o["ghost_noise_intensity"] = m.ghostNoiseIntensity;
	o["ghost_vertical_snap"] = m.ghostVerticalSnapRatio;
	o["ghost_noise_freq"] = m.ghostNoiseFreq;
	o["imm_filter_enabled"] = m.immFilterEnabled;
	o["imm_process_noise_pos"] = m.immProcessNoisePos;
	o["imm_process_noise_vel"] = m.immProcessNoiseVel;
	o["imm_process_noise_acc"] = m.immProcessNoiseAcc;
	o["imm_process_noise_turn"] = m.immProcessNoiseTurn;
	o["imm_measurement_noise_x"] = m.immMeasurementNoiseX;
	o["imm_measurement_noise_y"] = m.immMeasurementNoiseY;
	o["imm_active_models"] = m.immActiveModels;
	o["use_one_euro_filter"] = m.useOneEuroFilter;
	o["one_euro_min_cutoff"] = m.oneEuroMinCutoff;
	o["one_euro_beta"] = m.oneEuroBeta;
	o["one_euro_d_cutoff"] = m.oneEuroDCutoff;
	// aliases for thin WebUI
	o["enabled"] = p.enabled;
	o["hotkey_vk"] = m.hotkeyVirtualKey;
	o["controller"] = FullAimBridge::controller_name(m.controllerType);
	o["pidPMin"] = m.pidPMin;
	o["pidPMax"] = m.pidPMax;
	o["pidD"] = m.pidD;
	o["pidI"] = m.pidI;
	o["deadZonePixels"] = m.deadZonePixels;
	o["maxPixelMove"] = m.maxPixelMove;
	o["targetYOffset"] = m.targetYOffset;
	o["autoTriggerEnabled"] = m.autoTriggerEnabled;
	o["autoTriggerRadius"] = m.autoTriggerRadius;
	o["autoRecoilControlEnabled"] = m.autoRecoilControlEnabled;
	o["recoilStrength"] = m.recoilStrength;
	o["slot"] = i;
	return o;
}

void apply_profile_object(AimProfile &p, const json &o)
{
	jget_bool2(o, "enable_config", "enabled", p.enabled);
	jget_bool2(o, "continuous_aim", "continuous_aim", p.continuous_aim);
	jget_int2(o, "hotkey", "hotkey_vk", p.mc.hotkeyVirtualKey);

	if (o.contains("controller_type")) {
		int ct = 0;
		jget_int(o, "controller_type", ct);
		p.mc.controllerType = FullAimBridge::parse_controller_int(ct);
	} else if (o.contains("controller")) {
		std::string cs;
		jget_str(o, "controller", cs);
		p.mc.controllerType = FullAimBridge::parse_controller(cs);
	}

	jget_int2(o, "logi_driver_type", "logiDriverType", p.mc.logiDriverType);
	jget_str2(o, "makcu_port", "makcuPort", p.mc.makcuPort);
	jget_int2(o, "makcu_baud_rate", "makcuBaudRate", p.mc.makcuBaudRate);

	jget_float2(o, "p_min", "pidPMin", p.mc.pidPMin);
	jget_float2(o, "p_max", "pidPMax", p.mc.pidPMax);
	jget_float2(o, "p_slope", "pidPSlope", p.mc.pidPSlope);
	jget_float2(o, "d", "pidD", p.mc.pidD);
	jget_float2(o, "i", "pidI", p.mc.pidI);
	jget_float2(o, "derivative_filter_alpha", nullptr, p.mc.derivativeFilterAlpha);
	jget_float2(o, "adaptive_p_gain_rate", nullptr, p.mc.adaptivePGainRate);
	jget_float2(o, "d_term_scale", nullptr, p.mc.dTermScale);
	jget_float2(o, "target_y_offset", "targetYOffset", p.mc.targetYOffset);
	jget_float2(o, "max_pixel_move", "maxPixelMove", p.mc.maxPixelMove);
	jget_float2(o, "dead_zone_pixels", "deadZonePixels", p.mc.deadZonePixels);

	jget_int2(o, "screen_offset_x", nullptr, p.mc.screenOffsetX);
	jget_int2(o, "screen_offset_y", nullptr, p.mc.screenOffsetY);
	jget_int2(o, "screen_width", nullptr, p.mc.screenWidth);
	jget_int2(o, "screen_height", nullptr, p.mc.screenHeight);
	jget_bool2(o, "enable_y_axis_unlock", nullptr, p.mc.yUnlockEnabled);
	jget_int2(o, "y_axis_unlock_delay", nullptr, p.mc.yUnlockDelayMs);

	jget_bool2(o, "auto_trigger_group", "autoTriggerEnabled", p.mc.autoTriggerEnabled);
	jget_int2(o, "trigger_radius", "autoTriggerRadius", p.mc.autoTriggerRadius);
	jget_int2(o, "trigger_cooldown", nullptr, p.mc.autoTriggerCooldownMs);
	jget_int2(o, "trigger_fire_delay", nullptr, p.mc.autoTriggerFireDelay);
	jget_int2(o, "trigger_fire_duration", nullptr, p.mc.autoTriggerFireDuration);
	jget_int2(o, "trigger_interval", nullptr, p.mc.autoTriggerInterval);
	jget_bool2(o, "enable_trigger_delay_random", nullptr, p.mc.autoTriggerDelayRandomEnabled);
	jget_int2(o, "trigger_delay_random_min", nullptr, p.mc.autoTriggerDelayRandomMin);
	jget_int2(o, "trigger_delay_random_max", nullptr, p.mc.autoTriggerDelayRandomMax);
	jget_bool2(o, "enable_trigger_duration_random", nullptr, p.mc.autoTriggerDurationRandomEnabled);
	jget_int2(o, "trigger_duration_random_min", nullptr, p.mc.autoTriggerDurationRandomMin);
	jget_int2(o, "trigger_duration_random_max", nullptr, p.mc.autoTriggerDurationRandomMax);
	jget_int2(o, "trigger_move_compensation", nullptr, p.mc.autoTriggerMoveCompensation);

	jget_float2(o, "integral_limit", nullptr, p.mc.integralLimit);
	jget_float2(o, "integral_rate", nullptr, p.mc.integralRate);
	jget_float2(o, "p_gain_ramp_initial_scale", nullptr, p.mc.pGainRampInitialScale);
	jget_float2(o, "p_gain_ramp_duration", nullptr, p.mc.pGainRampDuration);
	jget_float2(o, "prediction_weight_x", nullptr, p.mc.predictionWeightX);
	jget_float2(o, "prediction_weight_y", nullptr, p.mc.predictionWeightY);

	jget_bool2(o, "recoil_group", "autoRecoilControlEnabled", p.mc.autoRecoilControlEnabled);
	jget_float2(o, "recoil_strength", "recoilStrength", p.mc.recoilStrength);
	jget_int2(o, "recoil_speed", nullptr, p.mc.recoilSpeed);
	jget_float2(o, "recoil_pid_gain_scale", nullptr, p.mc.recoilPidGainScale);

	jget_bool2(o, "derivative_predictor_group", nullptr, p.mc.useDerivativePredictor);
	jget_float2(o, "max_prediction_time", nullptr, p.mc.maxPredictionTime);
	jget_bool2(o, "smith_predictor_group", nullptr, p.mc.smithPredictorEnabled);
	jget_float2(o, "smith_model_gain", nullptr, p.mc.smithModelGain);
	jget_float2(o, "smith_model_tau", nullptr, p.mc.smithModelTau);
	jget_bool2(o, "smith_auto_tau", nullptr, p.mc.smithAutoTau);

	jget_bool2(o, "slew_rate_group", nullptr, p.mc.slewRateEnabled);
	jget_float2(o, "slew_rate_output_gain", nullptr, p.mc.slewRateOutputGain);
	jget_float2(o, "slew_rate_response_smoothing", nullptr, p.mc.slewRateResponseSmoothing);
	jget_float2(o, "slew_rate_approach_damping", nullptr, p.mc.slewRateApproachDamping);
	jget_float2(o, "slew_rate_update_interval_ms", nullptr, p.mc.slewRateUpdateIntervalMs);
	jget_float2(o, "slew_rate_normalization_scale", nullptr, p.mc.slewRateNormalizationScale);

	jget_float2(o, "adaptive_pid_kp", nullptr, p.mc.adaptivePidKp);
	jget_float2(o, "adaptive_pid_ki", nullptr, p.mc.adaptivePidKi);
	jget_float2(o, "adaptive_pid_kd", nullptr, p.mc.adaptivePidKd);
	jget_float2(o, "adaptive_pid_dead_zone", nullptr, p.mc.adaptivePidDeadZone);
	jget_float2(o, "adaptive_pid_integral_limit", nullptr, p.mc.adaptivePidIntegralLimit);
	jget_float2(o, "adaptive_pid_integral_deadzone", nullptr, p.mc.adaptivePidIntegralDeadzone);
	jget_float2(o, "adaptive_pid_integral_gain_threshold", nullptr, p.mc.adaptivePidIntegralGainThreshold);
	jget_float2(o, "adaptive_pid_integral_gain_rate", nullptr, p.mc.adaptivePidIntegralGainRate);
	jget_float2(o, "adaptive_pid_output_limit", nullptr, p.mc.adaptivePidOutputLimit);

	jget_bool2(o, "bezier_movement_group", nullptr, p.mc.enableBezierMovement);
	jget_float2(o, "bezier_curvature", nullptr, p.mc.bezierCurvature);
	jget_float2(o, "bezier_randomness", nullptr, p.mc.bezierRandomness);
	jget_bool2(o, "ghost_tracker_group", nullptr, p.mc.enableGhostTracker);
	jget_float2(o, "ghost_curvature", nullptr, p.mc.ghostCurvature);
	jget_float2(o, "ghost_noise_intensity", nullptr, p.mc.ghostNoiseIntensity);
	jget_float2(o, "ghost_vertical_snap", nullptr, p.mc.ghostVerticalSnapRatio);
	jget_float2(o, "ghost_noise_freq", nullptr, p.mc.ghostNoiseFreq);

	jget_bool2(o, "imm_filter_enabled", nullptr, p.mc.immFilterEnabled);
	jget_float2(o, "imm_process_noise_pos", nullptr, p.mc.immProcessNoisePos);
	jget_float2(o, "imm_process_noise_vel", nullptr, p.mc.immProcessNoiseVel);
	jget_float2(o, "imm_process_noise_acc", nullptr, p.mc.immProcessNoiseAcc);
	jget_float2(o, "imm_process_noise_turn", nullptr, p.mc.immProcessNoiseTurn);
	jget_float2(o, "imm_measurement_noise_x", nullptr, p.mc.immMeasurementNoiseX);
	jget_float2(o, "imm_measurement_noise_y", nullptr, p.mc.immMeasurementNoiseY);
	jget_int2(o, "imm_active_models", nullptr, p.mc.immActiveModels);
	jget_bool2(o, "use_one_euro_filter", nullptr, p.mc.useOneEuroFilter);
	jget_float2(o, "one_euro_min_cutoff", nullptr, p.mc.oneEuroMinCutoff);
	jget_float2(o, "one_euro_beta", nullptr, p.mc.oneEuroBeta);
	jget_float2(o, "one_euro_d_cutoff", nullptr, p.mc.oneEuroDCutoff);

	p.mc.enableMouseControl = p.enabled;
	p.mc.continuousAimEnabled = p.continuous_aim;
}

} // namespace

json ConfigStore::to_json(const EngineConfig &c)
{
	json root;
// capture — nested region + flat OBS keys (use_region / region_x / …)
		root["capture"] = {
		    {"backend", FrameSource::backend_name(c.capture_backend)},
		    {"mode", c.capture_mode},
		    {"width", c.width},
		    {"height", c.height},
		    {"use_region", c.use_region},
		    {"region_x", c.region_x},
		    {"region_y", c.region_y},
		    {"region_width", c.region_width},
		    {"region_height", c.region_height},
		    {"region",
		     {{"x", c.region_x},
		      {"y", c.region_y},
		      {"w", c.region_width > 0 ? c.region_width : c.width},
		      {"h", c.region_height > 0 ? c.region_height : c.height}}},
		};
		// flat OBS aliases at root (page 1 keys)
		root["use_region"] = c.use_region;
		root["region_x"] = c.region_x;
		root["region_y"] = c.region_y;
		root["region_width"] = c.region_width;
		root["region_height"] = c.region_height;
	// infer — also flat OBS-ish aliases
// rebuild text from vector if empty
		std::string classes_text = c.infer.target_classes_text;
		if (classes_text.empty() && !c.infer.target_classes.empty()) {
			std::ostringstream oss;
			for (size_t i = 0; i < c.infer.target_classes.size(); ++i) {
				if (i)
					oss << ',';
				oss << c.infer.target_classes[i];
			}
			classes_text = oss.str();
		}
// OBS single-class combo: -1 = all; else first of target_classes or sole id
			int target_class = -1;
			if (c.infer.target_classes.size() == 1)
				target_class = c.infer.target_classes[0];
			root["infer"] = {
			    {"enabled", c.infer_enabled},
			    {"is_inferencing", c.infer_enabled}, // OBS toggle_inference / is_inferencing
			    {"model_path", c.infer.model_path},
			    {"device", c.infer.device},
			    {"use_gpu", c.infer.device}, // OBS property name
			    {"model_version", c.infer.model_version},
			    {"confidence", c.infer.confidence},
			    {"confidence_threshold", c.infer.confidence},
			    {"nms", c.infer.nms},
			    {"nms_threshold", c.infer.nms},
			    {"input_size", c.infer.input_resolution},
			    {"input_resolution", c.infer.input_resolution},
			    {"interval_frames", c.infer.interval_frames},
			    {"inference_interval_frames", c.infer.interval_frames},
			    {"num_threads", c.infer.num_threads},
			    {"target_class", target_class},
			    {"target_classes_text", classes_text},
			    {"target_classes", c.infer.target_classes},
			};
			// Flat OBS page-0 keys at root (import OBS scene JSON)
			root["model_path"] = c.infer.model_path;
			root["model_version"] = c.infer.model_version;
			root["use_gpu"] = c.infer.device;
			root["input_resolution"] = c.infer.input_resolution;
			root["num_threads"] = c.infer.num_threads;
			root["confidence_threshold"] = c.infer.confidence;
			root["nms_threshold"] = c.infer.nms;
			root["target_class"] = target_class;
			root["target_classes_text"] = classes_text;
			root["inference_interval_frames"] = c.infer.interval_frames;
			root["is_inferencing"] = c.infer_enabled;
	root["models_dir"] = c.models_dir;
	root["model_folder"] = c.models_dir;
	root["model_search_dirs"] = c.model_search_dirs;
	// tracking — OBS keys
	root["tracking"] = {
	    {"iou_threshold", c.tracker.iou_threshold},
	    {"max_lost_frames", c.tracker.max_lost_frames},
	    {"max_reidentify_frames", c.tracker.max_reidentify_frames},
	    {"reidentify_center_threshold", c.tracker.reidentify_center_threshold},
	    {"tracking_weight_iou", c.tracker.weight_iou},
	    {"tracking_weight_center", c.tracker.weight_center},
	    {"tracking_weight_aspect", c.tracker.weight_aspect},
	    {"tracking_weight_area", c.tracker.weight_area},
	    {"weight_iou", c.tracker.weight_iou},
	    {"weight_center", c.tracker.weight_center},
	    {"weight_aspect", c.tracker.weight_aspect},
	    {"weight_area", c.tracker.weight_area},
{"use_kalman", c.tracker.use_kalman},
		    {"use_kalman_tracker", c.tracker.use_kalman},
		    {"kalman_generate_threshold", c.tracker.kalman_generate_threshold},
		    {"kalman_terminate_count", c.tracker.kalman_terminate_count},
		    {"show_kalman_predictions", c.tracker.show_kalman_predictions},
		    {"kalman_prediction_frames", c.tracker.kalman_prediction_frames},
		    {"show_kalman_trajectories", c.tracker.show_kalman_trajectories},
		};

#ifdef YA_WITH_AIM
	const auto &a = c.aim_full;
	json aim;
	aim["enabled"] = c.aim_enabled && a.enabled;
	aim["mouse_config_select"] = a.config_select;
	aim["config_select"] = a.config_select;
	aim["algorithm_type_global"] = static_cast<int>(a.algorithm);
	aim["algorithm"] = static_cast<int>(a.algorithm);
	aim["fov_radius"] = a.fov_radius;
	aim["show_fov"] = a.show_fov;
	aim["show_fov_circle"] = a.show_fov_circle;
	aim["show_fov_cross"] = a.show_fov_cross;
	aim["fov_cross_line_scale"] = a.fov_cross_line_scale;
	aim["fov_cross_line_thickness"] = a.fov_cross_line_thickness;
	aim["fov_circle_thickness"] = a.fov_circle_thickness;
	aim["fov_color"] = a.fov_color;
	aim["use_dynamic_fov"] = a.use_dynamic_fov;
	aim["show_fov2"] = a.show_fov2;
	aim["fov_radius2"] = a.fov_radius2;
	aim["fov_color2"] = a.fov_color2;
	aim["dynamic_fov_shrink_percent"] =
	    static_cast<int>(a.dynamic_fov_shrink_percent * 100.f); // OBS 10–100
	aim["dynamic_fov_transition_time"] = static_cast<int>(a.dynamic_fov_transition_ms);
	aim["target_switch_delay"] = a.target_switch_delay_ms;
	aim["target_switch_tolerance"] = a.target_switch_tolerance;

	// external PID
	aim["external_kp_x"] = a.external_kp_x;
	aim["external_ki_x"] = a.external_ki_x;
	aim["external_kd_x"] = a.external_kd_x;
	aim["external_kp_y"] = a.external_kp_y;
	aim["external_ki_y"] = a.external_ki_y;
	aim["external_kd_y"] = a.external_kd_y;
	aim["external_predict_x"] = a.external_predict_x;
	aim["external_predict_y"] = a.external_predict_y;
	aim["external_rate_x"] = a.external_rate_x;
	aim["external_rate_y"] = a.external_rate_y;
	aim["external_ki_mode"] = static_cast<int>(a.external_ki_mode);
	aim["external_kp_limit"] = a.external_kp_limit;
	aim["external_ki_limit"] = a.external_ki_limit;
	aim["external_kd_limit"] = a.external_kd_limit;
	aim["external_output_limit"] = a.external_output_limit;
	aim["external_ki_rate"] = a.external_ki_rate;
	aim["external_ki_deadband"] = a.external_ki_deadband;

	// aim controller
	aim["aim_kp"] = a.aim_kp;
	aim["aim_ki"] = a.aim_ki;
	aim["aim_kd"] = a.aim_kd;
	aim["aim_noise_enabled"] = a.aim_noise_enabled;
	aim["aim_noise_amplitude"] = a.aim_noise_amplitude;
	aim["aim_prediction_weight_x"] = a.aim_prediction_weight_x;
	aim["aim_prediction_weight_y"] = a.aim_prediction_weight_y;
	aim["aim_ramp_time"] = a.aim_ramp_time;
	aim["aim_init_scale"] = a.aim_init_scale;
	aim["aim_output_max"] = a.aim_output_max;

	aim["enable_neural_path"] = a.enable_neural_path;
	aim["neural_path_points"] = a.neural_path_points;
	aim["neural_mouse_step_size"] = a.neural_mouse_step_size;
	aim["neural_target_radius"] = a.neural_target_radius;
	aim["neural_consume_per_frame"] = a.neural_consume_per_frame;
	aim["enable_neural_path_debug"] = a.enable_neural_path_debug;

	// flat slot0 for quick Web (compat)
	const auto &p0 = a.profiles[0];
	aim["hotkey_vk"] = p0.mc.hotkeyVirtualKey;
	aim["controller"] = FullAimBridge::controller_name(p0.mc.controllerType);
	aim["continuous_aim"] = p0.continuous_aim;
	aim["pidPMin"] = p0.mc.pidPMin;
	aim["pidPMax"] = p0.mc.pidPMax;
	aim["pidD"] = p0.mc.pidD;
	aim["pidI"] = p0.mc.pidI;
	aim["deadZonePixels"] = p0.mc.deadZonePixels;
	aim["maxPixelMove"] = p0.mc.maxPixelMove;
	aim["targetYOffset"] = p0.mc.targetYOffset;
	aim["autoTriggerEnabled"] = p0.mc.autoTriggerEnabled;
	aim["autoTriggerRadius"] = p0.mc.autoTriggerRadius;
	aim["autoRecoilControlEnabled"] = p0.mc.autoRecoilControlEnabled;
	aim["recoilStrength"] = p0.mc.recoilStrength;
	aim["makcuPort"] = p0.mc.makcuPort;
	aim["makcuBaudRate"] = p0.mc.makcuBaudRate;
	aim["logiDriverType"] = p0.mc.logiDriverType;

	json configs = json::array();
	json obs_flat; // enable_config_0 style for import/export parity
	for (int i = 0; i < FullAimSettings::kSlots; ++i) {
		configs.push_back(profile_to_json(a.profiles[i], i));
		const auto &p = a.profiles[i];
		const auto &m = p.mc;
		obs_flat[slot_key("enable_config", i)] = p.enabled;
		obs_flat[slot_key("continuous_aim", i)] = p.continuous_aim;
		obs_flat[slot_key("hotkey", i)] = m.hotkeyVirtualKey;
		obs_flat[slot_key("controller_type", i)] = static_cast<int>(m.controllerType);
		obs_flat[slot_key("logi_driver_type", i)] = m.logiDriverType;
		obs_flat[slot_key("makcu_port", i)] = m.makcuPort;
		obs_flat[slot_key("makcu_baud_rate", i)] = m.makcuBaudRate;
		obs_flat[slot_key("p_min", i)] = m.pidPMin;
		obs_flat[slot_key("p_max", i)] = m.pidPMax;
		obs_flat[slot_key("p_slope", i)] = m.pidPSlope;
		obs_flat[slot_key("d", i)] = m.pidD;
		obs_flat[slot_key("i", i)] = m.pidI;
		obs_flat[slot_key("derivative_filter_alpha", i)] = m.derivativeFilterAlpha;
		obs_flat[slot_key("adaptive_p_gain_rate", i)] = m.adaptivePGainRate;
		obs_flat[slot_key("d_term_scale", i)] = m.dTermScale;
		obs_flat[slot_key("target_y_offset", i)] = m.targetYOffset;
		obs_flat[slot_key("max_pixel_move", i)] = m.maxPixelMove;
		obs_flat[slot_key("dead_zone_pixels", i)] = m.deadZonePixels;
		obs_flat[slot_key("screen_offset_x", i)] = m.screenOffsetX;
		obs_flat[slot_key("screen_offset_y", i)] = m.screenOffsetY;
		obs_flat[slot_key("screen_width", i)] = m.screenWidth;
		obs_flat[slot_key("screen_height", i)] = m.screenHeight;
		obs_flat[slot_key("enable_y_axis_unlock", i)] = m.yUnlockEnabled;
		obs_flat[slot_key("y_axis_unlock_delay", i)] = m.yUnlockDelayMs;
		obs_flat[slot_key("auto_trigger_group", i)] = m.autoTriggerEnabled;
		obs_flat[slot_key("trigger_radius", i)] = m.autoTriggerRadius;
		obs_flat[slot_key("trigger_cooldown", i)] = m.autoTriggerCooldownMs;
		obs_flat[slot_key("trigger_fire_delay", i)] = m.autoTriggerFireDelay;
		obs_flat[slot_key("trigger_fire_duration", i)] = m.autoTriggerFireDuration;
		obs_flat[slot_key("trigger_interval", i)] = m.autoTriggerInterval;
		obs_flat[slot_key("enable_trigger_delay_random", i)] = m.autoTriggerDelayRandomEnabled;
		obs_flat[slot_key("trigger_delay_random_min", i)] = m.autoTriggerDelayRandomMin;
		obs_flat[slot_key("trigger_delay_random_max", i)] = m.autoTriggerDelayRandomMax;
		obs_flat[slot_key("enable_trigger_duration_random", i)] =
		    m.autoTriggerDurationRandomEnabled;
		obs_flat[slot_key("trigger_duration_random_min", i)] = m.autoTriggerDurationRandomMin;
		obs_flat[slot_key("trigger_duration_random_max", i)] = m.autoTriggerDurationRandomMax;
		obs_flat[slot_key("trigger_move_compensation", i)] = m.autoTriggerMoveCompensation;
		obs_flat[slot_key("integral_limit", i)] = m.integralLimit;
		obs_flat[slot_key("integral_rate", i)] = m.integralRate;
		obs_flat[slot_key("p_gain_ramp_initial_scale", i)] = m.pGainRampInitialScale;
		obs_flat[slot_key("p_gain_ramp_duration", i)] = m.pGainRampDuration;
		obs_flat[slot_key("prediction_weight_x", i)] = m.predictionWeightX;
		obs_flat[slot_key("prediction_weight_y", i)] = m.predictionWeightY;
		obs_flat[slot_key("recoil_group", i)] = m.autoRecoilControlEnabled;
		obs_flat[slot_key("recoil_strength", i)] = m.recoilStrength;
		obs_flat[slot_key("recoil_speed", i)] = m.recoilSpeed;
		obs_flat[slot_key("recoil_pid_gain_scale", i)] = m.recoilPidGainScale;
		obs_flat[slot_key("derivative_predictor_group", i)] = m.useDerivativePredictor;
		obs_flat[slot_key("max_prediction_time", i)] = m.maxPredictionTime;
		obs_flat[slot_key("smith_predictor_group", i)] = m.smithPredictorEnabled;
		obs_flat[slot_key("smith_model_gain", i)] = m.smithModelGain;
		obs_flat[slot_key("smith_model_tau", i)] = m.smithModelTau;
		obs_flat[slot_key("smith_auto_tau", i)] = m.smithAutoTau;
		obs_flat[slot_key("slew_rate_group", i)] = m.slewRateEnabled;
		obs_flat[slot_key("slew_rate_output_gain", i)] = m.slewRateOutputGain;
		obs_flat[slot_key("slew_rate_response_smoothing", i)] = m.slewRateResponseSmoothing;
		obs_flat[slot_key("slew_rate_approach_damping", i)] = m.slewRateApproachDamping;
		obs_flat[slot_key("slew_rate_update_interval_ms", i)] = m.slewRateUpdateIntervalMs;
		obs_flat[slot_key("slew_rate_normalization_scale", i)] = m.slewRateNormalizationScale;
		obs_flat[slot_key("adaptive_pid_kp", i)] = m.adaptivePidKp;
		obs_flat[slot_key("adaptive_pid_ki", i)] = m.adaptivePidKi;
		obs_flat[slot_key("adaptive_pid_kd", i)] = m.adaptivePidKd;
		obs_flat[slot_key("adaptive_pid_dead_zone", i)] = m.adaptivePidDeadZone;
		obs_flat[slot_key("adaptive_pid_integral_limit", i)] = m.adaptivePidIntegralLimit;
		obs_flat[slot_key("adaptive_pid_integral_deadzone", i)] = m.adaptivePidIntegralDeadzone;
		obs_flat[slot_key("adaptive_pid_integral_gain_threshold", i)] =
		    m.adaptivePidIntegralGainThreshold;
		obs_flat[slot_key("adaptive_pid_integral_gain_rate", i)] = m.adaptivePidIntegralGainRate;
		obs_flat[slot_key("adaptive_pid_output_limit", i)] = m.adaptivePidOutputLimit;
		obs_flat[slot_key("bezier_movement_group", i)] = m.enableBezierMovement;
		obs_flat[slot_key("bezier_curvature", i)] = m.bezierCurvature;
		obs_flat[slot_key("bezier_randomness", i)] = m.bezierRandomness;
		obs_flat[slot_key("ghost_tracker_group", i)] = m.enableGhostTracker;
		obs_flat[slot_key("ghost_curvature", i)] = m.ghostCurvature;
		obs_flat[slot_key("ghost_noise_intensity", i)] = m.ghostNoiseIntensity;
		obs_flat[slot_key("ghost_vertical_snap", i)] = m.ghostVerticalSnapRatio;
		obs_flat[slot_key("ghost_noise_freq", i)] = m.ghostNoiseFreq;
	}
	aim["configs"] = configs;
	aim["obs_keys"] = obs_flat;
	root["aim"] = aim;

// vision (FOV subset also under aim for OBS-like pages)
			root["vision"] = {
			    {"show_detection_results", c.vision.show_detection_results},
			    {"bbox_line_width", c.vision.bbox_line_width},
			    {"show_floating_window", c.vision.show_floating_window},
			    {"floating_window_width", c.vision.floating_window_width},
			    {"floating_window_height", c.vision.floating_window_height},
			    {"show_track_id_in_floating_window", c.vision.show_track_id_in_floating_window},
			    {"label_font_scale", c.vision.label_font_scale},
			    {"export_coordinates", c.vision.export_coordinates},
			    {"coordinate_output_path", c.vision.coordinate_output_path},
			    {"preview_enabled", c.vision.preview_enabled},
			    {"show_fov", a.show_fov},
			    {"fov_radius", a.fov_radius},
			    {"show_fov_circle", a.show_fov_circle},
			    {"show_fov_cross", a.show_fov_cross},
			    {"use_dynamic_fov", a.use_dynamic_fov},
			    {"fov_radius2", a.fov_radius2},
			    {"dynamic_fov_shrink_percent", static_cast<int>(a.dynamic_fov_shrink_percent * 100.f)},
			    {"dynamic_fov_transition_time", static_cast<int>(a.dynamic_fov_transition_ms)},
			};
			// Flat OBS vision keys at root (scene JSON import)
			root["show_detection_results"] = c.vision.show_detection_results;
			root["bbox_line_width"] = c.vision.bbox_line_width;
			root["label_font_scale"] = c.vision.label_font_scale;
			root["export_coordinates"] = c.vision.export_coordinates;
			root["coordinate_output_path"] = c.vision.coordinate_output_path;
			root["show_floating_window"] = c.vision.show_floating_window;
			root["floating_window_width"] = c.vision.floating_window_width;
			root["floating_window_height"] = c.vision.floating_window_height;
			root["show_track_id_in_floating_window"] = c.vision.show_track_id_in_floating_window;
			root["preview_enabled"] = c.vision.preview_enabled;
			root["show_fov"] = a.show_fov;
			root["fov_radius"] = a.fov_radius;
			root["show_fov_circle"] = a.show_fov_circle;
			root["show_fov_cross"] = a.show_fov_cross;
			root["use_dynamic_fov"] = a.use_dynamic_fov;
			root["fov_radius2"] = a.fov_radius2;
			root["dynamic_fov_shrink_percent"] = static_cast<int>(a.dynamic_fov_shrink_percent * 100.f);
			root["dynamic_fov_transition_time"] = static_cast<int>(a.dynamic_fov_transition_ms);

	// prediction (slot0 + globals for page 6 convenience)
	const auto &m0 = p0.mc;
	root["prediction"] = {
	    {"derivative_predictor_group", m0.useDerivativePredictor},
	    {"prediction_weight_x", m0.predictionWeightX},
	    {"prediction_weight_y", m0.predictionWeightY},
	    {"max_prediction_time", m0.maxPredictionTime},
	    {"smith_predictor_group", m0.smithPredictorEnabled},
	    {"smith_model_gain", m0.smithModelGain},
	    {"smith_model_tau", m0.smithModelTau},
	    {"smith_auto_tau", m0.smithAutoTau},
	    {"use_one_euro_filter", m0.useOneEuroFilter},
	    {"one_euro_min_cutoff", m0.oneEuroMinCutoff},
	    {"one_euro_beta", m0.oneEuroBeta},
	    {"one_euro_d_cutoff", m0.oneEuroDCutoff},
	    {"bezier_movement_group", m0.enableBezierMovement},
	    {"bezier_curvature", m0.bezierCurvature},
	    {"bezier_randomness", m0.bezierRandomness},
	    {"ghost_tracker_group", m0.enableGhostTracker},
	    {"ghost_curvature", m0.ghostCurvature},
	    {"ghost_noise_intensity", m0.ghostNoiseIntensity},
	    {"ghost_vertical_snap", m0.ghostVerticalSnapRatio},
	    {"ghost_noise_freq", m0.ghostNoiseFreq},
	    {"slew_rate_group", m0.slewRateEnabled},
	    {"slew_rate_output_gain", m0.slewRateOutputGain},
	    {"slew_rate_response_smoothing", m0.slewRateResponseSmoothing},
	    {"slew_rate_approach_damping", m0.slewRateApproachDamping},
	    {"slew_rate_update_interval_ms", m0.slewRateUpdateIntervalMs},
	    {"slew_rate_normalization_scale", m0.slewRateNormalizationScale},
	    {"enable_neural_path", a.enable_neural_path},
	    {"neural_path_points", a.neural_path_points},
	    {"neural_mouse_step_size", a.neural_mouse_step_size},
	    {"neural_target_radius", a.neural_target_radius},
	};

	root["crosshair"] = {
	    {"crosshair_enabled", a.crosshair_enabled},
	    {"enabled", a.crosshair_enabled},
	    {"crosshair_h_min", a.crosshair_h_min},
	    {"crosshair_h_max", a.crosshair_h_max},
	    {"crosshair_s_min", a.crosshair_s_min},
	    {"crosshair_s_max", a.crosshair_s_max},
	    {"crosshair_v_min", a.crosshair_v_min},
	    {"crosshair_v_max", a.crosshair_v_max},
	    {"crosshair_manual_r", a.crosshair_manual_r},
	    {"crosshair_manual_g", a.crosshair_manual_g},
	    {"crosshair_manual_b", a.crosshair_manual_b},
	    {"crosshair_h_tolerance", a.crosshair_h_tolerance},
	    {"crosshair_s_tolerance", a.crosshair_s_tolerance},
	    {"crosshair_v_tolerance", a.crosshair_v_tolerance},
	    {"crosshair_morph_kernel", a.crosshair_morph_kernel},
	    {"crosshair_erode_iter", a.crosshair_erode_iter},
	    {"crosshair_dilate_iter", a.crosshair_dilate_iter},
	    {"crosshair_grid_rows", a.crosshair_grid_rows},
	    {"crosshair_grid_cols", a.crosshair_grid_cols},
	    {"crosshair_quantile_threshold", a.crosshair_quantile_threshold},
	    {"crosshair_template_path", a.crosshair_template_path},
	    {"crosshair_match_threshold", a.crosshair_match_threshold},
	    {"crosshair_min_area", a.crosshair_min_area},
	    {"crosshair_max_area", a.crosshair_max_area},
	    {"crosshair_shape_filter_enabled", a.crosshair_shape_filter_enabled},
	    {"crosshair_shape_type", a.crosshair_shape_type},
	    {"crosshair_min_fill_ratio", a.crosshair_min_fill_ratio},
	    {"crosshair_max_fill_ratio", a.crosshair_max_fill_ratio},
	    {"crosshair_min_aspect_ratio", a.crosshair_min_aspect_ratio},
	    {"crosshair_max_aspect_ratio", a.crosshair_max_aspect_ratio},
	    {"crosshair_detect_interval", a.crosshair_detect_interval},
	    {"crosshair_search_radius", a.crosshair_search_radius},
	    {"crosshair_color_isolation", a.crosshair_color_isolation},
	    {"crosshair_debug_mask", a.crosshair_debug_mask},
	};
#else
	root["aim"] = {{"enabled", c.aim_enabled}};
#endif

	root["engine"] = {{"preview_interval_sec", c.preview_interval_sec}};
	return root;
}

void ConfigStore::apply_obs_slot_keys(EngineConfig &c, const json &j, int slot)
{
#ifdef YA_WITH_AIM
	if (slot < 0 || slot >= FullAimSettings::kSlots)
		return;
	auto &p = c.aim_full.profiles[slot];
	// Build a pseudo object from suffixed keys
	json o;
	auto pull = [&](const char *base) {
		const std::string k = slot_key(base, slot);
		if (j.contains(k))
			o[base] = j[k];
	};
	const char *keys[] = {
	    "enable_config",
	    "continuous_aim",
	    "hotkey",
	    "controller_type",
	    "logi_driver_type",
	    "makcu_port",
	    "makcu_baud_rate",
	    "p_min",
	    "p_max",
	    "p_slope",
	    "d",
	    "i",
	    "derivative_filter_alpha",
	    "adaptive_p_gain_rate",
	    "d_term_scale",
	    "target_y_offset",
	    "max_pixel_move",
	    "dead_zone_pixels",
	    "screen_offset_x",
	    "screen_offset_y",
	    "screen_width",
	    "screen_height",
	    "enable_y_axis_unlock",
	    "y_axis_unlock_delay",
	    "auto_trigger_group",
	    "trigger_radius",
	    "trigger_cooldown",
	    "trigger_fire_delay",
	    "trigger_fire_duration",
	    "trigger_interval",
	    "enable_trigger_delay_random",
	    "trigger_delay_random_min",
	    "trigger_delay_random_max",
	    "enable_trigger_duration_random",
	    "trigger_duration_random_min",
	    "trigger_duration_random_max",
	    "trigger_move_compensation",
	    "integral_limit",
	    "integral_rate",
	    "p_gain_ramp_initial_scale",
	    "p_gain_ramp_duration",
	    "prediction_weight_x",
	    "prediction_weight_y",
	    "recoil_group",
	    "recoil_strength",
	    "recoil_speed",
	    "recoil_pid_gain_scale",
	    "derivative_predictor_group",
	    "max_prediction_time",
	    "smith_predictor_group",
	    "smith_model_gain",
	    "smith_model_tau",
	    "smith_auto_tau",
	    "slew_rate_group",
	    "slew_rate_output_gain",
	    "slew_rate_response_smoothing",
	    "slew_rate_approach_damping",
	    "slew_rate_update_interval_ms",
	    "slew_rate_normalization_scale",
	    "adaptive_pid_kp",
	    "adaptive_pid_ki",
	    "adaptive_pid_kd",
	    "adaptive_pid_dead_zone",
	    "adaptive_pid_integral_limit",
	    "adaptive_pid_integral_deadzone",
	    "adaptive_pid_integral_gain_threshold",
	    "adaptive_pid_integral_gain_rate",
	    "adaptive_pid_output_limit",
	    "bezier_movement_group",
	    "bezier_curvature",
	    "bezier_randomness",
	    "ghost_tracker_group",
	    "ghost_curvature",
	    "ghost_noise_intensity",
	    "ghost_vertical_snap",
	    "ghost_noise_freq",
	};
	for (const char *k : keys)
		pull(k);
	if (!o.empty())
		apply_profile_object(p, o);
#else
	(void)c;
	(void)j;
	(void)slot;
#endif
}

void ConfigStore::apply_obs_global_keys(EngineConfig &c, const json &j)
{
#ifdef YA_WITH_AIM
	auto &a = c.aim_full;
	{
		int algo = static_cast<int>(a.algorithm);
		jget_int2(j, "algorithm_type_global", "algorithm", algo);
		a.algorithm = FullAimBridge::parse_algorithm(algo);
	}
	jget_int2(j, "mouse_config_select", "config_select", a.config_select);
	jget_int2(j, "fov_radius", nullptr, a.fov_radius);
	jget_bool2(j, "show_fov", nullptr, a.show_fov);
	jget_bool2(j, "show_fov_circle", nullptr, a.show_fov_circle);
	jget_bool2(j, "show_fov_cross", nullptr, a.show_fov_cross);
	jget_bool2(j, "use_dynamic_fov", nullptr, a.use_dynamic_fov);
	jget_int2(j, "fov_radius2", nullptr, a.fov_radius2);
	if (j.contains("dynamic_fov_shrink_percent")) {
		int pct = static_cast<int>(a.dynamic_fov_shrink_percent * 100.f);
		jget_int(j, "dynamic_fov_shrink_percent", pct);
		if (pct > 1)
			a.dynamic_fov_shrink_percent = pct / 100.f;
		else
			a.dynamic_fov_shrink_percent = static_cast<float>(pct);
	}
	if (j.contains("dynamic_fov_transition_time")) {
		int ms = static_cast<int>(a.dynamic_fov_transition_ms);
		jget_int(j, "dynamic_fov_transition_time", ms);
		a.dynamic_fov_transition_ms = static_cast<float>(ms);
	}
	jget_int2(j, "target_switch_delay", nullptr, a.target_switch_delay_ms);
	jget_float2(j, "target_switch_tolerance", nullptr, a.target_switch_tolerance);

	jget_float2(j, "external_kp_x", nullptr, a.external_kp_x);
	jget_float2(j, "external_ki_x", nullptr, a.external_ki_x);
	jget_float2(j, "external_kd_x", nullptr, a.external_kd_x);
	jget_float2(j, "external_kp_y", nullptr, a.external_kp_y);
	jget_float2(j, "external_ki_y", nullptr, a.external_ki_y);
	jget_float2(j, "external_kd_y", nullptr, a.external_kd_y);
	jget_float2(j, "external_predict_x", nullptr, a.external_predict_x);
	jget_float2(j, "external_predict_y", nullptr, a.external_predict_y);
	jget_float2(j, "external_rate_x", nullptr, a.external_rate_x);
	jget_float2(j, "external_rate_y", nullptr, a.external_rate_y);
	{
		int mode = static_cast<int>(a.external_ki_mode);
		jget_int2(j, "external_ki_mode", nullptr, mode);
		a.external_ki_mode = static_cast<float>(mode);
	}
	jget_float2(j, "external_kp_limit", nullptr, a.external_kp_limit);
	jget_float2(j, "external_ki_limit", nullptr, a.external_ki_limit);
	jget_float2(j, "external_kd_limit", nullptr, a.external_kd_limit);
	jget_float2(j, "external_output_limit", nullptr, a.external_output_limit);
	jget_float2(j, "external_ki_rate", nullptr, a.external_ki_rate);
	jget_float2(j, "external_ki_deadband", nullptr, a.external_ki_deadband);

	jget_float2(j, "aim_kp", nullptr, a.aim_kp);
	jget_float2(j, "aim_ki", nullptr, a.aim_ki);
	jget_float2(j, "aim_kd", nullptr, a.aim_kd);
	jget_bool2(j, "aim_noise_enabled", nullptr, a.aim_noise_enabled);
	jget_float2(j, "aim_noise_amplitude", nullptr, a.aim_noise_amplitude);
	jget_float2(j, "aim_prediction_weight_x", nullptr, a.aim_prediction_weight_x);
	jget_float2(j, "aim_prediction_weight_y", nullptr, a.aim_prediction_weight_y);
	jget_float2(j, "aim_ramp_time", nullptr, a.aim_ramp_time);
	jget_float2(j, "aim_init_scale", nullptr, a.aim_init_scale);
	jget_float2(j, "aim_output_max", nullptr, a.aim_output_max);

	jget_bool2(j, "enable_neural_path", nullptr, a.enable_neural_path);
	jget_int2(j, "neural_path_points", nullptr, a.neural_path_points);
	if (j.contains("neural_mouse_step_size") && j["neural_mouse_step_size"].is_number())
		a.neural_mouse_step_size = j["neural_mouse_step_size"].get<double>();
	jget_int2(j, "neural_target_radius", nullptr, a.neural_target_radius);
	jget_int2(j, "neural_consume_per_frame", nullptr, a.neural_consume_per_frame);
	jget_bool2(j, "enable_neural_path_debug", nullptr, a.enable_neural_path_debug);

	jget_bool2(j, "crosshair_enabled", "crosshair_enabled", a.crosshair_enabled);
	jget_int2(j, "crosshair_h_min", nullptr, a.crosshair_h_min);
	jget_int2(j, "crosshair_h_max", nullptr, a.crosshair_h_max);
	jget_int2(j, "crosshair_s_min", nullptr, a.crosshair_s_min);
	jget_int2(j, "crosshair_s_max", nullptr, a.crosshair_s_max);
	jget_int2(j, "crosshair_v_min", nullptr, a.crosshair_v_min);
	jget_int2(j, "crosshair_v_max", nullptr, a.crosshair_v_max);
	jget_int2(j, "crosshair_manual_r", nullptr, a.crosshair_manual_r);
	jget_int2(j, "crosshair_manual_g", nullptr, a.crosshair_manual_g);
	jget_int2(j, "crosshair_manual_b", nullptr, a.crosshair_manual_b);
	jget_int2(j, "crosshair_h_tolerance", nullptr, a.crosshair_h_tolerance);
	jget_int2(j, "crosshair_s_tolerance", nullptr, a.crosshair_s_tolerance);
	jget_int2(j, "crosshair_v_tolerance", nullptr, a.crosshair_v_tolerance);
	jget_int2(j, "crosshair_morph_kernel", nullptr, a.crosshair_morph_kernel);
	jget_int2(j, "crosshair_erode_iter", nullptr, a.crosshair_erode_iter);
	jget_int2(j, "crosshair_dilate_iter", nullptr, a.crosshair_dilate_iter);
	jget_int2(j, "crosshair_grid_rows", nullptr, a.crosshair_grid_rows);
	jget_int2(j, "crosshair_grid_cols", nullptr, a.crosshair_grid_cols);
	jget_float2(j, "crosshair_quantile_threshold", nullptr, a.crosshair_quantile_threshold);
	jget_str2(j, "crosshair_template_path", nullptr, a.crosshair_template_path);
	jget_float2(j, "crosshair_match_threshold", nullptr, a.crosshair_match_threshold);
	jget_int2(j, "crosshair_min_area", nullptr, a.crosshair_min_area);
	jget_int2(j, "crosshair_max_area", nullptr, a.crosshair_max_area);
	jget_bool2(j, "crosshair_shape_filter_enabled", nullptr, a.crosshair_shape_filter_enabled);
	jget_int2(j, "crosshair_shape_type", nullptr, a.crosshair_shape_type);
	jget_float2(j, "crosshair_min_fill_ratio", nullptr, a.crosshair_min_fill_ratio);
	jget_float2(j, "crosshair_max_fill_ratio", nullptr, a.crosshair_max_fill_ratio);
	jget_float2(j, "crosshair_min_aspect_ratio", nullptr, a.crosshair_min_aspect_ratio);
	jget_float2(j, "crosshair_max_aspect_ratio", nullptr, a.crosshair_max_aspect_ratio);
	jget_int2(j, "crosshair_detect_interval", nullptr, a.crosshair_detect_interval);
	jget_int2(j, "crosshair_search_radius", nullptr, a.crosshair_search_radius);
	jget_bool2(j, "crosshair_color_isolation", nullptr, a.crosshair_color_isolation);
	jget_bool2(j, "crosshair_debug_mask", nullptr, a.crosshair_debug_mask);
#else
	(void)c;
	(void)j;
#endif
}

bool ConfigStore::merge_into(EngineConfig &c, const json &patch, std::string *err)
	{
		try {
			const bool has_capture =
			    patch.contains("capture") && patch["capture"].is_object();
			const bool has_infer = patch.contains("infer") && patch["infer"].is_object();

			// --- capture ---
			// Only read capture from nested object, or from flat root when this patch
			// is not an infer-only hot update (avoids clobbering width with unrelated keys).
			if (has_capture || !has_infer) {
				const json &cap = has_capture ? patch["capture"] : patch;
				std::string backend;
				jget_str(cap, "backend", backend);
				if (!backend.empty()) {
					if (backend == "gdi")
						c.capture_backend = CaptureBackend::Gdi;
					else if (backend == "wgc")
						c.capture_backend = CaptureBackend::Wgc;
					else
						c.capture_backend = CaptureBackend::Dxgi;
				}
jget_str(cap, "mode", c.capture_mode);
					jget_int(cap, "width", c.width);
					jget_int(cap, "height", c.height);
					// OBS keys (flat under capture.*) — WebUI writes these via data-path
					jget_bool(cap, "use_region", c.use_region);
					jget_int(cap, "region_x", c.region_x);
					jget_int(cap, "region_y", c.region_y);
					jget_int(cap, "region_width", c.region_width);
					jget_int(cap, "region_height", c.region_height);
					// nested region {x,y,w,h} also accepted
					if (cap.contains("region") && cap["region"].is_object()) {
						jget_int(cap["region"], "x", c.region_x);
						jget_int(cap["region"], "y", c.region_y);
						jget_int(cap["region"], "w", c.region_width);
						jget_int(cap["region"], "h", c.region_height);
					}
					// Keep capture size in sync with region size for ROI modes
					// (center/region capture always grabs region_width x region_height).
					if (c.capture_mode == "center" || c.capture_mode == "region") {
						if (c.region_width > 0)
							c.width = c.region_width;
						if (c.region_height > 0)
							c.height = c.region_height;
						// width/height fields also drive region size when user edits them
						if (cap.contains("width") && !cap.contains("region_width"))
							c.region_width = c.width;
						if (cap.contains("height") && !cap.contains("region_height"))
							c.region_height = c.height;
					}
					// When use_region is off and mode is full → treat as full desktop
					if (cap.contains("use_region") && !c.use_region &&
					    c.capture_mode != "full") {
						// keep mode; engine will open full desktop only if mode=full
					}
					if (c.capture_mode == "full" || c.capture_mode == "fullscreen") {
						c.capture_mode = "full";
						c.use_region = c.use_region; // may still crop software-side
					}
				}
				// flat root OBS aliases
				jget_bool(patch, "use_region", c.use_region);
				jget_int(patch, "region_x", c.region_x);
				jget_int(patch, "region_y", c.region_y);
				jget_int(patch, "region_width", c.region_width);
				jget_int(patch, "region_height", c.region_height);

// --- infer ---
				if (has_infer) {
					const json &inf = patch["infer"];
					jget_bool(inf, "enabled", c.infer_enabled);
					jget_bool(inf, "is_inferencing", c.infer_enabled); // OBS
					// model_path: only if non-empty string (never wipe with "")
					if (inf.contains("model_path") && inf["model_path"].is_string()) {
						const std::string mp = inf["model_path"].get<std::string>();
						if (!mp.empty())
							c.infer.model_path = mp;
					}
					// OBS use_gpu aliases device
					jget_str2(inf, "device", "use_gpu", c.infer.device);
					jget_int(inf, "model_version", c.infer.model_version);
					jget_float2(inf, "confidence", "confidence_threshold", c.infer.confidence);
					jget_float2(inf, "nms", "nms_threshold", c.infer.nms);
					jget_int2(inf, "input_size", "input_resolution", c.infer.input_resolution);
					jget_int2(inf, "interval_frames", "inference_interval_frames",
					          c.infer.interval_frames);
					jget_int(inf, "num_threads", c.infer.num_threads);
					if (inf.contains("target_classes") && inf["target_classes"].is_array()) {
						c.infer.target_classes.clear();
						for (const auto &it : inf["target_classes"]) {
							if (it.is_number_integer())
								c.infer.target_classes.push_back(it.get<int>());
							else if (it.is_number())
								c.infer.target_classes.push_back(
								    static_cast<int>(it.get<double>()));
						}
					}
					if (inf.contains("target_classes_text") &&
					    inf["target_classes_text"].is_string()) {
						c.infer.target_classes_text =
						    inf["target_classes_text"].get<std::string>();
						parse_target_classes_text(c.infer.target_classes_text,
						                          c.infer.target_classes);
					}
					// OBS single-class list (target_class: -1 = all)
					if (inf.contains("target_class") && inf["target_class"].is_number()) {
						const int tc = static_cast<int>(inf["target_class"].get<double>());
						if (tc < 0) {
							c.infer.target_classes.clear();
							c.infer.target_classes_text.clear();
						} else if (c.infer.target_classes_text.empty()) {
							c.infer.target_classes = {tc};
							c.infer.target_classes_text = std::to_string(tc);
						}
					}
				}
				// flat root OBS aliases (explicit only; never treat empty model_path as clear)
				jget_float2(patch, "confidence", "confidence_threshold", c.infer.confidence);
				jget_float2(patch, "nms", "nms_threshold", c.infer.nms);
				jget_int2(patch, "interval_frames", "inference_interval_frames",
				          c.infer.interval_frames);
				jget_int2(patch, "input_size", "input_resolution", c.infer.input_resolution);
				jget_str2(patch, "device", "use_gpu", c.infer.device);
				jget_bool(patch, "infer_enabled", c.infer_enabled);
				jget_bool(patch, "is_inferencing", c.infer_enabled);
				jget_int(patch, "model_version", c.infer.model_version);
				jget_int(patch, "num_threads", c.infer.num_threads);
				if (patch.contains("model_path") && patch["model_path"].is_string()) {
					const std::string mp = patch["model_path"].get<std::string>();
					if (!mp.empty())
						c.infer.model_path = mp;
				}
				if (patch.contains("target_classes_text") &&
				    patch["target_classes_text"].is_string()) {
					c.infer.target_classes_text =
					    patch["target_classes_text"].get<std::string>();
					parse_target_classes_text(c.infer.target_classes_text,
					                          c.infer.target_classes);
				}
				if (patch.contains("target_class") && patch["target_class"].is_number()) {
					const int tc = static_cast<int>(patch["target_class"].get<double>());
					if (tc < 0) {
						c.infer.target_classes.clear();
						c.infer.target_classes_text.clear();
					} else if (c.infer.target_classes_text.empty()) {
						c.infer.target_classes = {tc};
						c.infer.target_classes_text = std::to_string(tc);
					}
				}
			// models folder roots (optional, for WebUI folder picker)
			if (patch.contains("models_dir") && patch["models_dir"].is_string()) {
				const std::string d = patch["models_dir"].get<std::string>();
				if (!d.empty())
					c.models_dir = d;
			}
			if (patch.contains("model_search_dirs") && patch["model_search_dirs"].is_array()) {
				c.model_search_dirs.clear();
				for (const auto &it : patch["model_search_dirs"]) {
					if (it.is_string()) {
						const std::string d = it.get<std::string>();
						if (!d.empty())
							c.model_search_dirs.push_back(d);
					}
				}
			}
			if (patch.contains("model_folder") && patch["model_folder"].is_string()) {
				const std::string d = patch["model_folder"].get<std::string>();
				if (!d.empty()) {
					c.models_dir = d;
					// prepend so list_models finds it first
					auto &dirs = c.model_search_dirs;
					dirs.erase(std::remove(dirs.begin(), dirs.end(), d), dirs.end());
					dirs.insert(dirs.begin(), d);
				}
			}

		// --- tracking ---
		const json *tr = patch.contains("tracking") && patch["tracking"].is_object()
		                     ? &patch["tracking"]
		                     : &patch;
		{
			jget_float(*tr, "iou_threshold", c.tracker.iou_threshold);
			jget_int(*tr, "max_lost_frames", c.tracker.max_lost_frames);
			jget_int(*tr, "max_reidentify_frames", c.tracker.max_reidentify_frames);
			jget_float(*tr, "reidentify_center_threshold", c.tracker.reidentify_center_threshold);
			jget_float2(*tr, "tracking_weight_iou", "weight_iou", c.tracker.weight_iou);
			jget_float2(*tr, "tracking_weight_center", "weight_center", c.tracker.weight_center);
			jget_float2(*tr, "tracking_weight_aspect", "weight_aspect", c.tracker.weight_aspect);
jget_float2(*tr, "tracking_weight_area", "weight_area", c.tracker.weight_area);
				jget_bool2(*tr, "use_kalman_tracker", "use_kalman", c.tracker.use_kalman);
				jget_int(*tr, "kalman_generate_threshold", c.tracker.kalman_generate_threshold);
				jget_int(*tr, "kalman_terminate_count", c.tracker.kalman_terminate_count);
				jget_bool(*tr, "show_kalman_predictions", c.tracker.show_kalman_predictions);
				jget_int(*tr, "kalman_prediction_frames", c.tracker.kalman_prediction_frames);
				jget_bool(*tr, "show_kalman_trajectories", c.tracker.show_kalman_trajectories);
				// OBS: terminate count can drive max_lost when set
				if (tr->contains("kalman_terminate_count") && c.tracker.kalman_terminate_count > 0)
					c.tracker.max_lost_frames = c.tracker.kalman_terminate_count;
			}

#ifdef YA_WITH_AIM
		// --- aim ---
		jget_bool(patch, "aim_enabled", c.aim_enabled);
		if (patch.contains("aim") && patch["aim"].is_object()) {
			const json &aim = patch["aim"];
			jget_bool(aim, "enabled", c.aim_enabled);
			c.aim_full.enabled = c.aim_enabled;
			apply_obs_global_keys(c, aim);

			// nested configs array
			if (aim.contains("configs") && aim["configs"].is_array()) {
				int i = 0;
				for (const auto &item : aim["configs"]) {
					if (i >= FullAimSettings::kSlots)
						break;
					if (item.is_object())
						apply_profile_object(c.aim_full.profiles[i], item);
					++i;
				}
			}
			// obs_keys nested
			if (aim.contains("obs_keys") && aim["obs_keys"].is_object()) {
				for (int s = 0; s < FullAimSettings::kSlots; ++s)
					apply_obs_slot_keys(c, aim["obs_keys"], s);
			}
			// flat slot0 aliases on aim object
			int slot = 0;
			jget_int(aim, "slot", slot);
			if (slot < 0 || slot >= FullAimSettings::kSlots)
				slot = 0;
			// When thin WebUI sends flat fields without configs[], apply to selected slot
			bool has_flat = aim.contains("pidPMin") || aim.contains("p_min") ||
			                aim.contains("hotkey_vk") || aim.contains("controller") ||
			                aim.contains("autoTriggerEnabled") || aim.contains("continuous_aim");
			if (has_flat && !aim.contains("configs"))
				apply_profile_object(c.aim_full.profiles[slot], aim);
		}

		// root-level flat thin-WebUI keys
		{
			int slot = 0;
			jget_int(patch, "slot", slot);
			if (slot < 0 || slot >= FullAimSettings::kSlots)
				slot = 0;
			bool has_flat = patch.contains("pidPMin") || patch.contains("p_min") ||
			                patch.contains("hotkey_vk") || patch.contains("controller") ||
			                patch.contains("autoTriggerEnabled") || patch.contains("fov_radius") ||
			                patch.contains("algorithm") || patch.contains("continuous_aim") ||
			                patch.contains("aim_enabled");
			if (has_flat) {
				jget_bool(patch, "aim_enabled", c.aim_enabled);
				c.aim_full.enabled = c.aim_enabled;
				apply_obs_global_keys(c, patch);
				// only push profile fields if present
				apply_profile_object(c.aim_full.profiles[slot], patch);
				if (patch.contains("aim_enabled")) {
					c.aim_full.profiles[slot].enabled = c.aim_enabled;
					c.aim_full.profiles[slot].mc.enableMouseControl = c.aim_enabled;
				}
			}
		}

		// OBS flat keys at root: enable_config_0, p_min_0, …
		for (int s = 0; s < FullAimSettings::kSlots; ++s)
			apply_obs_slot_keys(c, patch, s);
		apply_obs_global_keys(c, patch);

			// vision / prediction / crosshair nested
if (patch.contains("vision") && patch["vision"].is_object()) {
					const json &vis = patch["vision"];
					jget_bool(vis, "show_detection_results", c.vision.show_detection_results);
					jget_int(vis, "bbox_line_width", c.vision.bbox_line_width);
					jget_float(vis, "label_font_scale", c.vision.label_font_scale);
					jget_bool(vis, "export_coordinates", c.vision.export_coordinates);
					jget_str(vis, "coordinate_output_path", c.vision.coordinate_output_path);
					jget_bool(vis, "show_floating_window", c.vision.show_floating_window);
					jget_int(vis, "floating_window_width", c.vision.floating_window_width);
					jget_int(vis, "floating_window_height", c.vision.floating_window_height);
					jget_bool(vis, "show_track_id_in_floating_window",
					          c.vision.show_track_id_in_floating_window);
					jget_bool(vis, "preview_enabled", c.vision.preview_enabled);
				apply_obs_global_keys(c, vis);
				}
				// flat vision aliases (OBS keys)
				jget_bool(patch, "show_detection_results", c.vision.show_detection_results);
				jget_int(patch, "bbox_line_width", c.vision.bbox_line_width);
				jget_bool(patch, "export_coordinates", c.vision.export_coordinates);
				jget_str(patch, "coordinate_output_path", c.vision.coordinate_output_path);
				jget_bool(patch, "show_floating_window", c.vision.show_floating_window);
				jget_int(patch, "floating_window_width", c.vision.floating_window_width);
				jget_int(patch, "floating_window_height", c.vision.floating_window_height);
				jget_bool(patch, "show_track_id_in_floating_window",
				          c.vision.show_track_id_in_floating_window);
				jget_bool(patch, "preview_enabled", c.vision.preview_enabled);
			if (patch.contains("engine") && patch["engine"].is_object() &&
				    patch["engine"].contains("preview_interval_sec") &&
				    patch["engine"]["preview_interval_sec"].is_number()) {
					c.preview_interval_sec =
					    patch["engine"]["preview_interval_sec"].get<double>();
				}
				if (patch.contains("preview_interval_sec") &&
				    patch["preview_interval_sec"].is_number()) {
					c.preview_interval_sec = patch["preview_interval_sec"].get<double>();
				}
		if (patch.contains("prediction") && patch["prediction"].is_object()) {
			// apply prediction fields to selected config_select slot
			int s = c.aim_full.config_select;
			if (s < 0 || s >= FullAimSettings::kSlots)
				s = 0;
			apply_profile_object(c.aim_full.profiles[s], patch["prediction"]);
			apply_obs_global_keys(c, patch["prediction"]);
		}
		if (patch.contains("crosshair") && patch["crosshair"].is_object()) {
			jget_bool2(patch["crosshair"], "crosshair_enabled", "enabled",
			           c.aim_full.crosshair_enabled);
			apply_obs_global_keys(c, patch["crosshair"]);
		}

		c.aim_full.enabled = c.aim_enabled;
		// stamp globals into profiles
		for (auto &p : c.aim_full.profiles) {
			c.aim_full.stamp_globals(p.mc);
			p.mc.continuousAimEnabled = p.continuous_aim;
			p.mc.enableMouseControl = p.enabled;
		}
#endif
		return true;
	} catch (const std::exception &e) {
		if (err)
			*err = e.what();
		return false;
	}
}

// Fix the intentional bad line in apply_obs_global_keys - I left a broken jget_int2
// I'll fix that below when writing - wait I already have broken code. Let me rewrite that function part.

bool ConfigStore::load_file(const std::string &path, EngineConfig &c, std::string *err)
{
	try {
		std::ifstream ifs(path);
		if (!ifs) {
			if (err)
				*err = "cannot open " + path;
			return false;
		}
		json j;
		ifs >> j;
		return merge_into(c, j, err);
	} catch (const std::exception &e) {
		if (err)
			*err = e.what();
		return false;
	}
}

bool ConfigStore::save_file(const std::string &path, const EngineConfig &c, std::string *err)
	{
		try {
			if (path.empty()) {
				if (err)
					*err = "empty path";
				return false;
			}
			namespace fs = std::filesystem;
			const fs::path p(path);
			std::error_code ec;
			if (p.has_parent_path())
				fs::create_directories(p.parent_path(), ec);
			// Atomic-ish write: temp then rename (avoid half-written user.json on crash).
			const fs::path tmp = p.string() + ".tmp";
			{
				std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
				if (!ofs) {
					if (err)
						*err = "cannot write " + tmp.string();
					return false;
				}
				const std::string body = to_json(c).dump(2);
				ofs << body;
				if (!ofs.good()) {
					if (err)
						*err = "write incomplete " + tmp.string();
					return false;
				}
				ofs.flush();
			}
			fs::remove(p, ec);
			fs::rename(tmp, p, ec);
			if (ec) {
				// Fallback: direct overwrite if rename fails (Windows antivirus etc.)
				std::ofstream ofs(p, std::ios::binary | std::ios::trunc);
				if (!ofs) {
					if (err)
						*err = "cannot write " + path + " (" + ec.message() + ")";
					fs::remove(tmp, ec); // cleanup stale .tmp
					return false;
				}
				ofs << to_json(c).dump(2);
				fs::remove(tmp, ec); // cleanup .tmp after fallback
			}
			return true;
		} catch (const std::exception &e) {
			if (err)
				*err = e.what();
			return false;
		}
	}

MouseControllerConfig ConfigStore::to_mouse_config(const EngineConfig &c, int slot)
{
#ifdef YA_WITH_AIM
	if (slot < 0 || slot >= FullAimSettings::kSlots)
		slot = 0;
	MouseControllerConfig cfg = c.aim_full.profiles[slot].mc;
	c.aim_full.stamp_globals(cfg);
	cfg.continuousAimEnabled = c.aim_full.profiles[slot].continuous_aim;
	cfg.enableMouseControl = c.aim_full.profiles[slot].enabled;
	return cfg;
#else
	(void)c;
	(void)slot;
	return {};
#endif
}

} // namespace ya
