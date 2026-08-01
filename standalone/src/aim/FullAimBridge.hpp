#pragma once

#ifdef _WIN32

#include "MouseControllerInterface.hpp"
#include "models/Detection.h"

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ya {

// One OBS-equivalent mouse profile slot (maps to enable_config_%d / hotkey_%d / …).
struct AimProfile {
	bool enabled = false;
	bool continuous_aim = false;
	MouseControllerConfig mc;
};

// Global aim settings shared across profiles (algorithm panels, FOV, external/aim PID, …).
// Mirrors filter_properties.cpp + applyConfigToController in yolo-detector-filter.
struct FullAimSettings {
	static constexpr int kSlots = 5;

	bool enabled = true;
	int config_select = 0; // UI selection; runtime uses hotkey/continuous
	AlgorithmType algorithm = AlgorithmType::AdvancedPID;

	// FOV (pixels in capture frame) — OBS: fov_radius, use_dynamic_fov, …
	int fov_radius = 120;
	bool show_fov = true;
	bool show_fov_circle = true;
	bool show_fov_cross = false;
	int fov_cross_line_scale = 100;
	int fov_cross_line_thickness = 1;
	int fov_circle_thickness = 2;
	int fov_color = 0x00FF00; // BGR-ish UI color int
	bool use_dynamic_fov = false;
	bool show_fov2 = false;
	int fov_radius2 = 80;
	int fov_color2 = 0x00FFFF;
	float dynamic_fov_shrink_percent = 0.7f; // 0.1–1.0 (OBS stores 10–100; we use fraction)
	float dynamic_fov_transition_ms = 200.f;

	// Target switch (global)
	int target_switch_delay_ms = 500;
	float target_switch_tolerance = 0.15f;

	// External / 专业 PID (global — applied to every slot on tick)
	float external_kp_x = 1.5f, external_ki_x = 0.f, external_kd_x = 1.5f;
	float external_kp_y = 1.5f, external_ki_y = 0.f, external_kd_y = 1.5f;
	float external_predict_x = 1.f, external_predict_y = 1.f;
	float external_rate_x = 0.3f, external_rate_y = 0.3f;
	float external_ki_mode = 1.f;
	float external_kp_limit = 9900.f, external_ki_limit = 9900.f, external_kd_limit = 9900.f;
	float external_output_limit = 0.f, external_ki_rate = 0.05f, external_ki_deadband = 0.5f;

	// Aim controller (global)
	float aim_kp = 0.6f, aim_ki = 0.01f, aim_kd = 0.007f;
	bool aim_noise_enabled = false;
	float aim_noise_amplitude = 2.f;
	float aim_prediction_weight_x = 0.3f, aim_prediction_weight_y = 0.1f;
	float aim_ramp_time = 0.3f, aim_init_scale = 0.6f, aim_output_max = 128.f;

	// Neural path (OBS global flags copied into each slot on apply)
	bool enable_neural_path = false;
	int neural_path_points = 25;
	double neural_mouse_step_size = 8.0;
	int neural_target_radius = 8;
	int neural_consume_per_frame = 2;
	bool enable_neural_path_debug = false;

	// Crosshair aim origin (pixel in frame; -1 = center)
	bool crosshair_enabled = false;
	float aim_origin_x = -1.f;
	float aim_origin_y = -1.f;
	int crosshair_h_min = 0, crosshair_h_max = 180;
	int crosshair_s_min = 100, crosshair_s_max = 255;
	int crosshair_v_min = 100, crosshair_v_max = 255;
	int crosshair_manual_r = 0, crosshair_manual_g = 255, crosshair_manual_b = 0;
	int crosshair_h_tolerance = 10, crosshair_s_tolerance = 40, crosshair_v_tolerance = 40;
	int crosshair_morph_kernel = 3, crosshair_erode_iter = 0, crosshair_dilate_iter = 1;
	int crosshair_grid_rows = 4, crosshair_grid_cols = 4;
	float crosshair_quantile_threshold = 0.01f;
	std::string crosshair_template_path;
	float crosshair_match_threshold = 0.6f;
	int crosshair_min_area = 10, crosshair_max_area = 50000;
	bool crosshair_shape_filter_enabled = false;
	int crosshair_shape_type = 0;
	float crosshair_min_fill_ratio = 0.05f, crosshair_max_fill_ratio = 0.8f;
	float crosshair_min_aspect_ratio = 0.3f, crosshair_max_aspect_ratio = 3.0f;
	int crosshair_detect_interval = 1;
	int crosshair_search_radius = 0;
	bool crosshair_color_isolation = false;
	bool crosshair_debug_mask = false;

	std::array<AimProfile, kSlots> profiles{};

	FullAimSettings()
	{
		// Slot 0 enabled by default — mirrors practical first-use (OBS defaults all false;
		// standalone enables slot0 so product works out of the box).
		profiles[0].enabled = true;
		profiles[0].mc.enableMouseControl = true;
		profiles[0].mc.hotkeyVirtualKey = 0x02; // RMB
		profiles[0].mc.controllerType = ControllerType::WindowsAPI;
		profiles[0].mc.fovRadiusPixels = 120;
		profiles[0].mc.algorithmType = AlgorithmType::AdvancedPID;
		profiles[0].mc.pidPMin = 0.153f;
		profiles[0].mc.pidPMax = 0.6f;
		profiles[0].mc.pidPSlope = 1.0f;
		profiles[0].mc.pidD = 0.007f;
		profiles[0].mc.pidI = 0.01f;
		profiles[0].mc.deadZonePixels = 5.f;
		profiles[0].mc.maxPixelMove = 128.f;
		profiles[0].mc.makcuPort = "COM5";
		profiles[0].mc.makcuBaudRate = 4000000;
		profiles[0].mc.useDerivativePredictor = true;
		profiles[0].mc.predictionWeightX = 0.5f;
		profiles[0].mc.predictionWeightY = 0.1f;

		for (int i = 1; i < kSlots; ++i) {
			profiles[i].mc.hotkeyVirtualKey = 0x05; // XBUTTON1 OBS default
			profiles[i].mc.makcuPort = "COM5";
			profiles[i].mc.makcuBaudRate = 4000000;
			profiles[i].mc.pidPMin = 0.153f;
			profiles[i].mc.pidPMax = 0.6f;
			profiles[i].mc.useDerivativePredictor = true;
		}
	}

	// Stamp global algorithm / external / aim / neural into a per-slot MouseControllerConfig.
	void stamp_globals(MouseControllerConfig &cfg) const
	{
		cfg.algorithmType = algorithm;
		cfg.fovRadiusPixels = fov_radius;
		cfg.targetSwitchDelayMs = target_switch_delay_ms;
		cfg.targetSwitchTolerance = target_switch_tolerance;

		cfg.externalKpX = external_kp_x;
		cfg.externalKiX = external_ki_x;
		cfg.externalKdX = external_kd_x;
		cfg.externalKpY = external_kp_y;
		cfg.externalKiY = external_ki_y;
		cfg.externalKdY = external_kd_y;
		cfg.externalPredictX = external_predict_x;
		cfg.externalPredictY = external_predict_y;
		cfg.externalRateX = external_rate_x;
		cfg.externalRateY = external_rate_y;
		cfg.externalKiMode = external_ki_mode;
		cfg.externalKpLimit = external_kp_limit;
		cfg.externalKiLimit = external_ki_limit;
		cfg.externalKdLimit = external_kd_limit;
		cfg.externalOutputLimit = external_output_limit;
		cfg.externalKiRate = external_ki_rate;
		cfg.externalKiDeadband = external_ki_deadband;

		cfg.aimKp = aim_kp;
		cfg.aimKi = aim_ki;
		cfg.aimKd = aim_kd;
		cfg.aimNoiseEnabled = aim_noise_enabled;
		cfg.aimNoiseAmplitude = aim_noise_amplitude;
		cfg.aimPredictionWeightX = aim_prediction_weight_x;
		cfg.aimPredictionWeightY = aim_prediction_weight_y;
		cfg.aimRampTime = aim_ramp_time;
		cfg.aimInitScale = aim_init_scale;
		cfg.aimOutputMax = aim_output_max;

		cfg.enableNeuralPath = enable_neural_path;
		cfg.neuralPathPoints = neural_path_points;
		cfg.neuralMouseStepSize = neural_mouse_step_size;
		cfg.neuralTargetRadius = neural_target_radius;
		cfg.neuralConsumePerFrame = neural_consume_per_frame;
		cfg.enableNeuralPathDebug = enable_neural_path_debug;
	}
};

struct FullAimStatus {
	bool controller_ok = false;
	int active_slot = -1;
	int controller_type = 0;
	int algorithm = 0;
	int fov_px = 0;
	bool hotkey_down = false;
	bool aiming = false;
	std::string last_error;
};

// Bridges host EngineLoop → parent MouseControllerInterface (full OBS aim stack).
class FullAimBridge {
public:
	FullAimBridge();
	~FullAimBridge();

	void set_settings(const FullAimSettings &s);
	FullAimSettings settings() const;

	// Call once at engine start (creates default WindowsAPI controller).
	bool ensure_controller();

	// Per-frame (OBS video_tick equivalent).
	// dets: tracked detections (normalized). frame_w/h = capture size.
	void tick(const std::vector<Detection> &dets, int frame_w, int frame_h, int crop_x,
	          int crop_y, float infer_ms);

	FullAimStatus status() const;

	// Test backend connectivity (MAKCU/Logi etc.) without aiming.
	bool test_controller(ControllerType type, const std::string &makcu_port, int makcu_baud,
	                     int logi_type, std::string *err_out);

	static const char *controller_name(ControllerType t);
	static const char *algorithm_name(AlgorithmType a);
	static ControllerType parse_controller(const std::string &s);
	static ControllerType parse_controller_int(int v);
	static AlgorithmType parse_algorithm(int v);

private:
	int select_active_slot() const; // continuous first, else hotkey
	void apply_slot(int slot, int frame_w, int frame_h);
	void recreate_if_needed(ControllerType type, const std::string &makcu_port, int baud,
	                        int logi_type);
	float update_dynamic_fov(bool has_target_in_fov);

	mutable std::mutex mu_;
	FullAimSettings settings_;
	std::unique_ptr<MouseControllerInterface> controller_;
	ControllerType current_type_ = ControllerType::WindowsAPI;
	std::string current_makcu_port_;
	int current_makcu_baud_ = 0;
	int current_logi_type_ = 0;

	float current_fov_ = 120.f;
	int last_active_slot_ = -1;
	FullAimStatus status_{};
};

} // namespace ya

#endif // _WIN32
