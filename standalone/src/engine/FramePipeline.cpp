// M3 帧管线实现。

#include "FramePipeline.hpp"

#include <chrono>
#include <cstdio>

namespace ya {

FullAimSettings aim_settings_from_config(const config::AimSection& a)
{
    FullAimSettings s;
    s.enabled = a.enabled;
    s.config_select = a.config_select;
    s.algorithm = a.algorithm;

    s.fov_radius = a.fov_radius;
    s.show_fov = a.show_fov;
    s.show_fov_circle = a.show_fov_circle;
    s.show_fov_cross = a.show_fov_cross;
    s.fov_cross_line_scale = a.fov_cross_line_scale;
    s.fov_cross_line_thickness = a.fov_cross_line_thickness;
    s.fov_circle_thickness = a.fov_circle_thickness;
    s.fov_color = a.fov_color;
    s.use_dynamic_fov = a.use_dynamic_fov;
    s.show_fov2 = a.show_fov2;
    s.fov_radius2 = a.fov_radius2;
    s.fov_color2 = a.fov_color2;
    s.dynamic_fov_shrink_percent = a.dynamic_fov_shrink_percent;
    s.dynamic_fov_transition_ms = a.dynamic_fov_transition_ms;

    s.target_switch_delay_ms = a.target_switch_delay_ms;
    s.target_switch_tolerance = a.target_switch_tolerance;

    s.external_kp_x = a.external_kp_x;
    s.external_ki_x = a.external_ki_x;
    s.external_kd_x = a.external_kd_x;
    s.external_kp_y = a.external_kp_y;
    s.external_ki_y = a.external_ki_y;
    s.external_kd_y = a.external_kd_y;
    s.external_predict_x = a.external_predict_x;
    s.external_predict_y = a.external_predict_y;
    s.external_rate_x = a.external_rate_x;
    s.external_rate_y = a.external_rate_y;
    s.external_ki_mode = a.external_ki_mode;
    s.external_kp_limit = a.external_kp_limit;
    s.external_ki_limit = a.external_ki_limit;
    s.external_kd_limit = a.external_kd_limit;
    s.external_output_limit = a.external_output_limit;
    s.external_ki_rate = a.external_ki_rate;
    s.external_ki_deadband = a.external_ki_deadband;

    s.aim_kp = a.aim_kp;
    s.aim_ki = a.aim_ki;
    s.aim_kd = a.aim_kd;
    s.aim_noise_enabled = a.aim_noise_enabled;
    s.aim_noise_amplitude = a.aim_noise_amplitude;
    s.aim_prediction_weight_x = a.aim_prediction_weight_x;
    s.aim_prediction_weight_y = a.aim_prediction_weight_y;
    s.aim_ramp_time = a.aim_ramp_time;
    s.aim_init_scale = a.aim_init_scale;
    s.aim_output_max = a.aim_output_max;

    s.enable_neural_path = a.enable_neural_path;
    s.neural_path_points = a.neural_path_points;
    s.neural_mouse_step_size = a.neural_mouse_step_size;
    s.neural_target_radius = a.neural_target_radius;
    s.neural_consume_per_frame = a.neural_consume_per_frame;
    s.enable_neural_path_debug = a.enable_neural_path_debug;

    s.crosshair_enabled = a.crosshair_enabled;
    s.aim_origin_x = a.aim_origin_x;
    s.aim_origin_y = a.aim_origin_y;
    s.crosshair_h_min = a.crosshair_h_min;
    s.crosshair_h_max = a.crosshair_h_max;
    s.crosshair_s_min = a.crosshair_s_min;
    s.crosshair_s_max = a.crosshair_s_max;
    s.crosshair_v_min = a.crosshair_v_min;
    s.crosshair_v_max = a.crosshair_v_max;
    s.crosshair_manual_r = a.crosshair_manual_r;
    s.crosshair_manual_g = a.crosshair_manual_g;
    s.crosshair_manual_b = a.crosshair_manual_b;
    s.crosshair_h_tolerance = a.crosshair_h_tolerance;
    s.crosshair_s_tolerance = a.crosshair_s_tolerance;
    s.crosshair_v_tolerance = a.crosshair_v_tolerance;
    s.crosshair_morph_kernel = a.crosshair_morph_kernel;
    s.crosshair_erode_iter = a.crosshair_erode_iter;
    s.crosshair_dilate_iter = a.crosshair_dilate_iter;
    s.crosshair_grid_rows = a.crosshair_grid_rows;
    s.crosshair_grid_cols = a.crosshair_grid_cols;
    s.crosshair_quantile_threshold = a.crosshair_quantile_threshold;
    s.crosshair_template_path = a.crosshair_template_path;
    s.crosshair_match_threshold = a.crosshair_match_threshold;
    s.crosshair_min_area = a.crosshair_min_area;
    s.crosshair_max_area = a.crosshair_max_area;
    s.crosshair_shape_filter_enabled = a.crosshair_shape_filter_enabled;
    s.crosshair_shape_type = a.crosshair_shape_type;
    s.crosshair_min_fill_ratio = a.crosshair_min_fill_ratio;
    s.crosshair_max_fill_ratio = a.crosshair_max_fill_ratio;
    s.crosshair_min_aspect_ratio = a.crosshair_min_aspect_ratio;
    s.crosshair_max_aspect_ratio = a.crosshair_max_aspect_ratio;
    s.crosshair_detect_interval = a.crosshair_detect_interval;
    s.crosshair_search_radius = a.crosshair_search_radius;
    s.crosshair_color_isolation = a.crosshair_color_isolation;
    s.crosshair_debug_mask = a.crosshair_debug_mask;

    for (int i = 0; i < FullAimSettings::kSlots; ++i) {
        s.profiles[i].enabled = a.slots[i].enabled;
        s.profiles[i].continuous_aim = a.slots[i].continuous_aim;
        s.profiles[i].mc = a.slots[i].mc;
    }
    return s;
}

FramePipeline::FramePipeline() = default;

void FramePipeline::attach(InferEngine* infer, TrackerEngine* tracker, FullAimBridge* aim)
{
    infer_ = infer;
    tracker_ = tracker;
    aim_ = aim;
}

void FramePipeline::process(const FramePacket& frame)
{
    const auto t0 = std::chrono::steady_clock::now();

    // 1. 推理
    std::vector<Detection> dets;
    double infer_ms = 0.0;
    if (infer_ && infer_->ready()) {
        InferResult r = infer_->run_bgr(frame.bgr.data(), frame.width, frame.height,
                                        frame.width * frame.channels);
        infer_ms = r.infer_ms;
        if (r.ok)
            dets = std::move(r.dets);
    }

    // 2. 跟踪
    if (tracker_ && !dets.empty())
        dets = tracker_->update(dets);
    else if (tracker_)
        tracker_->update({});

    // 3. 瞄准(每帧 tick,与 OBS video_tick 等价)
    FullAimStatus aim_status;
    if (aim_ && aim_ != nullptr) {
        aim_->tick(dets, frame.width, frame.height, frame.origin_x, frame.origin_y,
                   static_cast<float>(infer_ms));
        aim_status = aim_->status();
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 4. 统计累积
    stats_.frames++;
    stats_.detections = dets.size();
    stats_.last_dets = std::move(dets);
    stats_.infer_ms = infer_ms;
    stats_.post_ms = total_ms - infer_ms;
    stats_.aim_status = aim_status;
}

void FramePipeline::reset_stats()
{
    stats_ = PipelineStats();
}

} // namespace ya
