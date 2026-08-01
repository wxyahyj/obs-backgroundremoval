# 配置键对照表 (config-parity)

> 生成: 2026-08-02 · 源: src/filter_properties.cpp vs standalone/src/core/ConfigStore.cpp(旧) vs standalone/webui/index.html
> 状态: S=旧ConfigStore已覆盖 · W=webui已覆盖 · - =缺口(M2/M4 需补)

| 键 | OBS→Store | OBS→WebUI |
|---|---|---|
| advanced_group | - | - |
| aim_controller_group | - | - |
| aim_init_scale | ✓ | ✓ |
| aim_kd | ✓ | ✓ |
| aim_ki | ✓ | ✓ |
| aim_kp | ✓ | ✓ |
| aim_noise_amplitude | ✓ | ✓ |
| aim_noise_enabled | ✓ | ✓ |
| aim_output_max | ✓ | ✓ |
| aim_prediction_weight_x | ✓ | ✓ |
| aim_prediction_weight_y | ✓ | ✓ |
| aim_ramp_time | ✓ | ✓ |
| algorithm_type_global | ✓ | ✓ |
| avg_inference_time | - | - |
| bbox_color | - | - |
| bbox_line_width | ✓ | ✓ |
| confidence_threshold | ✓ | - |
| config_management_group | - | - |
| coordinate_output_path | ✓ | ✓ |
| crosshair_apply_rgb | - | - |
| crosshair_color_info | - | - |
| crosshair_color_isolation | ✓ | ✓ |
| crosshair_debug_mask | ✓ | ✓ |
| crosshair_detect_interval | ✓ | ✓ |
| crosshair_dilate_iter | ✓ | ✓ |
| crosshair_enabled | ✓ | ✓ |
| crosshair_erode_iter | ✓ | ✓ |
| crosshair_grid_cols | ✓ | ✓ |
| crosshair_grid_rows | ✓ | ✓ |
| crosshair_group | - | - |
| crosshair_h_max | ✓ | ✓ |
| crosshair_h_min | ✓ | ✓ |
| crosshair_h_tolerance | ✓ | ✓ |
| crosshair_manual_b | ✓ | ✓ |
| crosshair_manual_g | ✓ | ✓ |
| crosshair_manual_r | ✓ | ✓ |
| crosshair_match_threshold | ✓ | ✓ |
| crosshair_max_area | ✓ | ✓ |
| crosshair_max_aspect_ratio | ✓ | ✓ |
| crosshair_max_fill_ratio | ✓ | ✓ |
| crosshair_min_area | ✓ | ✓ |
| crosshair_min_aspect_ratio | ✓ | ✓ |
| crosshair_min_fill_ratio | ✓ | ✓ |
| crosshair_morph_kernel | ✓ | ✓ |
| crosshair_pick_color | - | - |
| crosshair_quantile_threshold | ✓ | ✓ |
| crosshair_s_max | ✓ | ✓ |
| crosshair_s_min | ✓ | ✓ |
| crosshair_s_tolerance | ✓ | ✓ |
| crosshair_search_radius | ✓ | ✓ |
| crosshair_shape_filter_enabled | ✓ | ✓ |
| crosshair_shape_type | ✓ | ✓ |
| crosshair_template_path | ✓ | ✓ |
| crosshair_v_max | ✓ | ✓ |
| crosshair_v_min | ✓ | ✓ |
| crosshair_v_tolerance | ✓ | ✓ |
| detected_objects | - | - |
| detection_group | - | - |
| dml_stats | - | - |
| dynamic_fov_shrink_percent | ✓ | ✓ |
| dynamic_fov_transition_time | ✓ | ✓ |
| enable_neural_path | ✓ | ✓ |
| enable_neural_path_debug | ✓ | ✓ |
| export_coordinates | ✓ | ✓ |
| external_kd_limit | ✓ | - |
| external_kd_x | ✓ | ✓ |
| external_kd_y | ✓ | ✓ |
| external_ki_deadband | ✓ | ✓ |
| external_ki_limit | ✓ | - |
| external_ki_mode | ✓ | ✓ |
| external_ki_rate | ✓ | ✓ |
| external_ki_x | ✓ | ✓ |
| external_ki_y | ✓ | ✓ |
| external_kp_limit | ✓ | - |
| external_kp_x | ✓ | ✓ |
| external_kp_y | ✓ | ✓ |
| external_output_limit | ✓ | ✓ |
| external_pid_group | - | - |
| external_predict_x | ✓ | ✓ |
| external_predict_y | ✓ | ✓ |
| external_rate_x | ✓ | ✓ |
| external_rate_y | ✓ | ✓ |
| floating_window_group | - | - |
| floating_window_height | - | ✓ |
| floating_window_width | - | ✓ |
| fov2_group | - | - |
| fov_circle_thickness | ✓ | ✓ |
| fov_color | ✓ | - |
| fov_color2 | ✓ | - |
| fov_cross_line_scale | ✓ | ✓ |
| fov_cross_line_thickness | ✓ | ✓ |
| fov_group | - | - |
| fov_radius | ✓ | ✓ |
| fov_radius2 | ✓ | ✓ |
| inference_interval_frames | - | - |
| inference_status | - | - |
| input_resolution | ✓ | - |
| iou_threshold | - | ✓ |
| kalman_generate_threshold | - | ✓ |
| kalman_prediction_frames | - | ✓ |
| kalman_terminate_count | - | ✓ |
| label_font_scale | - | ✓ |
| load_config | - | - |
| max_lost_frames | - | ✓ |
| max_reidentify_frames | - | ✓ |
| model_group | - | - |
| model_path | ✓ | ✓ |
| model_version | ✓ | ✓ |
| mouse_config_select | ✓ | ✓ |
| neural_consume_per_frame | - | ✓ |
| neural_mouse_step_size | - | ✓ |
| neural_path_points | - | ✓ |
| neural_target_radius | - | ✓ |
| nms_threshold | ✓ | - |
| num_threads | - | ✓ |
| region_group | - | - |
| region_height | - | ✓ |
| region_width | - | ✓ |
| region_x | - | ✓ |
| region_y | - | ✓ |
| reidentify_center_threshold | - | ✓ |
| render_group | - | - |
| save_config | - | - |
| settings_page | - | - |
| show_detection_results | ✓ | ✓ |
| show_floating_window | ✓ | ✓ |
| show_fov | ✓ | ✓ |
| show_fov2 | ✓ | ✓ |
| show_fov_circle | ✓ | ✓ |
| show_fov_cross | ✓ | ✓ |
| show_kalman_predictions | ✓ | ✓ |
| show_kalman_trajectories | ✓ | ✓ |
| show_pid_debug_window | - | - |
| show_track_id_in_floating_window | ✓ | ✓ |
| target_class | ✓ | ✓ |
| target_classes_text | ✓ | ✓ |
| target_switch_delay | ✓ | ✓ |
| target_switch_tolerance | ✓ | ✓ |
| test_makcu_connection | - | - |
| toggle_inference | - | - |
| tracking_group | - | - |
| tracking_weight_area | ✓ | ✓ |
| tracking_weight_aspect | ✓ | ✓ |
| tracking_weight_center | ✓ | ✓ |
| tracking_weight_iou | ✓ | ✓ |
| use_dynamic_fov | ✓ | ✓ |
| use_gpu | - | - |
| use_gpu_texture_inference | - | - |
| use_kalman_tracker | ✓ | ✓ |
| use_region | - | ✓ |

## 缺口汇总
- ConfigStore 未覆盖: 50 键
- WebUI 未覆盖: 38 键

## 键分类说明
- *_group / *_page / *_status / *_stats / *_info: 仅 UI 分组或只读标签,不持久化,不算缺口
- save_config / load_config / test_* / toggle_* / pick_*: 动作按钮(API 端点),非数据键
- 其余真实缺口(M2 ConfigModel 需覆盖): bbox_color, floating_window_*, inference_interval_frames,
  iou_threshold, kalman_*, max_lost_frames, max_reidentify_frames, reidentify_center_threshold,
  label_font_scale, neural_*, num_threads, region_*, use_gpu, use_gpu_texture_inference,
  use_region, crosshair_apply_rgb, crosshair_color_info
