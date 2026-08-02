// OBS 分类配置页结构 + 字段定位(M4 分类对齐)
// 页面/分组来自 filter_properties.cpp onPageChanged 可见性逻辑。
// path: ConfigDocument JSON 路径(slot 字段用 "{i}" 占位当前槽索引)。

const OBS_PAGES = [
  {
    name: "模型与检测",
    groups: [
      { name: "模型配置", fields: [
        ["model_path", ["infer", "model_path"]],
        ["model_version", ["infer", "model_version"]],
        ["use_gpu", ["infer", "device"]],
        ["input_resolution", ["infer", "input_resolution"]],
        ["num_threads", ["infer", "num_threads"]],
      ]},
      { name: "检测配置", fields: [
        ["confidence_threshold", ["infer", "confidence"]],
        ["nms_threshold", ["infer", "nms"]],
        ["target_classes_text", ["infer", "target_classes"]],
        ["inference_interval_frames", ["infer", "interval_frames"]],
      ]},
    ],
  },
  {
    name: "视觉与区域",
    groups: [
      { name: "渲染配置", fields: [
        ["show_detection_results", ["vision", "show_detection_results"]],
        ["bbox_line_width", ["vision", "bbox_line_width"]],
        ["label_font_scale", ["vision", "label_font_scale"]],
      ]},
      { name: "区域检测", fields: [
        ["use_region", ["capture", "use_region"]],
        ["region_x", ["capture", "region_x"]],
        ["region_y", ["capture", "region_y"]],
        ["region_width", ["capture", "region_width"]],
        ["region_height", ["capture", "region_height"]],
        ["backend", ["capture", "backend"]],
        ["mode", ["capture", "mode"]],
        ["max_fps", ["capture", "max_fps"]],
      ]},
      { name: "FOV 设置", fields: [
        ["show_fov", ["aim", "show_fov"]],
        ["fov_radius", ["aim", "fov_radius"]],
        ["show_fov_circle", ["aim", "show_fov_circle"]],
        ["show_fov_cross", ["aim", "show_fov_cross"]],
        ["fov_cross_line_scale", ["aim", "fov_cross_line_scale"]],
        ["fov_cross_line_thickness", ["aim", "fov_cross_line_thickness"]],
        ["fov_circle_thickness", ["aim", "fov_circle_thickness"]],
        ["fov_color", ["aim", "fov_color"]],
      ]},
      { name: "动态 FOV 设置", fields: [
        ["use_dynamic_fov", ["aim", "use_dynamic_fov"]],
        ["show_fov2", ["aim", "show_fov2"]],
        ["fov_radius2", ["aim", "fov_radius2"]],
        ["fov_color2", ["aim", "fov_color2"]],
        ["dynamic_fov_shrink_percent", ["aim", "dynamic_fov_shrink_percent"]],
        ["dynamic_fov_transition_time", ["aim", "dynamic_fov_transition_ms"]],
      ]},
    ],
  },
  {
    name: "鼠标控制 - 基础",
    slot: true,
    groups: [
      { name: "基础", fields: [
        ["enable_config", ["{i}", "enabled"]],
        ["continuous_aim", ["{i}", "continuous_aim"]],
        ["hotkey", ["{i}", "mc", "hotkeyVirtualKey"]],
        ["controller_type", ["{i}", "mc", "controllerType"]],
        ["makcu_port", ["{i}", "mc", "makcuPort"]],
        ["makcu_baud_rate", ["{i}", "mc", "makcuBaudRate"]],
        ["logi_driver_type", ["{i}", "mc", "logiDriverType"]],
        ["screen_offset_x", ["{i}", "mc", "screenOffsetX"]],
        ["screen_offset_y", ["{i}", "mc", "screenOffsetY"]],
        ["screen_width", ["{i}", "mc", "screenWidth"]],
        ["screen_height", ["{i}", "mc", "screenHeight"]],
        ["enable_y_axis_unlock", ["{i}", "mc", "yUnlockEnabled"]],
        ["y_axis_unlock_delay", ["{i}", "mc", "yUnlockDelayMs"]],
        ["target_y_offset", ["{i}", "mc", "targetYOffset"]],
        ["dead_zone_pixels", ["{i}", "mc", "deadZonePixels"]],
        ["max_pixel_move", ["{i}", "mc", "maxPixelMove"]],
      ]},
    ],
  },
  {
        name: "鼠标控制 - PID参数",
    slot: true,
    groups: [
      { name: "自适应 PID 参数", fields: [
        ["adaptive_pid_kp", ["{i}", "mc", "adaptivePidKp"]],
        ["adaptive_pid_ki", ["{i}", "mc", "adaptivePidKi"]],
        ["adaptive_pid_kd", ["{i}", "mc", "adaptivePidKd"]],
        ["adaptive_pid_dead_zone", ["{i}", "mc", "adaptivePidDeadZone"]],
        ["adaptive_pid_integral_limit", ["{i}", "mc", "adaptivePidIntegralLimit"]],
        ["adaptive_pid_integral_deadzone", ["{i}", "mc", "adaptivePidIntegralDeadzone"]],
        ["adaptive_pid_integral_gain_threshold", ["{i}", "mc", "adaptivePidIntegralGainThreshold"]],
        ["adaptive_pid_integral_gain_rate", ["{i}", "mc", "adaptivePidIntegralGainRate"]],
        ["adaptive_pid_output_limit", ["{i}", "mc", "adaptivePidOutputLimit"]],
      ]},
    ],
  },
  {
    name: "鼠标控制 - 扳机",
    slot: true,
    groups: [
      { name: "自动扳机", fields: [
        ["auto_trigger_group", ["{i}", "mc", "autoTriggerEnabled"]],
        ["trigger_radius", ["{i}", "mc", "autoTriggerRadius"]],
        ["trigger_cooldown", ["{i}", "mc", "autoTriggerCooldownMs"]],
        ["trigger_fire_delay", ["{i}", "mc", "autoTriggerFireDelay"]],
        ["trigger_fire_duration", ["{i}", "mc", "autoTriggerFireDuration"]],
        ["trigger_interval", ["{i}", "mc", "autoTriggerInterval"]],
        ["enable_trigger_delay_random", ["{i}", "mc", "autoTriggerDelayRandomEnabled"]],
        ["trigger_delay_random_min", ["{i}", "mc", "autoTriggerDelayRandomMin"]],
        ["trigger_delay_random_max", ["{i}", "mc", "autoTriggerDelayRandomMax"]],
        ["enable_trigger_duration_random", ["{i}", "mc", "autoTriggerDurationRandomEnabled"]],
        ["trigger_duration_random_min", ["{i}", "mc", "autoTriggerDurationRandomMin"]],
        ["trigger_duration_random_max", ["{i}", "mc", "autoTriggerDurationRandomMax"]],
        ["trigger_move_compensation", ["{i}", "mc", "autoTriggerMoveCompensation"]],
      ]},
      { name: "后坐力控制", fields: [
        ["recoil_group", ["{i}", "mc", "autoRecoilControlEnabled"]],
        ["recoil_strength", ["{i}", "mc", "recoilStrength"]],
        ["recoil_speed", ["{i}", "mc", "recoilSpeed"]],
        ["recoil_pid_gain_scale", ["{i}", "mc", "recoilPidGainScale"]],
      ]},
    ],
  },
  {
    name: "追踪与高级",
    groups: [
      { name: "目标追踪设置", fields: [
        ["iou_threshold", ["tracker", "iou_threshold"]],
        ["max_lost_frames", ["tracker", "max_lost_frames"]],
        ["max_reidentify_frames", ["tracker", "max_reidentify_frames"]],
        ["reidentify_center_threshold", ["tracker", "reidentify_center_threshold"]],
        ["use_kalman_tracker", ["tracker", "use_kalman"]],
        ["kalman_generate_threshold", ["tracker", "kalman_generate_threshold"]],
        ["kalman_terminate_count", ["tracker", "kalman_terminate_count"]],
        ["kalman_prediction_frames", ["tracker", "kalman_prediction_frames"]],
        ["show_kalman_predictions", ["tracker", "show_kalman_predictions"]],
        ["show_kalman_trajectories", ["tracker", "show_kalman_trajectories"]],
      ]},
      { name: "神经路径", fields: [
        ["enable_neural_path", ["aim", "enable_neural_path"]],
        ["neural_path_points", ["aim", "neural_path_points"]],
        ["neural_mouse_step_size", ["aim", "neural_mouse_step_size"]],
        ["neural_target_radius", ["aim", "neural_target_radius"]],
        ["neural_consume_per_frame", ["aim", "neural_consume_per_frame"]],
        ["enable_neural_path_debug", ["aim", "enable_neural_path_debug"]],
      ]},
      { name: "悬浮窗", fields: [
        ["show_floating_window", ["vision", "show_floating_window"]],
        ["floating_window_width", ["vision", "floating_window_width"]],
        ["floating_window_height", ["vision", "floating_window_height"]],
        ["show_track_id_in_floating_window", ["vision", "show_track_id_in_floating_window"]],
      ]},
      { name: "坐标导出", fields: [
        ["export_coordinates", ["vision", "export_coordinates"]],
        ["coordinate_output_path", ["vision", "coordinate_output_path"]],
      ]},
    ],
  },
  {
    name: "预测与滤波",
    slot: true,
    groups: [
      { name: "导数预测器", fields: [
        ["derivative_predictor_group", ["{i}", "mc", "useDerivativePredictor"]],
        ["max_prediction_time", ["{i}", "mc", "maxPredictionTime"]],
        ["prediction_weight_x", ["{i}", "mc", "predictionWeightX"]],
        ["prediction_weight_y", ["{i}", "mc", "predictionWeightY"]],
      ]},
      { name: "Smith 预估器", fields: [
        ["smith_predictor_group", ["{i}", "mc", "smithPredictorEnabled"]],
        ["smith_model_gain", ["{i}", "mc", "smithModelGain"]],
        ["smith_model_tau", ["{i}", "mc", "smithModelTau"]],
        ["smith_auto_tau", ["{i}", "mc", "smithAutoTau"]],
      ]},
      { name: "SlewRate 控制器", fields: [
        ["slew_rate_group", ["{i}", "mc", "slewRateEnabled"]],
        ["slew_rate_output_gain", ["{i}", "mc", "slewRateOutputGain"]],
        ["slew_rate_response_smoothing", ["{i}", "mc", "slewRateResponseSmoothing"]],
        ["slew_rate_approach_damping", ["{i}", "mc", "slewRateApproachDamping"]],
        ["slew_rate_update_interval_ms", ["{i}", "mc", "slewRateUpdateIntervalMs"]],
        ["slew_rate_normalization_scale", ["{i}", "mc", "slewRateNormalizationScale"]],
      ]},
      
      { name: "贝塞尔曲线", fields: [
        ["bezier_movement_group", ["{i}", "mc", "enableBezierMovement"]],
        ["bezier_curvature", ["{i}", "mc", "bezierCurvature"]],
        ["bezier_randomness", ["{i}", "mc", "bezierRandomness"]],
      ]},
      { name: "曲线轨迹(GhostTracker)", fields: [
        ["ghost_tracker_group", ["{i}", "mc", "enableGhostTracker"]],
        ["ghost_curvature", ["{i}", "mc", "ghostCurvature"]],
        ["ghost_noise_intensity", ["{i}", "mc", "ghostNoiseIntensity"]],
        ["ghost_vertical_snap", ["{i}", "mc", "ghostVerticalSnapRatio"]],
        ["ghost_noise_freq", ["{i}", "mc", "ghostNoiseFreq"]],
      ]},
      { name: "IMM 滤波器", fields: [
        ["imm_filter_group", ["{i}", "mc", "immFilterEnabled"]],
        ["imm_process_noise_pos", ["{i}", "mc", "immProcessNoisePos"]],
        ["imm_process_noise_vel", ["{i}", "mc", "immProcessNoiseVel"]],
        ["imm_process_noise_acc", ["{i}", "mc", "immProcessNoiseAcc"]],
        ["imm_process_noise_turn", ["{i}", "mc", "immProcessNoiseTurn"]],
        ["imm_measurement_noise_x", ["{i}", "mc", "immMeasurementNoiseX"]],
        ["imm_measurement_noise_y", ["{i}", "mc", "immMeasurementNoiseY"]],
        ["imm_active_models", ["{i}", "mc", "immActiveModels"]],
      ]},
      { name: "OneEuro 滤波", fields: [
        ["use_one_euro_filter", ["{i}", "mc", "useOneEuroFilter"]],
        ["one_euro_min_cutoff", ["{i}", "mc", "oneEuroMinCutoff"]],
        ["one_euro_beta", ["{i}", "mc", "oneEuroBeta"]],
        ["one_euro_d_cutoff", ["{i}", "mc", "oneEuroDCutoff"]],
      ]},
    ],
  },
  {
    name: "准星检测",
    groups: [
      { name: "准星检测", fields: [
        ["crosshair_enabled", ["aim", "crosshair_enabled"]],
        ["crosshair_h_min", ["aim", "crosshair_h_min"]], ["crosshair_h_max", ["aim", "crosshair_h_max"]],
        ["crosshair_s_min", ["aim", "crosshair_s_min"]], ["crosshair_s_max", ["aim", "crosshair_s_max"]],
        ["crosshair_v_min", ["aim", "crosshair_v_min"]], ["crosshair_v_max", ["aim", "crosshair_v_max"]],
        ["crosshair_h_tolerance", ["aim", "crosshair_h_tolerance"]],
        ["crosshair_s_tolerance", ["aim", "crosshair_s_tolerance"]],
        ["crosshair_v_tolerance", ["aim", "crosshair_v_tolerance"]],
        ["crosshair_manual_r", ["aim", "crosshair_manual_r"]],
        ["crosshair_manual_g", ["aim", "crosshair_manual_g"]],
        ["crosshair_manual_b", ["aim", "crosshair_manual_b"]],
        ["crosshair_morph_kernel", ["aim", "crosshair_morph_kernel"]],
        ["crosshair_erode_iter", ["aim", "crosshair_erode_iter"]],
        ["crosshair_dilate_iter", ["aim", "crosshair_dilate_iter"]],
        ["crosshair_grid_rows", ["aim", "crosshair_grid_rows"]],
        ["crosshair_grid_cols", ["aim", "crosshair_grid_cols"]],
        ["crosshair_quantile_threshold", ["aim", "crosshair_quantile_threshold"]],
        ["crosshair_template_path", ["aim", "crosshair_template_path"]],
        ["crosshair_match_threshold", ["aim", "crosshair_match_threshold"]],
        ["crosshair_min_area", ["aim", "crosshair_min_area"]],
        ["crosshair_max_area", ["aim", "crosshair_max_area"]],
        ["crosshair_shape_filter_enabled", ["aim", "crosshair_shape_filter_enabled"]],
        ["crosshair_shape_type", ["aim", "crosshair_shape_type"]],
        ["crosshair_min_fill_ratio", ["aim", "crosshair_min_fill_ratio"]],
        ["crosshair_max_fill_ratio", ["aim", "crosshair_max_fill_ratio"]],
        ["crosshair_min_aspect_ratio", ["aim", "crosshair_min_aspect_ratio"]],
        ["crosshair_max_aspect_ratio", ["aim", "crosshair_max_aspect_ratio"]],
        ["crosshair_detect_interval", ["aim", "crosshair_detect_interval"]],
        ["crosshair_search_radius", ["aim", "crosshair_search_radius"]],
        ["crosshair_color_isolation", ["aim", "crosshair_color_isolation"]],
        ["crosshair_debug_mask", ["aim", "crosshair_debug_mask"]],
      ]},
    ],
  },
];

// 字段中文标签(obs 键 → 中文);槽键用 SLOT_LABELS
function fieldLabel(obsKey) {
  if (FIELD_LABELS && FIELD_LABELS[obsKey]) return FIELD_LABELS[obsKey];
  if (SLOT_LABELS && SLOT_LABELS[obsKey]) return SLOT_LABELS[obsKey];
  return obsKey.replace(/_/g, " ");
}

// 下拉框选项(OBS obs_properties_add_list 对齐)
const FIELD_OPTIONS = {
  use_gpu: [["CPU", "cpu"], ["CUDA", "cuda"], ["ROCm", "rocm"], ["TensorRT", "tensorrt"], ["DirectML", "dml"]],
  model_version: [["自动检测", -1], ["YOLOv5", 0], ["YOLOv8", 1], ["YOLOv11", 2]],
  input_resolution: [["320x320", 320], ["416x416", 416], ["640x640", 640], ["960x960", 960], ["1280x1280", 1280]],
  controller_type: [
    ["Windows API", 0], ["MAKCU", 1], ["罗技/雷蛇驱动", 2], ["UU remote GvInput", 3],
    ["NtUserSendInput", 5], ["NtUserInjectMouse", 6], ["NtUserInjectPointer", 7],
  ],
  algorithm_type_global: [["自适应PID(位置式)", 4]],
  logi_driver_type: [["自动检测", 0], ["Logitech G HUB", 1], ["Logitech LGS", 2], ["Razer Synapse", 3]],
  makcu_baud_rate: [["9600", 9600], ["19200", 19200], ["38400", 38400], ["57600", 57600],
                    ["115200", 115200], ["2000000", 2000000], ["4000000", 4000000]],
  hotkey: [
    ["鼠标左键", 1], ["鼠标右键", 2], ["侧键1", 5], ["侧键2", 6],
    ["空格", 32], ["Shift", 16], ["Control", 17],
    ["A", 65], ["D", 68], ["W", 87], ["S", 83],
    ["F1", 112], ["F2", 113], ["F3", 114], ["F4", 115],
  ],
  crosshair_shape_type: [["任意形状", 0], ["十字形(+字)", 1], ["点状(圆点)", 2], ["T字形", 3]],
  backend: [["DXGI", "dxgi"], ["WGC", "wgc"], ["GDI", "gdi"]],
  mode: [["屏幕中心裁剪", "center"], ["固定区域", "region"], ["全屏", "full"]],
  target_class: [["全部类别", -1], ["0", 0], ["1", 1], ["2", 2], ["3", 3], ["4", 4], ["5", 5]],
};

// 滑块字段(OBS slider 对齐):字段 → [min, max, step]
const FIELD_SLIDERS = {"num_threads":[1,16,1],"confidence_threshold":[0.01,1,0.01],"nms_threshold":[0.01,1,0.01],"inference_interval_frames":[0,10,1],"bbox_line_width":[1,5,1],"label_font_scale":[0.2,1,0.05],"fov_radius":[1,500,1],"fov_cross_line_scale":[1,300,5],"fov_cross_line_thickness":[1,10,1],"fov_circle_thickness":[1,10,1],"fov_radius2":[1,200,1],"dynamic_fov_shrink_percent":[10,100,1],"dynamic_fov_transition_time":[0,1000,10],"kalman_generate_threshold":[1,10,1],"kalman_terminate_count":[1,10,1],"kalman_prediction_frames":[1,20,1],"neural_path_points":[10,100,5],"neural_mouse_step_size":[1,20,0.5],"neural_target_radius":[1,50,1],"neural_consume_per_frame":[1,5,1],"iou_threshold":[0.1,0.9,0.05],"max_lost_frames":[0,30,1],"target_switch_delay":[0,1500,50],"target_switch_tolerance":[0,0.5,0.05],"tracking_weight_iou":[0,1,0.05],"tracking_weight_center":[0,1,0.05],"tracking_weight_aspect":[0,1,0.05],"tracking_weight_area":[0,1,0.05],"max_reidentify_frames":[0,60,5],"reidentify_center_threshold":[0.01,0.3,0.01],"floating_window_width":[320,1920,10],"floating_window_height":[240,1080,10],"external_kp_x":[0,10,0.01],"external_ki_x":[0,5,0.001],"external_kd_x":[0,10,0.01],"external_kp_y":[0,10,0.01],"external_ki_y":[0,5,0.001],"external_kd_y":[0,10,0.01],"external_predict_x":[0,5,0.01],"external_predict_y":[0,5,0.01],"external_rate_x":[0,1,0.001],"external_rate_y":[0,1,0.001],"external_ki_mode":[0,1,1],"external_kp_limit":[0,10000,1],"external_ki_limit":[0,10000,1],"external_kd_limit":[0,10000,1],"external_output_limit":[0,10000,1],"external_ki_rate":[0,1,0.001],"external_ki_deadband":[0,10,0.01],"crosshair_manual_r":[0,255,1],"crosshair_manual_g":[0,255,1],"crosshair_manual_b":[0,255,1],"crosshair_h_min":[0,180,1],"crosshair_h_max":[0,180,1],"crosshair_s_min":[0,255,1],"crosshair_s_max":[0,255,1],"crosshair_v_min":[0,255,1],"crosshair_v_max":[0,255,1],"crosshair_h_tolerance":[1,90,1],"crosshair_s_tolerance":[1,128,1],"crosshair_v_tolerance":[1,128,1],"crosshair_morph_kernel":[1,15,2],"crosshair_erode_iter":[0,5,1],"crosshair_dilate_iter":[0,10,1],"crosshair_grid_rows":[2,20,1],"crosshair_grid_cols":[2,20,1],"crosshair_quantile_threshold":[0,1,0.01],"crosshair_match_threshold":[0,1,0.05],"crosshair_min_area":[1,10000,10],"crosshair_max_area":[1,50000,100],"crosshair_min_fill_ratio":[0,1,0.05],"crosshair_max_fill_ratio":[0,1,0.05],"crosshair_min_aspect_ratio":[0.1,5,0.1],"crosshair_max_aspect_ratio":[0.1,5,0.1],"crosshair_search_radius":[0,1920,10],"crosshair_detect_interval":[1,60,1],"recoil_group":[0,50,1],"recoil_strength":[0,50,1],"recoil_speed":[0,1,0.05],"recoil_pid_gain_scale":[0,1,0.05],"derivative_predictor_group":[0,1,0.1],"prediction_weight_x":[0,1,0.1],"prediction_weight_y":[0,1,0.1],"max_prediction_time":[0.01,0.3,0.01],"smith_model_gain":[0.1,5,0.1],"smith_model_tau":[0.005,0.2,0.005],"slew_rate_group":[0,3,0.01],"slew_rate_output_gain":[0,3,0.01],"slew_rate_response_smoothing":[0,0.01,0.0001],"slew_rate_approach_damping":[0,20,0.1],"slew_rate_update_interval_ms":[1,50,0.5],"slew_rate_normalization_scale":[1,30,0.5],"adaptive_pid_group":[0,5,0.01],"adaptive_pid_kp":[0,5,0.01],"adaptive_pid_ki":[0,1,0.001],"adaptive_pid_kd":[0,1,0.001],"adaptive_pid_dead_zone":[0,10,0.01],"adaptive_pid_integral_limit":[1,500,1],"adaptive_pid_integral_deadzone":[0,10,0.01],"adaptive_pid_integral_gain_threshold":[1,200,1],"adaptive_pid_integral_gain_rate":[0.001,0.1,0.001],"adaptive_pid_output_limit":[1,200,1],"bezier_movement_group":[0,1,0.05],"bezier_curvature":[0,1,0.05],"bezier_randomness":[0,0.5,0.05],"ghost_curvature":[0,1,0.05],"ghost_noise_intensity":[0,30,1],"ghost_vertical_snap":[1,10,0.5],"ghost_noise_freq":[0.1,2,0.1]};
