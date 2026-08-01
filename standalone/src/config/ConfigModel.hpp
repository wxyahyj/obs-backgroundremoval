#pragma once

// 配置数据模型 — M2 新配置层核心。
// 纯数据结构 + 嵌套 JSON 序列化,不依赖任何引擎/OBS 代码。
// 字段对齐 filter_properties.cpp(OBS 键)与旧 standalone 配置,全量覆盖。

#include "MouseControllerInterface.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <string>
#include <vector>

namespace ya {
namespace config {

// ---- 截图节 ----
struct CaptureSection {
    std::string backend = "dxgi";      // dxgi | wgc | gdi
    std::string mode = "center";       // center=屏幕中心裁剪 | region=固定区域 | full=全屏
    bool use_region = true;
    int region_x = 0;
    int region_y = 0;
    int region_width = 640;
    int region_height = 640;
    int width = 640;                   // center 模式裁剪尺寸
    int height = 640;
};

// ---- 推理节 ----
struct InferSection {
    bool enabled = true;
    std::string model_path;            // 相对 exe 目录解析
    std::string device = "cuda";       // cpu | cuda | tensorrt | dml | rocm
    int model_version = 2;             // 0=v5 1=v8 2=v11
    float confidence = 0.45f;
    float nms = 0.45f;
    int input_resolution = 640;
    int num_threads = 4;
    int interval_frames = 1;           // 每 N 帧推理一次
    std::vector<int> target_classes;   // 空 = 全部
};

// ---- 跟踪节 ----
struct TrackerSection {
    float iou_threshold = 0.3f;
    int max_lost_frames = 30;
    int max_reidentify_frames = 45;
    float reidentify_center_threshold = 0.08f;
    float weight_iou = 0.5f;
    float weight_center = 0.3f;
    float weight_aspect = 0.1f;
    float weight_area = 0.1f;
    bool use_kalman = false;
    int kalman_generate_threshold = 3;
    int kalman_terminate_count = 5;
    int kalman_prediction_frames = 5;
    bool show_kalman_predictions = false;
    bool show_kalman_trajectories = false;
};

// ---- 显示节 ----
struct VisionSection {
    bool show_detection_results = true;
    int bbox_line_width = 2;
    float label_font_scale = 0.5f;
    bool export_coordinates = false;
    std::string coordinate_output_path;  // 空 = exe 目录 detections.json
    bool show_floating_window = false;
    int floating_window_width = 640;
    int floating_window_height = 480;
    bool show_track_id_in_floating_window = false;
    bool preview_enabled = true;
};

// ---- 瞄准节 ----
struct AimSlot {
    bool enabled = false;
    bool continuous_aim = false;
    MouseControllerConfig mc;
};

struct AimSection {
    bool enabled = true;
    int config_select = 0;             // UI 选中槽;运行时由热键/持续切换
    AlgorithmType algorithm = AlgorithmType::AdvancedPID;

    // FOV
    int fov_radius = 120;
    bool show_fov = true;
    bool show_fov_circle = true;
    bool show_fov_cross = false;
    int fov_cross_line_scale = 100;
    int fov_cross_line_thickness = 1;
    int fov_circle_thickness = 2;
    int fov_color = 0x00FF00;
    bool use_dynamic_fov = false;
    bool show_fov2 = false;
    int fov_radius2 = 80;
    int fov_color2 = 0x00FFFF;
    float dynamic_fov_shrink_percent = 0.7f;
    float dynamic_fov_transition_ms = 200.f;

    // 目标切换
    int target_switch_delay_ms = 500;
    float target_switch_tolerance = 0.15f;

    // 外部 PID(全局,作用于每个槽)
    float external_kp_x = 1.5f, external_ki_x = 0.f, external_kd_x = 1.5f;
    float external_kp_y = 1.5f, external_ki_y = 0.f, external_kd_y = 1.5f;
    float external_predict_x = 1.f, external_predict_y = 1.f;
    float external_rate_x = 0.3f, external_rate_y = 0.3f;
    float external_ki_mode = 1.f;
    float external_kp_limit = 9900.f, external_ki_limit = 9900.f, external_kd_limit = 9900.f;
    float external_output_limit = 0.f, external_ki_rate = 0.05f, external_ki_deadband = 0.5f;

    // Aim 控制器
    float aim_kp = 0.6f, aim_ki = 0.01f, aim_kd = 0.007f;
    bool aim_noise_enabled = false;
    float aim_noise_amplitude = 2.f;
    float aim_prediction_weight_x = 0.3f, aim_prediction_weight_y = 0.1f;
    float aim_ramp_time = 0.3f, aim_init_scale = 0.6f, aim_output_max = 128.f;

    // 神经路径
    bool enable_neural_path = false;
    int neural_path_points = 25;
    double neural_mouse_step_size = 8.0;
    int neural_target_radius = 8;
    int neural_consume_per_frame = 2;
    bool enable_neural_path_debug = false;

    // 准星(aim origin)
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

    static constexpr int kSlots = 5;
    std::array<AimSlot, kSlots> slots{};
};

// ---- 顶层文档 ----
struct ConfigDocument {
    CaptureSection capture;
    InferSection infer;
    TrackerSection tracker;
    VisionSection vision;
    AimSection aim;

    // 默认配置:槽 0 开启(开箱即用,镜像旧 standalone 行为)
    ConfigDocument()
    {
        auto& s0 = aim.slots[0];
        s0.enabled = true;
        s0.mc.enableMouseControl = true;
        s0.mc.hotkeyVirtualKey = 0x02; // RMB
        s0.mc.controllerType = ControllerType::WindowsAPI;
        s0.mc.fovRadiusPixels = 120;
        s0.mc.algorithmType = AlgorithmType::AdvancedPID;
        s0.mc.pidPMin = 0.153f;
        s0.mc.pidPMax = 0.6f;
        s0.mc.pidPSlope = 1.0f;
        s0.mc.pidD = 0.007f;
        s0.mc.pidI = 0.01f;
        s0.mc.deadZonePixels = 5.f;
        s0.mc.maxPixelMove = 128.f;
        s0.mc.makcuPort = "COM5";
        s0.mc.makcuBaudRate = 4000000;
        s0.mc.useDerivativePredictor = true;
        s0.mc.predictionWeightX = 0.5f;
        s0.mc.predictionWeightY = 0.1f;

        for (int i = 1; i < AimSection::kSlots; ++i) {
            aim.slots[i].mc.hotkeyVirtualKey = 0x05; // XBUTTON1
            aim.slots[i].mc.makcuPort = "COM5";
            aim.slots[i].mc.makcuBaudRate = 4000000;
            aim.slots[i].mc.pidPMin = 0.153f;
            aim.slots[i].mc.pidPMax = 0.6f;
            aim.slots[i].mc.useDerivativePredictor = true;
        }
    }
};

// ---- 序列化 ----
// 嵌套 JSON 结构。merge 支持扁平 OBS 键 patch(见 ConfigKeyMap)。
nlohmann::json capture_to_json(const CaptureSection& s);
nlohmann::json infer_to_json(const InferSection& s);
nlohmann::json tracker_to_json(const TrackerSection& s);
nlohmann::json vision_to_json(const VisionSection& s);
nlohmann::json aim_to_json(const AimSection& s);
nlohmann::json slot_to_json(const AimSlot& s, int index);
nlohmann::json document_to_json(const ConfigDocument& d);

// from_json:每个函数返回 false 表示类型不匹配(字段保留默认)。
bool capture_from_json(CaptureSection& s, const nlohmann::json& j);
bool infer_from_json(InferSection& s, const nlohmann::json& j);
bool tracker_from_json(TrackerSection& s, const nlohmann::json& j);
bool vision_from_json(VisionSection& s, const nlohmann::json& j);
bool aim_from_json(AimSection& s, const nlohmann::json& j);
bool slot_from_json(AimSlot& s, const nlohmann::json& j, int index);
bool document_from_json(ConfigDocument& d, const nlohmann::json& j);

} // namespace config
} // namespace ya
